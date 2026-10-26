'Instrumented LDPIHT; synthetic diagnostics, NOT a private release service.'
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from paper_suite.algorithms.common import hard_threshold, project_l2, privunit_batch
from dp_safety.rr import rr_debias_factor


@dataclass(frozen=True)
class Config:
    name: str
    mode: str = "paper_constant"
    iterations: int = 2
    step_size: float = .01
    x_clip: float = 2.
    y_clip: float = 2.
    sparsity: int = 10
    gradient_clip: float | None = None

    def validate(self, d: int, count: int):
        if self.mode not in {"paper_constant", "public_iterate", "nonprivate"}:
            raise ValueError("unknown mode")
        for value in (self.iterations, self.sparsity):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("positive integer required")
        if self.iterations > count or self.sparsity > d:
            raise ValueError("cohort/support dimensions invalid")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.step_size, self.x_clip, self.y_clip)):
            raise ValueError("positive finite parameters required")
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError("gradient clip must be public, finite, and positive")


def radius_for(cfg: Config, theta: np.ndarray) -> float:
    'Bound valid for EVERY clipped record, conditional on broadcast theta.'
    if np.linalg.norm(theta) > 1 + 1e-12 or np.count_nonzero(theta) > cfg.sparsity:
        raise ValueError("iterate invariant broken")
    norm_bound = (float(np.abs(theta).sum()) if cfg.mode == "public_iterate"
                  else math.sqrt(cfg.sparsity))
    return math.sqrt(len(theta)) * cfg.x_clip * (cfg.x_clip * norm_bound + cfg.y_clip)


def report_scale(d: int, radius: float, epsilon: float) -> float:
    c = math.exp(math.lgamma(d / 2) - .5 * math.log(math.pi) - math.lgamma((d + 1) / 2))
    return radius * rr_debias_factor(epsilon) / c


def fit(x, y, epsilon: float, seed: int, cfg: Config, *, private: bool = True):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] < 1:
        raise ValueError("invalid data shape")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive epsilon required")
    cfg.validate(x.shape[1], len(x))
    if not isinstance(private, bool):
        raise ValueError("private must be boolean")
    private = private and cfg.mode != "nonprivate"
    rng = np.random.default_rng(seed)
    cohorts = np.array_split(rng.permutation(len(x)), cfg.iterations)
    theta = np.zeros(x.shape[1])
    trace = []
    for cohort in cohorts:
        legal_radius = radius_for(cfg, theta)
        radius = legal_radius if cfg.gradient_clip is None else min(legal_radius, cfg.gradient_clip)
        report_sum, gradient_sum, gradient_squares = (np.zeros(x.shape[1]) for _ in range(3))
        unclipped_sum = np.zeros(x.shape[1])
        max_norm = 0.
        max_sent_norm = 0.
        clipped_gradients = 0
        x_clipped = y_clipped = 0
        previous = theta.copy()
        for start in range(0, len(cohort), 1024):
            ids = cohort[start:start + 1024]
            raw_x, raw_y = x[ids], y[ids]
            if not np.isfinite(raw_x).all() or not np.isfinite(raw_y).all():
                raise ValueError("nonfinite data")
            x_clipped += int(np.count_nonzero(np.abs(raw_x) > cfg.x_clip))
            y_clipped += int(np.count_nonzero(np.abs(raw_y) > cfg.y_clip))
            xx, yy = np.clip(raw_x, -cfg.x_clip, cfg.x_clip), np.clip(raw_y, -cfg.y_clip, cfg.y_clip)
            gradients = xx * (xx @ theta - yy)[:, None]
            norms = np.linalg.norm(gradients, axis=1)
            max_norm = max(max_norm, float(norms.max()))
            if np.any(norms > legal_radius * (1 + 1e-12)):
                raise AssertionError("declared gradient bound violated")
            unclipped_sum += gradients.sum(axis=0)
            clipped_gradients += int(np.count_nonzero(norms > radius))
            gradients *= np.minimum(1., radius / np.maximum(norms, 1e-300))[:, None]
            sent_norms = np.linalg.norm(gradients, axis=1)
            max_sent_norm = max(max_sent_norm, float(sent_norms.max()))
            if np.any(sent_norms > radius * (1 + 1e-12)):
                raise AssertionError("post-clipping randomizer bound violated")
            gradient_sum += gradients.sum(axis=0)
            gradient_squares += (gradients ** 2).sum(axis=0)
            reports = (gradients if not private else
                       privunit_batch(gradients, radius, epsilon, rng))
            report_sum += reports.sum(axis=0)
        mean = report_sum / len(cohort)
        raw_mean = gradient_sum / len(cohort)
        update = theta - cfg.step_size * mean
        thresholded = hard_threshold(update, cfg.sparsity)
        theta = project_l2(thresholded, 1.)
        if private:
            B = report_scale(x.shape[1], radius, epsilon)

            variance = (B * B / x.shape[1] - gradient_squares / len(cohort)) / len(cohort)
            if variance.min() < -1e-12:
                raise AssertionError("invalid theoretical variance")
            expected_sd = float(np.sqrt(np.maximum(variance, 0.)).mean())
        else:
            expected_sd = 0.
        trace.append(dict(iteration=len(trace) + 1, cohort_count=len(cohort),
                          legal_unclipped_radius=legal_radius, gradient_radius=radius,
                          max_actual_gradient_norm=max_norm, max_sent_gradient_norm=max_sent_norm,
                          gradient_bound_ratio=max_sent_norm / radius,
                          gradient_clip_fraction=clipped_gradients / len(cohort),
                          gradient_clipping_mean_bias_l2=float(np.linalg.norm((gradient_sum-unclipped_sum)/len(cohort))),
                          x_clip_fraction=x_clipped / (len(cohort)*x.shape[1]),
                          y_clip_fraction=y_clipped / len(cohort),
                          private_noise_coordinate_rms=float(np.sqrt(np.mean((mean-raw_mean)**2))),
                          expected_noise_coordinate_sd=expected_sd,
                          projection_active=bool(np.linalg.norm(thresholded) > 1.),
                          theta_before=previous.tolist(), theta_after=theta.tolist(),
                          unclipped_gradient_mean=(unclipped_sum/len(cohort)).tolist(),
                          raw_gradient_mean=raw_mean.tolist(), report_mean=mean.tolist()))
    return theta, dict(config=asdict(cfg), trace=trace, epsilon=epsilon,
                       private=private,
                       delta=0. if private else None,
                       privacy_unit="record" if private else "none",
                       private_release_safe=False,
                       warning="Raw diagnostics and seeded random coins are research-only; do not publish for protected data.")
