from __future__ import annotations

import math
import numpy as np

from .common import hard_threshold, privunit, project_l2, privunit_batch


ZHU24_SOURCE = "Zhu et al., ICLR 2024, Algorithm 2 / Theorem 7"


def _topk_indices(vector: np.ndarray, k: int) -> np.ndarray:
    vector = np.asarray(vector, float)
    keep = min(max(int(k), 0), len(vector))
    if keep == 0:
        return np.empty(0, dtype=int)
    if keep == len(vector):
        return np.arange(len(vector), dtype=int)
    return np.sort(np.argpartition(np.abs(vector), -keep)[-keep:]).astype(int)


def paper_reference_parameters(participant_count: int, k: int, *, sigma_scale: float = 1.0,
                               log_multiplier: float = 1.0,
                               clip_multiplier: float = 1.0,
                               step_size: float = 1.0) -> dict:
    'Concrete Theorem-7-order reference profile for Zhu et al. Algorithm 2.'
    if (not isinstance(participant_count, int) or participant_count < 2 or
        not isinstance(k, int) or k < 1 or
        not all(math.isfinite(v) and v > 0 for v in
                (sigma_scale, log_multiplier, clip_multiplier, step_size))):
        raise ValueError("invalid paper-reference LDP-IHT parameters")
    logn = math.log(participant_count)
    tau = clip_multiplier * sigma_scale * math.sqrt(logn)
    iterations = max(1, int(math.ceil(log_multiplier * logn)))
    return dict(iterations=iterations, step_size=float(step_size),
                x_clip=float(tau), y_clip=float(tau), sparsity=8 * k,
                vectorized=True)


def ma24_empirical_parameter_grid() -> list[dict]:
    'Empirical LDPIHT grid reported by Ma et al. (ICML 2024, Appendix E.1).'
    configs=[]
    for iterations in (2,5,10,20,50):
        for step_size in (0.01,0.1,1.0):
            for x_clip in (2.0,4.0,8.0):
                for y_clip in (2.0,4.0,8.0):
                    for sparsity in (5,10,20,50):
                        configs.append(dict(iterations=iterations, step_size=step_size,
                                            x_clip=x_clip, y_clip=y_clip, sparsity=sparsity,
                                            vectorized=True))
    return configs


def qualification_parameter_grid() -> list[dict]:
    'Small optional LDP-IHT parameter-grid check.'
    tuples = (
        (2, 0.01, 10),
        (5, 0.01, 10),
        (10, 0.01, 10),
        (20, 0.01, 10),
        (2, 0.1, 10),
        (10, 0.1, 10),
        (20, 0.1, 10),
        (2, 0.01, 5),
    )
    grid = [
        dict(iterations=T, step_size=eta, x_clip=2.0, y_clip=2.0,
             sparsity=kprime, vectorized=True)
        for T, eta, kprime in tuples
    ]
    reported = ma24_empirical_parameter_grid()
    if not all(cfg in reported for cfg in grid):
        raise AssertionError("qualification grid must stay inside the Ma24 reported grid")
    return grid


def paper_parameter_grid(participant_count: int, k: int, *, sigma_scale: float = 1.0) -> list[dict]:
    'Reference parameter grid constrained to Theorem-7 orders.'
    configs = []
    for log_multiplier in (0.5, 1.0, 2.0):
        for step_size in (0.5, 1.0):
            for clip_multiplier in (0.5, 1.0, 2.0):
                cfg = paper_reference_parameters(
                    participant_count, k, sigma_scale=sigma_scale,
                    log_multiplier=log_multiplier,
                    clip_multiplier=clip_multiplier,
                    step_size=step_size,
                )
                cfg.update(theorem_log_multiplier=log_multiplier,
                           theorem_clip_multiplier=clip_multiplier,
                           theorem_sigma_scale=sigma_scale)
                configs.append(cfg)
    return configs


def reference_parameters(k: int, participant_count: int = 40000) -> dict:
    'Backward-compatible alias for the paper-reference Theorem-7 profile.'
    return paper_reference_parameters(participant_count, k)


def fit_ldp_iht(x: np.ndarray, y: np.ndarray, epsilon: float, k: int, seed: int,
                iterations: int = 5, step_size: float = 0.1,
                x_clip: float = 4.0, y_clip: float = 4.0,
                threshold_multiplier: int = 8, sparsity: int | None = None,
                private: bool = True, vectorized: bool = False,
                gradient_clip: float | None = None, radius_mode: str = "paper_constant") -> tuple[np.ndarray, dict]:
    'Sequentially interactive sample-level LDP-IHT (Zhu et al., Algorithm 2).'


    if radius_mode not in ("paper_constant", "public_iterate"):
        raise ValueError("invalid public radius policy")
    if gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError("gradient_clip must be public, finite and positive")
    x, y = np.asarray(x, float), np.asarray(y, float)
    if private is not True:
        raise ValueError("Frozen LDP-IHT requires private reports")
    if (not isinstance(iterations, int) or iterations < 1 or not isinstance(k, int) or k < 1 or
        not isinstance(threshold_multiplier, int) or threshold_multiplier < 1 or
        not all(math.isfinite(v) and v > 0 for v in (epsilon, step_size, x_clip, y_clip)) or
        not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid finite data or IHT parameters")
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < iterations:
        raise ValueError("expected at least one record per iteration")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cohorts = np.array_split(order, iterations)
    theta = np.zeros(x.shape[1])
    if sparsity is not None and (not isinstance(sparsity, int) or isinstance(sparsity, bool) or sparsity < 1):
        raise ValueError("sparsity must be a positive integer")
    requested_sparse_k = threshold_multiplier * k if sparsity is None else sparsity
    sparse_k = min(x.shape[1], requested_sparse_k)
    gradient_radius = math.sqrt(x.shape[1]) * x_clip * (math.sqrt(sparse_k) * x_clip + y_clip)


    def _clip_fraction_matrix(a: np.ndarray, threshold: float, chunk: int = 8192) -> float:
        count = 0
        total = a.size
        for start in range(0, len(a), chunk):
            block = a[start:start + chunk]
            count += int(np.count_nonzero(np.abs(block) > threshold))
        return count / total if total else 0.0

    x_clip_fraction = _clip_fraction_matrix(x, x_clip)
    y_clip_fraction = float(np.mean(np.abs(y) > y_clip))
    trace = []
    for cohort in cohorts:
        analytic_radius = (gradient_radius if radius_mode == "paper_constant" else
                           math.sqrt(x.shape[1]) * x_clip * (x_clip * float(np.abs(theta).sum()) + y_clip))
        report_radius = analytic_radius if gradient_clip is None else min(analytic_radius, gradient_clip)
        gradient_clipped_count = 0


        report_sum = np.zeros(x.shape[1], dtype=float)
        if vectorized:
            for start in range(0, len(cohort), 1024):
                ids = cohort[start:start + 1024]
                xx = np.clip(x[ids], -x_clip, x_clip)
                yy = np.clip(y[ids], -y_clip, y_clip)
                gradients = xx * (xx @ theta - yy)[:, None]
                norms = np.linalg.norm(gradients, axis=1)
                gradient_clipped_count += int(np.count_nonzero(norms > report_radius))
                gradients *= np.minimum(1.0, report_radius / np.maximum(norms, 1e-300))[:, None]
                reports = privunit_batch(gradients, report_radius, epsilon, rng)
                report_sum += reports.sum(axis=0)
        else:
            for i in cohort:
                xx = np.clip(x[i], -x_clip, x_clip)
                yy = float(np.clip(y[i], -y_clip, y_clip))
                gradient = xx * (xx @ theta - yy)


                gradient_clipped_count += int(np.linalg.norm(gradient) > report_radius)
                gradient = project_l2(gradient, report_radius)
                report_sum += privunit(gradient, report_radius, epsilon, rng)
        theta_update = theta - step_size * (report_sum / len(cohort))
        thresholded = hard_threshold(theta_update, sparse_k)
        trace.append({
            "analytic_radius": analytic_radius,
            "report_radius": report_radius,
            "gradient_clip_fraction": gradient_clipped_count / len(cohort),
            "update_norm": float(np.linalg.norm(theta_update)),
            "thresholded_norm": float(np.linalg.norm(thresholded)),
            "projection_active": bool(np.linalg.norm(thresholded) > 1.0),
        })
        theta = project_l2(thresholded, 1.0)
    return theta, {
        "privacy_unit": "one sample / flattened row",
        "private": private,
        "epsilon_ldp_target":float(epsilon),
        "delta_ldp_target":0.0,
        "rr_probability_policy":"conservative_53bit_floor",
        "privacy_validation":"epsilon-LDP vector channel; disjoint row cohorts; see PRIVACY_SCOPE.md",
        "release_mode":"research_simulation",
        "private_release_safe":False,
        "paper_source": ZHU24_SOURCE,
        "iterations": iterations,
        "gradient_radius": max(t["report_radius"] for t in trace),
        "paper_constant_radius": gradient_radius,
        "radius_mode": radius_mode,
        "gradient_clip": gradient_clip,
        "implementation": ("LDPIHT gradient-clipped stabilization" if gradient_clip is not None else "Zhu24 Algorithm 2"),
        "original_utility_theorem_claimed": False,
        "gradient_unbiased_for": "clipped gradient" if gradient_clip is not None else "coordinate-truncated gradient",
        "support": np.flatnonzero(np.abs(theta) > 1e-8),
        "top_k_support": _topk_indices(theta, k),
        "step_size": step_size,
        "x_clip": x_clip,
        "y_clip": y_clip,
        "hard_threshold_k": sparse_k,
        "requested_hard_threshold_k": requested_sparse_k,
        "theorem7_kprime_equals_8k": bool(requested_sparse_k == 8 * k),
        "l2_radius": 1.0,
        "participant_count": len(x),
        "cohort_sizes": [len(c) for c in cohorts],
        "iteration_diagnostics": trace,
        "projection_rate": float(np.mean([r["projection_active"] for r in trace])),
        "vectorized": vectorized,
        "x_clip_fraction": float(x_clip_fraction),
        "y_clip_fraction": float(y_clip_fraction),
        "parameter_status": "explicit caller parameters; paper-reference profile available separately",
        "protocol_version": ("ldpiht_gradient_clipped_stabilization_v1" if gradient_clip is not None else
                             "ldpiht_zhu24_algorithm2_v1"),
    }
