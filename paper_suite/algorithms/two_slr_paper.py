'Paper-aligned two-round ULDP sparse linear regression (Ma, Jia & Yang, ICML 2024).'
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, asdict

import numpy as np
from dp_safety.rr import rr_keep_probability, rr_debias_factor
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import hadamard
from sklearn.linear_model import Lasso, LassoCV, LinearRegression

from ..vendor.ma2024.ULDPFS.heavy_hitters import HeavyHitters


_LOCK = threading.Lock()
PROTOCOL = "ma24_algorithm1_alg7to9_paper_aligned_v4_hh_alpha0p05_exp5_context"
PAPER_SCREENED_FEATURES = 64
PAPER_LOCAL_SELECTED_GRID = (2, 4, 8, 16)
PAPER_B_GRID = (1.0, 2.0, 3.0)
PAPER_BIN_GRID = (2, 4, 8, 16, 32)
FROZEN_HH_ALPHA = 0.05


@dataclass(frozen=True)
class TwoSLRProfile:


    local_selected: int
    hh_target: int
    screening_features: int = PAPER_SCREENED_FEATURES
    B: float = 1.0
    num_bins: int = 4
    hh_alpha: float = FROZEN_HH_ALPHA

    def validate(self, d: int) -> None:
        if self.local_selected < 1 or self.hh_target < 1:
            raise ValueError("positive selector/HH sizes required")
        if self.screening_features < 1 or d < 1:
            raise ValueError("invalid screening dimension")
        if not (0.0 < self.hh_alpha < 1.0):
            raise ValueError("HeavyHitter alpha must be in (0,1)")
        if abs(self.hh_alpha - FROZEN_HH_ALPHA) > 1e-15:
            raise ValueError("formal 2-SLR runner freezes HeavyHitter alpha at 0.05")
        if self.B not in PAPER_B_GRID or self.num_bins not in PAPER_BIN_GRID:
            raise ValueError("2-SLR B/bins must stay on the Appendix-E.1 grid")
        if self.num_bins & (self.num_bins - 1):
            raise ValueError("Hadamard range estimator requires power-of-two bins")


def reference_profile(k: int, d: int) -> TwoSLRProfile:
    'Public paper-grid reference, without test-truth tuning.'
    p = TwoSLRProfile(local_selected=min(2,d), hh_target=max(1, k),
                      screening_features=min(PAPER_SCREENED_FEATURES, d),
                      B=1.0, num_bins=4)
    p.validate(d)
    return p


def paper_parameter_grid(k: int, d: int) -> list[TwoSLRProfile]:
    'Appendix-E.1 experimental grid, represented as executable profiles.'
    profiles=[]
    for selected in PAPER_LOCAL_SELECTED_GRID:
        if selected > d:
            continue
        for B in PAPER_B_GRID:
            for bins in PAPER_BIN_GRID:
                profile=TwoSLRProfile(
                    local_selected=int(selected),
                    hh_target=max(1, min(int(k), int(d))),
                    screening_features=min(PAPER_SCREENED_FEATURES, int(d)),
                    B=float(B),
                    num_bins=int(bins),
                )
                profile.validate(d)
                profiles.append(profile)
    return profiles


def _screen_indices(X: np.ndarray, y: np.ndarray, count: int) -> np.ndarray:
    'Appendix-E feature screening using the paper/author positive marginal correlation.'
    d = X.shape[1]
    count = min(max(1, int(count)), d)
    corr = np.asarray(y @ X, dtype=float).ravel()
    if count == d:
        return np.arange(d, dtype=int)


    order = np.argsort(-corr, kind="mergesort")
    return np.asarray(order[:count], dtype=int)


def _lasso_candidates(X: np.ndarray, y: np.ndarray, target: int,
                      screening_features: int) -> np.ndarray:
    'Appendix-E Lasso-after-screening selector, following author tuning logic.'
    screened = _screen_indices(X, y, screening_features)
    Xs = X[:, screened]
    target = min(max(1, int(target)), Xs.shape[1])


    clf = LassoCV().fit(Xs, y)
    alpha = float(clf.alpha_)
    coef = np.asarray(clf.coef_, float)
    for _ in range(15):
        nz = np.flatnonzero(np.abs(coef) > 1e-12)
        if len(nz) < target and len(nz) < Xs.shape[1]:
            alpha /= 2.0
        elif len(nz) > 2 * target:
            alpha *= 1.5
        else:
            break
        clf = Lasso(alpha=alpha).fit(Xs, y)
        coef = np.asarray(clf.coef_, float)

    nz = np.flatnonzero(np.abs(coef) > 1e-12)
    if len(nz) == 0:


        corr = np.asarray(y @ Xs, dtype=float).ravel()
        nz = np.array([int(np.argmax(corr))])
    return np.asarray(screened[nz], dtype=int)


def _local_votes(x: np.ndarray, y: np.ndarray, profile: TwoSLRProfile,
                 rng: np.random.Generator) -> tuple[list[int], dict]:
    votes: list[int] = []
    sizes = []; warning_count=0; failure_count=0
    for Xi, yi in zip(x, y):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            try:
                candidates = _lasso_candidates(Xi, yi, profile.local_selected,
                                               profile.screening_features)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):

                candidates = np.array([0],dtype=int);failure_count+=1
        warning_count+=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
        sizes.append(len(candidates))
        votes.append(int(rng.choice(candidates)))
    return votes, {
        'local_lasso_warning_count':warning_count,
        'local_lasso_totalization_count':failure_count,
        "local_candidate_size_mean": float(np.mean(sizes)),
        "local_candidate_size_min": int(np.min(sizes)),
        "local_candidate_size_max": int(np.max(sizes)),
    }


class _OneReportHeavyHitters(HeavyHitters):
    'Author Appendix-E HH implementation with one RR report per user.'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paper_report_cache: dict[tuple[int, bool], float] = {}
        self.report_draw_count = np.zeros(self.n, dtype=int)

    def LocalRnd(self, i, final):  # noqa: N802 - upstream API name


        key = (int(i), bool(final))
        if key not in self._paper_report_cache:
            self._paper_report_cache[key] = super().LocalRnd(i, final)
            self.report_draw_count[int(i)] += 1
        return self._paper_report_cache[key]


def _select_support_from_votes(votes: list[int], d: int, epsilon: float,
                               profile: TwoSLRProfile, seed: int) -> tuple[np.ndarray, dict]:


    with _LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            hh = _OneReportHeavyHitters(
                epsilon=epsilon, d=d, user_values=votes,
                min_hitters=profile.hh_target, alpha=profile.hh_alpha,
                if_est_freq=False,
            )
            support, _ = hh.apply()
        finally:
            np.random.set_state(state)
    support = np.asarray(sorted(set(int(j) for j in support if 0 <= int(j) < d)), dtype=int)
    draws = hh.report_draw_count
    return support, {
        "hh_report_draws_max": int(draws.max(initial=0)),
        "hh_report_draws_mean": float(draws.mean()),
        "hh_one_report_per_user": bool(np.all(draws <= 1)),
        "hh_parameterization": "public-author experimental target-list heuristic; HeavyHitter alpha fixed at 0.05 before final evaluation because Appendix E.1 does not publish numerical rho",
        "hh_alpha": float(profile.hh_alpha),
    }


def select_support(x: np.ndarray, y: np.ndarray, epsilon: float, k: int,
                   seed: int, profile: TwoSLRProfile | None = None,
                   data_regime: str = "synthetic_positive") -> tuple[np.ndarray, dict]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.ndim != 3 or y.shape != x.shape[:2] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite aligned user data required")
    n, m, d = x.shape
    if n < 80 or m < 2 or d < 2 or not 1 <= k <= d or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("unsupported 2-SLR setting")
    profile = reference_profile(k, d) if profile is None else profile
    profile.validate(d)
    if data_regime not in ("synthetic_positive", "real_standardized"):
        raise ValueError("unknown 2-SLR data regime")
    n1 = n // 2
    rng = np.random.default_rng(seed)
    votes, local_meta = _local_votes(x[:n1], y[:n1], profile, rng)
    support, hh_meta = _select_support_from_votes(votes, d, epsilon, profile, seed + 17011)
    meta = {
        "support": support,
        "protocol_version": PROTOCOL,
        "implementation": "Ma24 Algorithm 1; Appendix-E Lasso+64-screening; one-report HeavyHitter",
        "parameters": asdict(profile),
        "paper_parameter_grid": {"local_selected": list(PAPER_LOCAL_SELECTED_GRID), "B": list(PAPER_B_GRID), "num_bins": list(PAPER_BIN_GRID), "reporting": "Appendix E.1: best average over >=30 replications; no validation/CV set for private-model tuning"},
        "selection_count": n1,
        "privacy_unit": "user-level LDP",
        "release_mode":"research_simulation", "private_release_safe":False,
        "rr_probability_policy":"conservative_53bit_floor",
        "privacy_validation": "epsilon-LDP one-report HH / composed Range / one-coordinate Mean; disjoint users; see PRIVACY_SCOPE.md",
        "local_rule": "Lasso nonzero support after 64-feature positive marginal screening; uniform one-index vote",
        "coefficient_threshold": None,
        "screening_adaptation": (
            "none; frozen synthetic active coefficients are positive"
            if data_regime == "synthetic_positive" else
            "none; Exp5 follows the Ma24 author real-data PostLassoSelector, which also uses positive marginal screening before local Lasso"
        ),
        "data_regime": data_regime,
        **local_meta, **hh_meta,
    }
    return support, meta


def _local_ols_coefficients(x: np.ndarray, y: np.ndarray, support: np.ndarray) -> np.ndarray:
    coeffs = []
    for Xi, yi in zip(x, y):
        Xs = Xi[:, support]


        try:
            with np.errstate(over='raise',invalid='raise'):
                coef = LinearRegression(fit_intercept=False).fit(Xs, yi).coef_
            if not np.isfinite(coef).all() or np.max(np.abs(coef),initial=0.)>1e150:
                raise FloatingPointError('nonfinite local coefficient')
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            coef=np.zeros(len(support))
        coeffs.append(np.asarray(coef, float).ravel())
    out = np.asarray(coeffs, float)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite local OLS coefficient")
    return out


def _next_power_of_two(s: int) -> int:
    return 1 if s <= 1 else 1 << (int(s - 1).bit_length())


def _range_private(values: np.ndarray, B: float, num_bins: int,
                   epsilon: float, rng: np.random.Generator) -> tuple[float, float, int]:
    'Algorithm 7 Range, generalized from [-1,1] to Appendix-E [-B,B].'
    values = np.asarray(values, float).ravel()
    if len(values) == 0 or epsilon <= 0:
        raise ValueError("Range requires users and positive epsilon")
    if num_bins & (num_bins - 1):
        raise ValueError("Range Hadamard bins must be power of two")
    tau = B / num_bins
    centers = -B + tau + 2.0 * tau * np.arange(num_bins)
    H = hadamard(num_bins)
    keep_probability = rr_keep_probability(epsilon)
    ae = rr_debias_factor(epsilon)
    freq_vec = np.zeros(num_bins, float)

    nearest = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    for v in nearest:
        j = int(rng.integers(num_bins))
        row = H[j].astype(float)
        p_plus = keep_probability if row[int(v)] > 0 else 1.0 - keep_probability
        sign = 1.0 if rng.random() < p_plus else -1.0
        freq_vec += sign * row * ae
    ell = int(np.argmax(freq_vec))
    center = float(centers[ell])
    return center - 3.0 * tau, center + 3.0 * tau, ell


def _mean_private(values: np.ndarray, interval: tuple[float, float], epsilon: float,
                  rng: np.random.Generator) -> float:
    'Algorithm 8 Mean: project first, then add Laplace noise.'
    values = np.asarray(values, float).ravel()
    if len(values) == 0 or epsilon <= 0:
        raise ValueError("Mean requires users and positive epsilon")
    a, b = map(float, interval)
    if not a < b:
        raise ValueError("invalid range interval")
    projected = np.clip(values, a, b)
    noisy = projected + rng.laplace(0.0, (b - a) / epsilon, size=len(values))
    return float(np.mean(noisy))


def _uldp_mean(beta: np.ndarray, epsilon: float, B: float, num_bins: int,
               rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    'Algorithms 7--9 with the two disjoint coefficient-estimation groups.'
    beta = np.asarray(beta, float)
    if beta.ndim != 2 or len(beta) < 4:
        raise ValueError("ULDPMean needs a coefficient matrix and enough users")
    n, s = beta.shape
    n_range = n // 2
    n_mean = n - n_range
    if n_range < 1 or n_mean < 1:
        raise ValueError("two nonempty Stage-2 groups required")


    K = _next_power_of_two(s)
    padded = np.pad(beta, ((0, 0), (0, K - s)))
    D = rng.choice(np.array([-1.0, 1.0]), size=K)
    H = hadamard(K).astype(float)
    U = (H * D[None, :]) / math.sqrt(K)

    rotated = padded @ U.T

    first = rotated[:n_range]
    second = rotated[n_range:]
    intervals: list[tuple[float, float]] = []
    range_bins = []
    for ell in range(K):
        interval = _range_private(first[:, ell], B, num_bins, epsilon / K, rng)
        intervals.append((interval[0], interval[1]))
        range_bins.append(interval[2])

    z = np.zeros(K, float)
    assignment_counts = np.zeros(K, int)
    projection_counts = np.zeros(K, int)
    for ell in range(K):
        ids = np.arange(n_mean)[np.arange(n_mean) % K == ell]
        if len(ids) == 0:
            raise ValueError("not enough Stage-2 mean users for all rotated coordinates")
        vals = second[ids, ell]
        a, b = intervals[ell]
        assignment_counts[ell] = len(vals)
        projection_counts[ell] = int(np.sum((vals < a) | (vals > b)))


        z[ell] = _mean_private(vals, (a, b), epsilon, rng)


    mean_padded = z @ U
    out = np.asarray(mean_padded[:s], float)
    return out, {
        "stage2_range_users": int(n_range),
        "stage2_mean_users": int(n_mean),
        "stage2_groups_disjoint": True,
        "rotation_dimension": int(K),
        "range_budget_per_coordinate": float(epsilon / K),
        "mean_budget_per_user": float(epsilon),
        "mean_assignment_min": int(assignment_counts.min()),
        "mean_assignment_max": int(assignment_counts.max()),
        "mean_projection_rate": float(projection_counts.sum() / max(n_mean, 1)),
        "range_selected_bins": range_bins,
        "tau": float(B / num_bins),
        "algorithm9_normalization": "Lemma C.4/proof and public-code subgroup mean; displayed Algorithm 9 has an inconsistent extra s multiplier",
    }


def fit(x: np.ndarray, y: np.ndarray, epsilon: float, k: int, seed: int,
        profile: TwoSLRProfile | None = None,
        data_regime: str = "synthetic_positive") -> tuple[np.ndarray, dict]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    n, m, d = x.shape if x.ndim == 3 else (0, 0, 0)
    if y.shape != (n, m) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite aligned user data required")
    profile = reference_profile(k, d) if profile is None else profile
    profile.validate(d)
    support, sel_meta = select_support(x, y, epsilon, k, seed, profile, data_regime=data_regime)
    n1 = n // 2
    if len(support)==0:
        return np.zeros(d),{**sel_meta,'algorithm_status':'empty_global_support'}
    if _next_power_of_two(len(support)) > (n-n1)-(n-n1)//2:

        return np.zeros(d),{**sel_meta,'algorithm_status':'public_group_size_insufficient_for_selected_support'}
    local_beta = _local_ols_coefficients(x[n1:], y[n1:], support)
    rng = np.random.default_rng(seed + 35023)
    low_beta, mean_meta = _uldp_mean(local_beta, epsilon, profile.B, profile.num_bins, rng)
    beta = np.zeros(d, float)
    beta[support] = low_beta
    if not np.isfinite(beta).all():
        raise RuntimeError("2-SLR produced non-finite coefficients")
    return beta, {
        **sel_meta,
        **mean_meta,
        "support": support,
        "implementation": "Ma24 Algorithm 1 paper-aligned; Appendix-E post-Lasso; corrected Algorithms 7-9 semantics",
        "privacy_validation": "epsilon-LDP mechanisms with conservative RR; disjoint groups; see PRIVACY_SCOPE.md",
        "stage2_local_estimator": "OLS fit_intercept=False on selected coordinates",
    }
