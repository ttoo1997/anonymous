'SDP1 original flow, changing only local selection steps 1/2 to Lasso.'

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


MAX_LOCAL_INTEGER = 2**52


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(name: str, value: float, *, positive: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


@dataclass(frozen=True)
class StageBudget:
    epsilon: float
    delta: float

    def validate(self) -> None:
        _real("epsilon", self.epsilon)
        if _real("delta", self.delta) >= 1:
            raise ValueError("delta must be less than 1")


@dataclass(frozen=True)
class OriginalLassoConfig:

    n_selection: int
    n_coarse: int
    n_fine: int
    lasso_alpha: float
    hh_threshold: float
    coarse_radius: float
    tau_beta: float
    noise_sigma: float
    c1: float
    c2: float
    precision_constant: float
    noise_constant: float
    selection_budget: StageBudget
    coarse_budget: StageBudget
    fine_budget: StageBudget
    seed: int = 20260831
    parameter_table_mode: bool = False
    revised_stage1: bool = False
    simulation_skip_ikos: bool = False
    tau_beta_constant: float = 1.0
    lasso_fit_tol: float = 1e-5
    lasso_zero_tol: float = 1e-8
    lasso_max_iter: int = 5000
    lasso_retry_max_iter: int = 20000
    ols_rcond: float = 1e-10

    @classmethod
    def from_dict(cls, payload: dict) -> "OriginalLassoConfig":
        values = dict(payload)
        for key in ("selection_budget", "coarse_budget", "fine_budget"):
            values[key] = StageBudget(**values[key])
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        _integer("n_selection", self.n_selection, 1)
        _integer("n_coarse", self.n_coarse, 19)
        _integer("n_fine", self.n_fine, 19)
        _integer("seed", self.seed)
        if not isinstance(self.parameter_table_mode, (bool, np.bool_)):
            raise ValueError("parameter_table_mode must be boolean")
        if not isinstance(self.revised_stage1, (bool, np.bool_)):
            raise ValueError("revised_stage1 must be boolean")
        if not isinstance(self.simulation_skip_ikos, (bool, np.bool_)):
            raise ValueError("simulation_skip_ikos must be boolean")
        for name in ("lasso_alpha", "coarse_radius", "c1", "c2", "precision_constant",
                     "noise_constant", "lasso_fit_tol"):
            _real(name, getattr(self, name))
        for name in ("hh_threshold", "tau_beta", "noise_sigma", "lasso_zero_tol", "ols_rcond"):
            _real(name, getattr(self, name), positive=False)
        _real("tau_beta_constant", self.tau_beta_constant)
        if self.ols_rcond >= 1:
            raise ValueError("ols_rcond must be less than 1")
        _integer("lasso_max_iter", self.lasso_max_iter, 1)
        _integer("lasso_retry_max_iter", self.lasso_retry_max_iter, self.lasso_max_iter)
        for budget in (self.selection_budget, self.coarse_budget, self.fine_budget):
            if not isinstance(budget, StageBudget):
                raise ValueError("budgets must be StageBudget instances")
            budget.validate()
        rr_flip_probability(self.n_selection, self.selection_budget, self.revised_stage1)


@dataclass(frozen=True)
class UserPartition:
    selection: tuple[int, ...]
    coarse: tuple[int, ...]
    fine: tuple[int, ...]

    def validate(self, n: int, config: OriginalLassoConfig) -> None:
        groups = (self.selection, self.coarse, self.fine)
        sizes = (config.n_selection, config.n_coarse, config.n_fine)
        if tuple(map(len, groups)) != sizes:
            raise ValueError("partition sizes disagree with config")
        indices = [_integer("user index", i) for group in groups for i in group]
        if len(indices) != n or sorted(indices) != list(range(n)):
            raise ValueError("partition must contain every input user exactly once")


@dataclass(frozen=True)
class RandomStreams:
    seed: int
    transport_seed: int | None = None

    def rng(self, name: str, *, transport: bool = False) -> np.random.Generator:
        seed = self.transport_seed if transport and self.transport_seed is not None else self.seed
        _integer("random seed", seed)
        prefix = "transport/" if transport else "local/"
        label = (prefix + name).encode("utf-8")
        words = [int.from_bytes(label[i:i+4].ljust(4, b"\0"), "little")
                 for i in range(0, len(label), 4)]
        return np.random.default_rng(np.random.SeedSequence([int(seed), len(label), *words]))


def make_partition(n: int, config: OriginalLassoConfig) -> UserPartition:
    if n != config.n_selection + config.n_coarse + config.n_fine:
        raise ValueError("user counts must sum to input n")
    order = RandomStreams(config.seed).rng("partition").permutation(n).tolist()
    end = config.n_selection + config.n_coarse
    return UserPartition(tuple(order[:config.n_selection]), tuple(order[config.n_selection:end]),
                         tuple(order[end:]))


def _validate_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError("X and y must be real-valued")
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 2 or y.shape != x.shape[:2] or min(x.shape) < 1:
        raise ValueError("expected nonempty X[n,m,d] and y[n,m]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite; preprocessing belongs outside this algorithm")
    return x, y


@dataclass(frozen=True)
class LocalLassoFit:
    'Threshold-independent stage-1 fits reusable across tau_HH variants.'
    coefficients: np.ndarray
    attempts: tuple[dict, ...]
    signature: tuple


def _local_lasso_signature(x: np.ndarray, config: OriginalLassoConfig) -> tuple:
    return (tuple(x.shape), config.n_selection, bool(config.revised_stage1),
            float(config.lasso_alpha), float(config.lasso_fit_tol),
            int(config.lasso_max_iter), int(config.lasso_retry_max_iter),
            float(config.ols_rcond))


def fit_local_lasso_models(x: np.ndarray, y: np.ndarray,
                           config: OriginalLassoConfig) -> LocalLassoFit:
    'Fit stage-1 Lasso models once; tau_HH is intentionally not an input.'
    x, y = _validate_data(x, y)
    config.validate()
    coefficients = np.zeros((len(x), x.shape[2]), dtype=float)
    attempts = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        if not np.any(xi) or not np.any(yi):
            attempts.append({"iterations": 0, "retried": False, "analytic_zero": True})
            continue
        if config.revised_stage1:


            pilot, _, rank, _ = np.linalg.lstsq(xi, yi, rcond=config.ols_rcond)
            residual = yi-xi@pilot
            dof = len(yi)-int(rank)
            if dof > 0:
                sigma_hat = math.sqrt(float(residual@residual)/dof)
                sigma_method = "local_ols_residual_standard_error"
            else:
                centered = yi-float(np.mean(yi))
                sigma_hat = math.sqrt(float(centered@centered)/max(1,len(yi)-1))
                sigma_method = "centered_response_rms_fallback"
            alpha = 2.0*sigma_hat*math.sqrt(2.0*math.log(x.shape[2]*config.n_selection)/x.shape[1])
        else:
            sigma_hat, sigma_method = None, "fixed_alpha_legacy"
            alpha = config.lasso_alpha
        converged = False
        limits = list(dict.fromkeys((config.lasso_max_iter, config.lasso_retry_max_iter)))
        for attempt, limit in enumerate(limits):
            model = Lasso(alpha=max(alpha,np.finfo(float).tiny), fit_intercept=False, tol=config.lasso_fit_tol,
                          max_iter=limit, selection="cyclic")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(xi, yi)
            if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
                converged = True
                break
        if not converged or not np.isfinite(model.coef_).all():
            raise RuntimeError(f"local Lasso did not converge for selection user {i}; alpha unchanged")
        coefficients[i] = model.coef_
        attempts.append({"iterations": int(model.n_iter_), "retried": attempt > 0,
                         "analytic_zero": False, "sigma_hat": sigma_hat,
                         "sigma_method": sigma_method, "lambda_loc": alpha})
    coefficients.setflags(write=False)
    return LocalLassoFit(coefficients, tuple(attempts), _local_lasso_signature(x, config))


def local_lasso_candidates(x: np.ndarray, y: np.ndarray, config: OriginalLassoConfig,
                           fitted: LocalLassoFit | None = None):
    'Threshold cached Lasso fits; keep empty supports for the original fallback.'
    x, y = _validate_data(x, y)
    config.validate()
    fitted = fit_local_lasso_models(x, y, config) if fitted is None else fitted
    if not isinstance(fitted, LocalLassoFit) or fitted.signature != _local_lasso_signature(x, config):
        raise ValueError("cached local Lasso fit is incompatible with stage-1 data/config")
    if (fitted.coefficients.shape != (len(x), x.shape[2]) or
            len(fitted.attempts) != len(x) or not np.isfinite(fitted.coefficients).all()):
        raise ValueError("cached local Lasso coefficient array is invalid")
    coefficient_threshold = (config.hh_threshold/(2.0*config.n_selection)
                             if config.revised_stage1 else config.lasso_zero_tol)
    supports = [np.flatnonzero(np.abs(row) >= coefficient_threshold)
                for row in fitted.coefficients]
    attempts = [dict(record, coefficient_threshold=coefficient_threshold)
                for record in fitted.attempts]
    return supports, attempts


def rr_flip_probability(n_selection: int, budget: StageBudget, amplified: bool = False) -> float:
    _integer("n_selection", n_selection, 1)
    budget.validate()
    if amplified:
        epsilon_local = budget.epsilon*math.sqrt(
            n_selection/(2.0*math.log(2.0/budget.delta)))
        if epsilon_local > 15.0:
            return 1e-6
        gamma = 1.0/(math.exp(epsilon_local)+1.0)
        if not 0 < gamma < 0.5:
            raise ValueError("amplified RR probability is outside (0,1/2)")
        return gamma
    log_gamma = (math.log(8.0) + math.log(math.log(4.0) - math.log(budget.delta))
                 - math.log(n_selection) - 2 * math.log(budget.epsilon))
    if log_gamma >= math.log(0.5):
        raise ValueError("original RR formula requires gamma<1/2; no silent clipping")
    gamma = math.exp(log_gamma)
    if not 0 < gamma < 0.5:
        raise ValueError("RR probability is outside representable float64 range")
    return gamma


def selection_client(support: Sequence[int], d: int, gamma: float,
                     sample_rng: np.random.Generator, rr_rng: np.random.Generator):
    _integer("dimension", d, 1)
    if not math.isfinite(gamma) or not 0 <= gamma < 0.5:
        raise ValueError("gamma must be in [0,1/2)")
    coords = sorted({_integer("candidate coordinate", v) for v in support})
    if coords and coords[-1] >= d:
        raise ValueError("candidate coordinate outside [d]")
    fallback = not coords
    v = int(sample_rng.choice(coords)) if coords else int(sample_rng.integers(d))
    bits = (np.arange(d) == v) ^ (rr_rng.random(d) < gamma)
    return [(j, int(bits[j])) for j in range(d)], v, fallback


def shuffle_labeled_messages(messages: Sequence[tuple[int, int]], rng: np.random.Generator):

    return tuple(messages[int(i)] for i in rng.permutation(len(messages)))


def aggregate_selection_messages(messages: Sequence[tuple[int, int]], n_selection: int, d: int):
    _integer("n_selection", n_selection, 1)
    _integer("dimension", d, 1)
    if len(messages) != n_selection * d:
        raise ValueError("incorrect selection message count")
    counts, totals = np.zeros(d, dtype=np.int64), np.zeros(d, dtype=np.int64)
    for label, bit in messages:
        label, bit = _integer("label", label), _integer("bit", bit)
        if label >= d or bit > 1:
            raise ValueError("invalid selection message")
        counts[label] += bit
        totals[label] += 1
    if not np.all(totals == n_selection):
        raise ValueError("each coordinate must have exactly n_selection reports")
    return counts


def select_support_by_threshold(counts: np.ndarray, n_selection: int, gamma: float, threshold: float):
    _integer("n_selection", n_selection, 1)
    _real("hh_threshold", threshold, positive=False)
    counts = np.asarray(counts)
    if counts.ndim != 1 or len(counts) < 1 or counts.dtype.kind not in "iu":
        raise ValueError("counts must be a nonempty integer vector")
    if np.any(counts < 0) or np.any(counts > n_selection) or not 0 <= gamma < 0.5:
        raise ValueError("invalid counts or gamma")
    debiased = (counts.astype(float) - n_selection * gamma) / (1 - 2 * gamma)
    return np.flatnonzero(debiased >= threshold), debiased


def local_ols_coefficients(x: np.ndarray, y: np.ndarray, selected: Sequence[int], rcond: float):
    'Unpenalized minimum-norm least squares at an explicit numerical rank tolerance.'
    x, y = _validate_data(x, y)
    _real("ols_rcond", rcond, positive=False)
    if rcond >= 1:
        raise ValueError("ols_rcond must be less than 1")
    selected = [_integer("selected coordinate", j) for j in selected]
    if not selected or len(set(selected)) != len(selected) or max(selected) >= x.shape[2]:
        raise ValueError("selected must contain distinct valid coordinates")
    coefficients, ranks = [], []
    for xi, yi in zip(x, y):
        beta, _, rank, _ = np.linalg.lstsq(xi[:, selected], yi, rcond=rcond)
        if not np.isfinite(beta).all():
            raise FloatingPointError("OLS returned nonfinite coefficients")
        coefficients.append(beta)
        ranks.append(int(rank))
    return np.asarray(coefficients), ranks


def local_ols_coefficients_appendix_b(x: np.ndarray, y: np.ndarray, selected: Sequence[int]):
    'Algorithm 4 line 3 exactly: (X_S^T X_S)^(-1) X_S^T y.'
    x, y = _validate_data(x, y)
    selected = [_integer("selected coordinate", j) for j in selected]
    if not selected or len(set(selected)) != len(selected) or max(selected) >= x.shape[2]:
        raise ValueError("selected must contain distinct valid coordinates")
    s = len(selected)
    if s > x.shape[1]:
        raise ValueError(f"Appendix-B Algorithm 4 local inverse inapplicable: support size s={s} exceeds m={x.shape[1]}")
    coefficients=[]
    for user,(xi,yi) in enumerate(zip(x,y)):
        xs=xi[:,selected]
        gram=xs.T@xs
        if np.linalg.matrix_rank(gram) < s:
            raise ValueError(f"Appendix-B Algorithm 4 local inverse singular for user {user}: rank < s")
        beta=np.linalg.solve(gram,xs.T@yi)
        if not np.isfinite(beta).all():
            raise FloatingPointError("Appendix-B local OLS returned nonfinite coefficients")
        coefficients.append(beta)
    return np.asarray(coefficients)


def local_ols_coefficients_private_totalized(x: np.ndarray, y: np.ndarray, selected: Sequence[int], *, return_diagnostics: bool = False):
    'Totalized private-release version of Algorithm-4 local OLS.'
    x, y = _validate_data(x, y)
    selected = [_integer("selected coordinate", j) for j in selected]
    if not selected or len(set(selected)) != len(selected) or max(selected) >= x.shape[2]:
        raise ValueError("selected must contain distinct valid coordinates")
    s = len(selected)
    coefficients=[]; failures=0
    for xi,yi in zip(x,y):
        beta=np.zeros(s,dtype=float); success=False
        if s <= x.shape[1]:
            try:
                xs=xi[:,selected]
                with np.errstate(over="raise", invalid="raise"):
                    gram=xs.T@xs
                if np.linalg.matrix_rank(gram) == s:
                    candidate=np.linalg.solve(gram,xs.T@yi)


                    if np.isfinite(candidate).all() and np.max(np.abs(candidate),initial=0.0) <= 1e150:
                        beta=candidate; success=True
            except Exception:
                pass
        failures += int(not success)
        coefficients.append(beta)
    out=np.asarray(coefficients)
    return (out,{"local_ols_fallback_count":failures}) if return_diagnostics else out

def compute_stage_radii(config: OriginalLassoConfig, s: int, n: int, m: int):
    config.validate()
    _integer("selected dimension", s, 1)
    _integer("n", n, 2)
    _integer("m", m, 1)
    if not config.parameter_table_mode:
        return config.coarse_radius, config.tau_beta
    tau_beta = (config.tau_beta_constant * config.noise_sigma *
                math.sqrt(s * math.log(n) / m))
    coarse_radius = 1.0 + tau_beta
    if coarse_radius > 2.0:
        raise ValueError("parameter-table assumption R0=1+tau_beta<=2 is violated")
    return coarse_radius, tau_beta


def compute_fine_radius(config: OriginalLassoConfig, s: int, n: int, m: int):
    config.validate()
    _integer("selected dimension", s, 1)
    _integer("n", n, 2)
    _integer("m", m, 1)
    coarse_radius, local = compute_stage_radii(config, s, n, m)
    sampling = config.c1 * config.noise_sigma * math.sqrt(s * math.log(n) / (config.n_coarse * m))
    center_noise = (config.c2 * math.sqrt(s) * coarse_radius *
                    math.sqrt(-math.log(config.coarse_budget.delta)) /
                    (config.n_coarse * config.coarse_budget.epsilon))
    radius = local + sampling + center_noise
    _real("fine radius", radius)
    return radius, {"tau_beta": local, "coarse_radius": coarse_radius,
                    "sampling": sampling, "center_noise": center_noise, "total": radius}


@dataclass(frozen=True)
class AggregationParameters:
    n_users: int
    dimension: int
    radius: float
    precision: int
    requested_trials: int
    local_trials: int
    total_trials: int
    modulus: int
    shares: int
    security_bits: int
    delta_noise: float
    delta_security: float
    integer_sensitivity: float
    privacy_certificate: dict | None = None

    def validate(self) -> None:
        _integer("n_users", self.n_users, 19)
        _integer("dimension", self.dimension, 1)
        _real("radius", self.radius)
        _integer("precision", self.precision, 1)
        _integer("local_trials", self.local_trials, 2)
        _integer("requested_trials", self.requested_trials, 1)
        _integer("total_trials", self.total_trials, 1)
        _integer("modulus", self.modulus, 3)
        _integer("shares", self.shares, 3)
        _integer("security_bits", self.security_bits, 1)
        _real("integer_sensitivity", self.integer_sensitivity)
        if self.local_trials % 2 or self.total_trials != self.n_users * self.local_trials:
            raise ValueError("local noise needs even N0 and M=n_users*N0")
        if self.total_trials < self.requested_trials:
            raise ValueError("actual M must cover requested trials")
        scaled_radius = self.precision * self.radius
        if not math.isfinite(scaled_radius) or max(self.precision, self.local_trials, scaled_radius) > MAX_LOCAL_INTEGER:
            raise ValueError("local encoding/trials exceed safe float64 integer limits")
        bound = self.n_users * math.ceil(scaled_radius) + self.total_trials // 2
        if self.modulus % 2 != 1 or self.modulus <= 2 * bound:
            raise ValueError("modulus must be odd and cover the full signed sum")
        for name in ("delta_noise", "delta_security"):
            if _real(name, getattr(self, name)) >= 1:
                raise ValueError(f"{name} must be less than 1")


def make_aggregation_parameters(n_users: int, dimension: int, radius: float, n_total: int,
                                budget: StageBudget, precision_constant: float, noise_constant: float):
    _integer("n_users", n_users, 19)
    _integer("dimension", dimension, 1)
    _integer("n_total", n_total, n_users)
    _real("radius", radius)
    _real("precision_constant", precision_constant)
    _real("noise_constant", noise_constant)
    budget.validate()
    delta_noise = delta_security = budget.delta / 2
    _real("delta_noise", delta_noise)
    log_delta = -math.log(delta_noise)
    p_float = precision_constant * math.sqrt(n_users * math.log(n_total)) * budget.epsilon / (radius * math.sqrt(log_delta))
    _real("precision expression", p_float)
    precision = max(1, math.ceil(p_float))
    if precision > MAX_LOCAL_INTEGER or precision * radius > MAX_LOCAL_INTEGER:
        raise ValueError("requested precision exceeds safe local integer limits")
    sensitivity = 2 * precision * radius + 2 * math.sqrt(dimension)
    m_float = noise_constant * (sensitivity / budget.epsilon)**2 * log_delta
    _real("binomial trial expression", m_float)
    requested = max(2, math.ceil(m_float))
    local_trials = 2 * ((requested + 2*n_users - 1) // (2*n_users))
    total_trials = n_users * local_trials
    bound = n_users * math.ceil(precision * radius) + total_trials // 2
    modulus = 2 * bound + 1
    log_one_plus_exp = budget.epsilon + math.log1p(math.exp(-budget.epsilon))
    kappa = max(1, math.ceil((log_one_plus_exp + math.log(dimension) - math.log(delta_security)) / math.log(2)))
    shuffled = max(3, math.ceil((2*kappa + math.log2(modulus)) / (math.log2(n_users)-math.log2(math.e)) + 1))
    params = AggregationParameters(n_users, dimension, radius, precision, requested, local_trials,
                                   total_trials, modulus, shuffled, kappa, delta_noise,
                                   delta_security, sensitivity)
    params.validate()
    return params


def make_table_aggregation_parameters(n_users: int, dimension: int, radius: float, n_total: int,
                                      budget: StageBudget, precision_constant: float):
    'Concrete p, M, N0, q, and r calibration from the supplied table.'
    _integer("n_users", n_users, 19)
    _integer("dimension", dimension, 1)
    _integer("n_total", n_total, n_users)
    _real("radius", radius)
    _real("precision_constant", precision_constant)
    budget.validate()
    log_delta = math.log(1.0 / budget.delta)
    p_float = max(
        precision_constant * math.sqrt(n_users * math.log(n_total)) * budget.epsilon /
        (radius * math.sqrt(log_delta)),
        (math.sqrt(dimension) + 2) * max(1.0, budget.epsilon) / radius,
    )
    _real("precision expression", p_float)
    precision = max(1, math.ceil(p_float))
    sensitivity = 2.0 * precision * radius + max(1.0, budget.epsilon) * math.sqrt(dimension) + 2
    requested = max(2, math.ceil(32.0 * (sensitivity / budget.epsilon)**2 *
                                 math.log(4.0 * dimension / budget.delta)))
    local_trials = 2 * ((requested + 2*n_users - 1) // (2*n_users))
    total_trials = n_users * local_trials

    modulus = math.ceil(4 * n_users * (precision * radius + local_trials))
    modulus += 1 - modulus % 2
    sigma_security = math.log2(8.0 * dimension / budget.delta) + budget.epsilon / math.log(2)
    denominator = math.log2(n_users) - math.log2(math.e)
    shares = max(3, math.ceil((2.0 * sigma_security + math.log2(modulus)) / denominator) + 1)
    params = AggregationParameters(
        n_users, dimension, radius, precision, requested, local_trials, total_trials,
        modulus, shares, math.ceil(sigma_security), budget.delta, budget.delta, sensitivity,
    )
    params.validate()
    return params


def clip_offset(value: np.ndarray, center: np.ndarray, radius: float):
    'Stable radial L2 projection, without forming a potentially overflowing norm.'
    _real("radius", radius)
    value, center = np.asarray(value, dtype=float), np.asarray(center, dtype=float)
    if value.ndim != 1 or value.shape != center.shape or not np.isfinite(value).all() or not np.isfinite(center).all():
        raise ValueError("finite matching vectors required")
    with np.errstate(over="raise", invalid="raise"):
        offset = value - center
    largest = float(np.max(np.abs(offset), initial=0.0))
    clipped = False
    if largest:
        unit_norm = float(np.linalg.norm(offset / largest))
        clipped = largest > radius / unit_norm
        if clipped:
            offset = (offset / largest) * (radius / unit_norm)

    return np.clip(offset, -radius, radius), clipped


def encode_local_vector(value: np.ndarray, center: np.ndarray, params: AggregationParameters,
                        rounding_rng: np.random.Generator, noise_rng: np.random.Generator):
    offset, was_clipped = clip_offset(value, center, params.radius)
    if len(offset) != params.dimension:
        raise ValueError("local dimension disagrees with parameters")
    scaled = params.precision * offset
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) > MAX_LOCAL_INTEGER):
        raise ValueError("unsafe local quantization magnitude")
    floors = np.floor(scaled)
    rounded = (floors + (rounding_rng.random(len(offset)) < scaled-floors)).astype(np.int64)
    noise = noise_rng.binomial(params.local_trials, 0.5, size=len(offset)) - params.local_trials // 2
    integers = [int(u) + int(v) for u, v in zip(rounded, noise)]
    messages = tuple(z % params.modulus for z in integers)

    return messages, {"offset": offset, "rounded": rounded, "noise": noise,
                      "signed_integer": integers, "was_clipped": was_clipped}


def _randbelow(q: int, rng: np.random.Generator) -> int:
    bits = (q-1).bit_length()
    while True:
        value = int.from_bytes(rng.bytes((bits+7)//8), "little") & ((1 << bits)-1)
        if value < q:
            return value


def ikos_client(w: int, q: int, r: int, rng: np.random.Generator) -> tuple[int, ...]:
    'Draw r shuffled shares and append one determined unshuffled share.'
    w, q, r = _integer("w", w), _integer("q", q, 2), _integer("r", r, 3)
    if w >= q:
        raise ValueError("w must be in Z_q")
    first = tuple(_randbelow(q, rng) for _ in range(r))
    return first + ((w - sum(first)) % q,)


def ikos_shuffler(lane: Sequence[int], rng: np.random.Generator) -> tuple[int, ...]:
    return tuple(int(lane[int(i)]) for i in rng.permutation(len(lane)))


@dataclass(frozen=True)
class IKOSTranscript:
    shuffled: tuple[tuple[int, ...], ...]
    direct: tuple[int, ...]
    modulus: int


def ikos_aggregate(transcript: IKOSTranscript, n_users: int, r: int) -> int:
    _integer("n_users", n_users, 1)
    _integer("r", r, 3)
    q = _integer("modulus", transcript.modulus, 2)
    if len(transcript.shuffled) != r or len(transcript.direct) != n_users:
        raise ValueError("incorrect IKOS lane/direct-message count")
    total = 0
    for lane in (*transcript.shuffled, transcript.direct):
        if len(lane) != n_users:
            raise ValueError("incomplete IKOS lane")
        for value in lane:
            value = _integer("share", value)
            if value >= q:
                raise ValueError("share outside Z_q")
            total = (total + value) % q
    return total


def execute_ikos(values: Sequence[int], q: int, r: int, streams: RandomStreams, namespace: str,
                 audit: dict | None = None) -> int:
    'Simulation router; each shuffler receives ONLY its single scalar lane.'
    if not len(values):
        raise ValueError("IKOS needs participants")
    q, r = _integer("q", q, 2), _integer("r", r, 3)
    lanes, direct, local_shares = [[] for _ in range(r)], [], []
    share_rng = streams.rng(namespace + "/shares", transport=True)
    for w in values:
        shares = ikos_client(w, q, r, share_rng)
        for k in range(r):
            lanes[k].append(shares[k])
        direct.append(shares[-1])
        if audit is not None:
            local_shares.append(shares)
    shuffled = tuple(ikos_shuffler(lane, streams.rng(f"{namespace}/shuffle/{k}", transport=True))
                     for k, lane in enumerate(lanes))
    view = IKOSTranscript(shuffled, tuple(direct), q)
    result = ikos_aggregate(view, len(values), r)
    if audit is not None:
        audit.update({"local_shares": tuple(local_shares), "transcript": view})
    return result


def signed_decode(residue: int, q: int) -> int:
    residue, q = _integer("residue", residue), _integer("q", q, 3)
    if q % 2 != 1 or residue >= q:
        raise ValueError("signed decoder requires an odd q and residue in Z_q")
    return residue-q if residue > q//2 else residue


def distributed_private_mean_ikos(values: np.ndarray, center: np.ndarray,
                                  params: AggregationParameters, streams: RandomStreams,
                                  stage: str, audit: dict | None = None,
                                  materialize_ikos: bool = True):
    params.validate()
    values, center = np.asarray(values, dtype=float), np.asarray(center, dtype=float)
    if values.shape != (params.n_users, params.dimension) or center.shape != (params.dimension,):
        raise ValueError("mean inputs disagree with parameters")
    if not np.isfinite(values).all() or not np.isfinite(center).all():
        raise ValueError("mean inputs must be finite")
    messages, local_audit = [], []
    for i, value in enumerate(values):
        message, record = encode_local_vector(value, center, params,
            streams.rng(f"{stage}/round/{i}"), streams.rng(f"{stage}/noise/{i}"))
        messages.append(message)
        if audit is not None:
            local_audit.append(record)
    signed, transcripts = [], []
    for j in range(params.dimension):
        if materialize_ikos:
            coordinate_audit = {} if audit is not None else None
            residue = execute_ikos([row[j] for row in messages], params.modulus, params.shares,
                                   streams, f"{stage}/coord/{j}", coordinate_audit)
            if audit is not None:
                transcripts.append(coordinate_audit)
        else:


            # rounding or each user's Bin(N0,1/2)-N0/2 draw.
            residue = sum(row[j] for row in messages) % params.modulus
        signed.append(signed_decode(residue, params.modulus))
    with np.errstate(over="raise", invalid="raise"):
        estimate = center + np.asarray(signed, dtype=float) / (params.n_users * params.precision)
    if not np.isfinite(estimate).all():
        raise FloatingPointError("nonfinite decoded mean")
    if audit is not None:
        audit.update({"local": local_audit, "coordinates": transcripts, "signed_sum": signed})
    return estimate, {"parameters": asdict(params),
                      "protocol": ("materialized_parallel_ikos" if materialize_ikos
                                   else "aggregate_equivalent_skip_ikos"),
                      "local_noise_draws": params.n_users * params.dimension,
                      "shuffled_messages": (params.n_users*params.dimension*params.shares
                                            if materialize_ikos else 0),
                      "direct_messages": params.n_users * params.dimension,
                      "equivalent_ikos_messages": params.n_users*params.dimension*(params.shares+1)}


@dataclass
class SDP1Result:
    coefficients: np.ndarray
    support: np.ndarray
    metadata: dict


def fit_sdp1_original_lasso(user_x: np.ndarray, user_y: np.ndarray, config: OriginalLassoConfig,
                            partition: UserPartition | None = None, *,
                            selection_lasso_fit: LocalLassoFit | None = None,
                            transport_seed: int | None = None,
                            audit: dict | None = None) -> SDP1Result:
    'Complete original SDP1 flow. Audit is simulator-local and never an aggregator input.'
    config.validate()
    x, y = _validate_data(user_x, user_y)
    n, m, d = x.shape
    partition = make_partition(n, config) if partition is None else partition
    partition.validate(n, config)
    streams = RandomStreams(config.seed, transport_seed)
    i1, ia, ib = (np.asarray(group, dtype=int) for group in
                  (partition.selection, partition.coarse, partition.fine))
    candidates, lasso_attempts = local_lasso_candidates(
        x[i1], y[i1], config, selection_lasso_fit)
    gamma = rr_flip_probability(config.n_selection, config.selection_budget,
                                config.revised_stage1)
    epsilon_local = (config.selection_budget.epsilon*math.sqrt(
        config.n_selection/(2.0*math.log(2.0/config.selection_budget.delta)))
        if config.revised_stage1 else None)
    reports, picks, fallbacks = [], [], []
    pick_rng, rr_rng = streams.rng("selection/pick"), streams.rng("selection/rr")
    for support in candidates:
        messages, picked, fallback = selection_client(support, d, gamma, pick_rng, rr_rng)
        reports.extend(messages)
        picks.append(picked)
        fallbacks.append(fallback)
    shuffled = shuffle_labeled_messages(reports, streams.rng("selection/shuffle", transport=True))
    counts = aggregate_selection_messages(shuffled, config.n_selection, d)
    selected, debiased = select_support_by_threshold(counts, config.n_selection, gamma, config.hh_threshold)
    meta = {"algorithm": "SDP1-original-Lasso", "execution": "full_protocol_single_process_simulation",
            "privacy_target": "sample_replace_one_fixed_partition",
            "privacy_status": "not_an_end_to_end_certificate", "config": asdict(config),
            "partition": asdict(partition), "shape": [n, m, d], "selected": selected.tolist(),
            "selection": {"gamma": gamma, "epsilon_local": epsilon_local,
                          "rho": config.hh_threshold/config.n_selection,
                          "coefficient_threshold": (config.hh_threshold/(2*config.n_selection)
                                                    if config.revised_stage1 else config.lasso_zero_tol),
                          "counts": counts.tolist(), "debiased": debiased.tolist(),
                          "threshold_count": config.hh_threshold, "messages": len(shuffled)}}
    if audit is not None:
        audit.update({"local_candidates": candidates, "local_lasso": lasso_attempts,
                      "picked": picks, "fallbacks": fallbacks, "selection_transcript": shuffled})
    beta = np.zeros(d)
    if not len(selected):
        meta.update({"status": "empty_global_support", "coarse": None, "fine": None})
        return SDP1Result(beta, selected, meta)
    s = len(selected)
    coarse_radius, _ = compute_stage_radii(config, s, n, m)
    coarse, coarse_ranks = local_ols_coefficients(x[ia], y[ia], selected, config.ols_rcond)
    if config.parameter_table_mode:
        params_a = make_table_aggregation_parameters(config.n_coarse, s, coarse_radius, n,
            config.coarse_budget, config.precision_constant)
    else:
        params_a = make_aggregation_parameters(config.n_coarse, s, coarse_radius, n,
            config.coarse_budget, config.precision_constant, config.noise_constant)
    coarse_audit = {} if audit is not None else None
    center, coarse_meta = distributed_private_mean_ikos(
        coarse,np.zeros(s),params_a,streams,"coarse",coarse_audit,
        materialize_ikos=not config.simulation_skip_ikos)
    radius, components = compute_fine_radius(config, s, n, m)
    fine, fine_ranks = local_ols_coefficients(x[ib], y[ib], selected, config.ols_rcond)
    if config.parameter_table_mode:
        params_b = make_table_aggregation_parameters(config.n_fine, s, radius, n,
            config.fine_budget, config.precision_constant)
    else:
        params_b = make_aggregation_parameters(config.n_fine, s, radius, n, config.fine_budget,
            config.precision_constant, config.noise_constant)
    fine_audit = {} if audit is not None else None
    estimate, fine_meta = distributed_private_mean_ikos(
        fine,center,params_b,streams,"fine",fine_audit,
        materialize_ikos=not config.simulation_skip_ikos)
    beta[selected] = estimate
    meta.update({"status": "ok", "center": center.tolist(), "coarse_radius_used": coarse_radius,
                 "fine_radius_components": components,
                 "coarse": coarse_meta, "fine": fine_meta})
    if audit is not None:
        audit.update({"coarse_coefficients": coarse, "coarse_ranks": coarse_ranks,
                      "fine_coefficients": fine, "fine_ranks": fine_ranks,
                      "coarse": coarse_audit, "fine": fine_audit})
    return SDP1Result(beta, selected, meta)
