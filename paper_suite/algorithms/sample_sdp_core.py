'Frozen appendix Sample SDP statistics and exact aggregate-noise simulation. Extracted from the verified implementation; no alternate estimator or plot code.'
from __future__ import annotations
from dataclasses import dataclass, asdict
from dp_safety.aggregation import certify_existing_noise, security_bits, ensure_local_trials
from dp_safety.parameters import APPROXIMATE_DP_DELTA
import math
import numpy as np

@dataclass(frozen=True)
class Config:
    delta: float = APPROXIMATE_DP_DELTA
    sdp2_clip: float = 200.0
    sdp2_dimension: int = 0
    sdp2_log_population: int = 0
    sdp2_clip_constant: float = 0.1
    sdp2_threshold_constant: float = 2.0

    sdp2_noise_scale: float = 0.25
    sdp2_kappa_infinity: float = 1.0
    l1_radius: float = 1.0

def project_l1_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    if np.linalg.norm(vector, 1) <= radius:
        return vector.copy()
    u = np.sort(np.abs(vector))[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > cssv - radius)[0][-1]
    theta = (cssv[rho] - radius) / (rho + 1.0)
    return np.sign(vector) * np.maximum(np.abs(vector) - theta, 0.0)

def _statistic_block(xb, yb, upper, clip):
    'Bounded local mapping including exceptional floating-point inputs.'
    with np.errstate(over='ignore',invalid='ignore'):
        f=np.concatenate((xb[:,upper[0]]*xb[:,upper[1]],xb*yb[:,None]),axis=1)
        norms=np.linalg.norm(f,axis=1)
    bad=~np.isfinite(f).all(axis=1) | ~np.isfinite(norms)
    if np.any(bad):
        f[bad]=0.;norms[bad]=0.
    return f,norms,int(np.sum(bad))

def stochastic_round(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    floors = np.floor(values)
    return (floors + (rng.random(values.shape) < values - floors)).astype(np.int64)

def ikos_shuffled_share_count(n_users: int, q: int, delta_security: float, streams: int) -> int:
    'Corollary 6.1 ``r``: shuffled random shares per user/coordinate.'
    sigma = math.log2(8.0 * streams / delta_security)
    denominator = math.log2(n_users) - math.log2(math.e)
    if n_users < 19 or denominator <= 0:
        return -1
    return math.ceil((2.0 * sigma + math.log2(q)) / denominator) + 1

def clipped_sufficient_statistic_sum(x: np.ndarray, y: np.ndarray, clip: float, chunk_size: int=256):
    dimension = x.shape[1]
    upper = np.triu_indices(dimension)
    stat_dimension = len(upper[0]) + dimension
    total = np.zeros(stat_dimension)
    clipped_count = 0
    norm_quantiles_sample = []
    for start in range(0, len(x), chunk_size):
        xb = x[start:start + chunk_size]
        yb = y[start:start + chunk_size]
        outer = xb[:, :, None] * xb[:, None, :]
        f = np.concatenate((outer[:, upper[0], upper[1]], xb * yb[:, None]), axis=1)
        norms = np.linalg.norm(f, axis=1)
        scales = np.minimum(1.0, clip / np.maximum(norms, 1e-12))
        clipped_count += int(np.sum(norms > clip))
        if len(norm_quantiles_sample) < 5000:
            norm_quantiles_sample.extend(norms[:max(0, 5000 - len(norm_quantiles_sample))].tolist())
        total += np.sum(f * scales[:, None], axis=0)
    return (total, {'n': len(x), 'd': dimension, 'D': stat_dimension, 'clip_rate': clipped_count / len(x), 'raw_norm_percentiles': np.percentile(norm_quantiles_sample, [50, 90, 95, 99]).tolist()})

def sdp2_table_parameters(n: int, d: int, epsilon: float, delta: float, log_population: int | None=None, *, clip_constant: float=0.1, threshold_constant: float=2.0, noise_scale: float=0.25, kappa_infinity: float=1.0) -> dict:
    'Concrete sufficient-statistic parameters.'
    log_population = n if log_population is None else log_population
    if n < 19 or log_population < n or d < 1 or (epsilon <= 0) or (not 0 < delta < 1):
        raise ValueError('B.2 parameters require n>=19, d>=1, epsilon>0 and delta in (0,1)')
    D = d * (d + 1) // 2 + d
    log_n = math.log(log_population)
    if not all((math.isfinite(v) and v > 0 for v in (clip_constant, threshold_constant))):
        raise ValueError('positive finite SDP2 constants required')
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError('noise_scale must be finite and nonnegative')
    if not math.isfinite(kappa_infinity) or kappa_infinity <= 0:
        raise ValueError('kappa_infinity must be finite and positive')
    clip = clip_constant * d * log_n
    precision = max(1, math.ceil(math.sqrt(n) * epsilon / (d * log_n * math.sqrt(math.log(4.0 * D / delta)))))
    sensitivity = 2.0 * precision * clip + math.sqrt(D)
    requested = max(1, math.ceil(32.0 * sensitivity ** 2 * math.log(4.0 * D / delta) / epsilon ** 2))
    local_trials_before_even = max(1, math.ceil(requested / n))
    local_trials = local_trials_before_even + local_trials_before_even % 2
    table_local_trials = local_trials
    local_trials = ensure_local_trials(int(n),int(D),sensitivity,epsilon,delta,int(local_trials))
    total_trials = n * local_trials
    modulus = math.ceil(4.0 * n * (precision * clip + local_trials))
    sigma_security = math.log2(8.0 * D / delta)
    shuffled_shares = ikos_shuffled_share_count(n, modulus, delta, D)
    if total_trials > 2**53-1 or modulus >= 2**63-1:
        raise ValueError('public Sample-SDP parameters exceed safe integer arithmetic limits')
    bound = n * math.ceil(precision*clip) + total_trials//2
    if modulus <= 2*bound or bound >= 2**62:
        raise ValueError('modulus or integer sum range does not safely cover the query')
    cert = certify_existing_noise(n_users=int(n),dimension=int(D),modulus=int(modulus),
        total_trials=int(total_trials),sensitivity_l2=sensitivity,
        epsilon=float(epsilon),delta=float(delta),current_shares=shuffled_shares)
    shuffled_shares = cert.shuffled_shares
    sigma_security = security_bits(epsilon,delta,D)
    threshold_statistical = noise_scale * math.sqrt(math.log(d) / n)
    threshold_private = clip * math.sqrt(math.log(4.0 * D / delta) * math.log(D)) / (n * epsilon)
    threshold = threshold_constant / kappa_infinity * (threshold_statistical + threshold_private)
    return {'privacy_certificate':asdict(cert),'table_local_trials':table_local_trials, 'noise_recalibrated':local_trials!=table_local_trials, 'n': n, 'log_population': log_population, 'd': d, 'D': D, 'clip': clip, 'precision': precision, 'integer_sensitivity': sensitivity, 'requested_trials': requested, 'local_trials_before_even': local_trials_before_even, 'local_trials': local_trials, 'total_trials': total_trials, 'modulus': modulus, 'sigma_security': sigma_security, 'shuffled_shares': shuffled_shares, 'unshuffled_shares': 1, 'total_shares': shuffled_shares + 1, 'shuffled_streams': D * shuffled_shares, 'direct_streams': D, 'total_message_streams': D * (shuffled_shares + 1), 'soft_threshold': threshold, 'threshold_rule': 'Table2_full_20261021', 'threshold_noise_scale': noise_scale, 'threshold_kappa_infinity': kappa_infinity, 'threshold_statistical_component': threshold_constant / kappa_infinity * threshold_statistical, 'threshold_private_component': threshold_constant / kappa_infinity * threshold_private}

def sdp2_precision(n: int, epsilon: float, config: Config) -> int:
    if n < 2 or epsilon <= 0 or config.sdp2_clip <= 0 or (not 0 < config.delta < 1):
        raise ValueError('invalid SDP2 count, epsilon, clip, or delta')
    d = config.sdp2_dimension
    if d < 1:
        raise ValueError('sdp2_dimension is required for supplied B.2 parameters')
    log_population = config.sdp2_log_population or n
    return sdp2_table_parameters(n, d, epsilon, config.delta, log_population, clip_constant=config.sdp2_clip_constant, threshold_constant=config.sdp2_threshold_constant, noise_scale=config.sdp2_noise_scale, kappa_infinity=config.sdp2_kappa_infinity)['precision']

def exact_rounded_statistic_sums(x: np.ndarray, y: np.ndarray, clip: float, precision: int, rngs: list[np.random.Generator], chunk_size: int=256) -> np.ndarray:
    'Exact independent per-record unbiased rounding, memory bounded.'
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 1:
        raise ValueError('x/y shapes do not match')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite inputs required')
    if clip <= 0 or not np.isfinite(clip) or precision < 1 or (int(precision) != precision) or (chunk_size < 1) or (not rngs):
        raise ValueError('positive clip, integer precision/chunk size and RNGs required')
    upper = np.triu_indices(x.shape[1])
    dimension = len(upper[0]) + x.shape[1]
    sums = np.zeros((len(rngs), dimension), dtype=np.int64)
    for start in range(0, len(x), chunk_size):
        xb, yb = (x[start:start + chunk_size], y[start:start + chunk_size])
        f,norms,_ = _statistic_block(xb,yb,upper,clip)
        f *= np.minimum(1.0, clip / np.maximum(norms, 1e-12))[:, None]
        scaled = precision * f
        floors = np.floor(scaled)
        fractions = scaled - floors
        floor_sum = floors.sum(axis=0).astype(np.int64)
        for repeat, rng in enumerate(rngs):
            sums[repeat] += floor_sum + (rng.random(fractions.shape) < fractions).sum(axis=0)
    return sums

def exact_local_binomial_noise_sum(n: int, dimension: int, local_trials: int, rng: np.random.Generator, chunk_size: int=256) -> np.ndarray:
    "Draw every sample's local centered-binomial vector, then sum it."
    if n < 1 or dimension < 1 or chunk_size < 1:
        raise ValueError('positive n, dimension, and chunk_size are required')
    if local_trials < 2 or local_trials % 2:
        raise ValueError('local binomial trial count N0 must be positive and even')
    total = np.zeros(dimension, dtype=np.int64)
    for start in range(0, n, chunk_size):
        count = min(chunk_size, n - start)
        local = rng.binomial(local_trials, 0.5, size=(count, dimension))
        local = local.astype(np.int64, copy=False) - local_trials // 2
        total += local.sum(axis=0, dtype=np.int64)
    return total

def exact_aggregate_binomial_noise_sum(n, dimension, local_trials, rng):
    'Exact convolution law for the aggregate, not a Gaussian approximation.'
    if any((isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (n, dimension, local_trials))):
        raise ValueError('integer dimensions/trials required')
    if n < 1 or dimension < 1 or local_trials < 2 or local_trials % 2:
        raise ValueError('positive sizes and even local trials required')
    total = int(n) * int(local_trials)
    if total > 2 ** 53 - 1:
        raise ValueError('aggregate trials exceed exact floating integer range')
    return rng.binomial(total, 0.5, size=dimension).astype(np.int64) - total // 2

def prepare_exact_sample_statistics(x, y, clip, precision, rng, chunk_size=64):
    'Fuse clipping diagnostics and exact per-record rounding into one pass.'
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 1 or (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
        raise ValueError('finite nonempty aligned x/y required')
    if clip <= 0 or not np.isfinite(clip) or precision < 1 or (int(precision) != precision) or (chunk_size < 1):
        raise ValueError('invalid clipping/precision/chunk')
    n, d = x.shape
    upper = np.triu_indices(d)
    dim = len(upper[0]) + d
    rounded = np.zeros(dim, dtype=np.int64)
    clipped = 0
    local_totalizations = 0
    sample = []
    for start in range(0, n, chunk_size):
        xb = x[start:start + chunk_size]
        yb = y[start:start + chunk_size]
        f,norms,bad_count = _statistic_block(xb,yb,upper,clip)
        local_totalizations += bad_count
        clipped += int(np.sum(norms > clip))
        if len(sample) < 5000:
            sample.extend(norms[:5000 - len(sample)].tolist())
        f *= np.minimum(1.0, clip / np.maximum(norms, 1e-12))[:, None]
        f *= precision
        floors = np.floor(f)
        f -= floors
        rounded += floors.sum(axis=0).astype(np.int64) + (rng.random(f.shape) < f).sum(axis=0)
    return {'rounded': rounded, 'precision': int(precision), 'clip': float(clip), 'meta': {'n': n, 'd': d, 'D': dim, 'clip_rate': clipped / n, 'local_statistic_totalization_count':local_totalizations, 'raw_norm_percentiles': np.percentile(sample, [50, 90, 95, 99]).tolist()}}

def fit_sdp2_from_sum(statistic_sum: np.ndarray, statistic_meta: dict, epsilon: float, config: Config, rng: np.random.Generator, exact_rounded_sum: np.ndarray | None=None, ):
    n = statistic_meta['n']
    dimension = statistic_meta['d']
    stat_dimension = statistic_meta['D']
    if config.sdp2_dimension != dimension:
        raise ValueError('configured SDP2 dimension disagrees with statistic metadata')
    if (not math.isclose(config.l1_radius, 1.0, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError('final SDP2 requires projection onto the unit L1 ball')
    log_term = math.sqrt(2.0 * math.log(1.25 / config.delta))
    precision = sdp2_precision(n, epsilon, config)
    if exact_rounded_sum is not None:
        exact_rounded_sum = np.asarray(exact_rounded_sum)
        if exact_rounded_sum.shape != (stat_dimension,) or not np.issubdtype(exact_rounded_sum.dtype, np.integer):
            raise ValueError('exact rounded sum must be an integer statistic vector')
        rounded_sum = exact_rounded_sum
    else:
        raise ValueError('final SDP2 requires exact per-sample unbiased rounding')
    log_population = config.sdp2_log_population or n
    table = sdp2_table_parameters(n, dimension, epsilon, config.delta, log_population, clip_constant=config.sdp2_clip_constant, threshold_constant=config.sdp2_threshold_constant, noise_scale=config.sdp2_noise_scale, kappa_infinity=config.sdp2_kappa_infinity)
    if not math.isclose(config.sdp2_clip, table['clip'], rel_tol=1e-12):
        raise ValueError('configured clip radius disagrees with B.2 table')
    integer_sensitivity = table['integer_sensitivity']
    requested_trials = table['requested_trials']
    local_trials = table['local_trials']
    total_trials = table['total_trials']
    sigma_integer = math.sqrt(total_trials) / 2.0
    sampler = exact_aggregate_binomial_noise_sum
    binomial_noise = sampler(n, stat_dimension, local_trials, rng)
    aggregate_noise_bias = 0.0
    integer_sum = rounded_sum + binomial_noise.astype(np.int64)
    modulus = table['modulus']
    modular_sum = np.mod(integer_sum, modulus)
    signed_sum = np.where(modular_sum > modulus // 2, modular_sum - modulus, modular_sum)
    f_hat = (signed_sum.astype(float) - aggregate_noise_bias) / (n * precision)
    upper = np.triu_indices(dimension)
    h_hat = np.zeros((dimension, dimension))
    h_hat[upper] = f_hat[:len(upper[0])]
    h_hat[upper[1], upper[0]] = h_hat[upper]
    g_hat = f_hat[len(upper[0]):]
    solver_fallback = False
    try:
        eigenvalues = np.linalg.eigvalsh(h_hat)
        beta_raw = np.linalg.solve(h_hat, g_hat)
        if not np.isfinite(beta_raw).all() or np.max(np.abs(beta_raw), initial=0.)>1e150:
            raise FloatingPointError('unsafe post-private inverse')
    except (np.linalg.LinAlgError, FloatingPointError):

        eigenvalues = np.zeros(dimension)
        beta_raw = np.zeros(dimension)
        solver_fallback = True
    coordinate_noise_sd = sigma_integer / (n * precision)
    threshold = table['soft_threshold']
    beta_thresholded = np.sign(beta_raw) * np.maximum(np.abs(beta_raw) - threshold, 0.0)
    thresholded_l1 = float(np.linalg.norm(beta_thresholded, 1))
    beta = project_l1_ball(beta_thresholded, config.l1_radius)
    final_l1 = float(np.linalg.norm(beta, 1))
    return (beta, {**statistic_meta, 'privacy_target': 'sample_replace_one_fixed_partition', 'privacy_guarantee_status': 'noise_and_IKOS_certified_under_documented_assumptions', 'privacy_certificate':table['privacy_certificate'], 'release_mode':'research_simulation', 'private_release_safe':False, 'rounding_method': 'exact_per_sample_unbiased', 'binomial_sampling': 'exact_aggregate_convolution', 'integer_sum_l2_sensitivity': integer_sensitivity, 'precision': precision, 'binomial_trials': total_trials, 'requested_binomial_trials': requested_trials, 'local_binomial_trials': local_trials, 'modulus': modulus, 'coordinate_noise_sd': coordinate_noise_sd, 'soft_threshold': threshold, **{key: table[key] for key in ('threshold_rule', 'threshold_noise_scale', 'threshold_kappa_infinity', 'threshold_statistical_component', 'threshold_private_component')}, 'solver': 'appendix_direct_solve', 'postprivate_solver_fallback':solver_fallback, 'raw_beta_l1': float(np.linalg.norm(beta_raw, 1)), 'raw_beta_l2': float(np.linalg.norm(beta_raw, 2)), 'thresholded_beta_l1': thresholded_l1, 'final_beta_l1': final_l1, 'l1_projection_active': bool(thresholded_l1 > config.l1_radius + 1e-12), 'raw_min_eigenvalue': float(eigenvalues.min()), 'raw_max_eigenvalue': float(eigenvalues.max()), 'raw_min_abs_eigenvalue': float(np.min(np.abs(eigenvalues))), 'raw_matrix_positive_definite': bool(np.all(eigenvalues > 0)), 'raw_matrix_condition_number': float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))) if np.min(np.abs(eigenvalues)) > 0 else None, 'eigenvalues_floored': 0, 'ikos_shuffled_shares_per_coordinate': table['shuffled_shares'], 'ikos_unshuffled_shares_per_coordinate': table['unshuffled_shares'], 'ikos_total_shares_per_coordinate': table['total_shares'], 'ikos_shuffled_streams': table['shuffled_streams'], 'ikos_direct_streams': table['direct_streams'], 'ikos_transport': 'aggregate_equivalent_not_materialized', 'parameter_rule': 'supplied_B2_table'})
