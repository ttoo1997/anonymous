'SampleSDP-Ridge: unchanged private statistics, explicit server postprocessing.'
import math
import numpy as np
from paper_suite.algorithms import sample_sdp_core as core


def release_statistics(rounded, statistic_meta, epsilon, config, rng):
    'Exactly reproduce the frozen integer query, noise and modular decoding.'
    n, d, D = (int(statistic_meta[k]) for k in ('n', 'd', 'D'))
    if d != config.sdp2_dimension or D != d * (d + 1) // 2 + d:
        raise ValueError('inconsistent statistic dimensions')
    if not math.isclose(config.l1_radius, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError('SampleSDP-Ridge retains the unit L1 ball')
    rounded = np.asarray(rounded)
    if rounded.shape != (D,) or not np.issubdtype(rounded.dtype, np.integer):
        raise ValueError('exact rounded integer vector required')
    table = core.sdp2_table_parameters(n, d, epsilon, config.delta,
        config.sdp2_log_population or n,
        clip_constant=config.sdp2_clip_constant,
        threshold_constant=config.sdp2_threshold_constant,
        noise_scale=config.sdp2_noise_scale,kappa_infinity=config.sdp2_kappa_infinity)
    if not math.isclose(config.sdp2_clip, table['clip'], rel_tol=1e-12):
        raise ValueError('clip radius differs from original table')
    noise = core.exact_aggregate_binomial_noise_sum(n, D, table['local_trials'], rng)

    if np.max(np.abs(rounded.astype(float)), initial=0.) > n * math.ceil(table['precision'] * table['clip']):
        raise ValueError('rounded statistic exceeds public query bound')
    modular = np.mod(rounded.astype(np.int64) + noise, table['modulus'])
    signed = np.where(modular > table['modulus'] // 2, modular - table['modulus'], modular)
    z = signed.astype(float) / (n * table['precision'])
    upper = np.triu_indices(d)
    G = np.zeros((d, d)); G[upper] = z[:len(upper[0])]
    G[upper[1], upper[0]] = G[upper]
    h = z[len(upper[0]):]
    meta = {**statistic_meta, 'privacy_certificate': table['privacy_certificate'],
        'privacy_target': 'sample_replace_one_fixed_partition',
        'privacy_guarantee_status': 'unchanged_certified_statistics_then_postprocessing',
        'release_mode': 'research_simulation', 'private_release_safe': False,
        'rounding_method': 'exact_per_sample_unbiased',
        'binomial_sampling': 'exact_aggregate_convolution',
        'integer_sum_l2_sensitivity': table['integer_sensitivity'],
        'precision': table['precision'], 'binomial_trials': table['total_trials'],
        'requested_binomial_trials': table['requested_trials'],
        'local_binomial_trials': table['local_trials'], 'modulus': table['modulus'],
        'coordinate_noise_sd': math.sqrt(table['total_trials']) / (2 * n * table['precision']),
        'soft_threshold': table['soft_threshold'],
        **{key:table[key] for key in ('threshold_rule','threshold_noise_scale','threshold_kappa_infinity',
                                    'threshold_statistical_component','threshold_private_component')},
        'ikos_shuffled_shares_per_coordinate': table['shuffled_shares'],
        'ikos_unshuffled_shares_per_coordinate': table['unshuffled_shares'],
        'ikos_total_shares_per_coordinate': table['total_shares'],
        'ikos_shuffled_streams': table['shuffled_streams'],
        'ikos_direct_streams': table['direct_streams'],
        'ikos_transport': 'aggregate_equivalent_not_materialized',
        'parameter_rule': 'supplied_B2_table'}
    return G, h, meta


def ridge_from_eigh(eigenvalues, eigenvectors, h, threshold, noise_sd, multiplier):
    'Postprocess ONLY private G/h and public parameters.'
    values = np.asarray(eigenvalues, float); Q = np.asarray(eigenvectors, float)
    h = np.asarray(h, float); d = len(h)
    if values.shape != (d,) or Q.shape != (d, d) or d < 1:
        raise ValueError('invalid eigensystem shapes')
    if not all(np.isfinite(x).all() for x in (values, Q, h)):
        raise ValueError('nonfinite private statistics')
    if not (math.isfinite(multiplier) and multiplier > 0 and
            math.isfinite(noise_sd) and noise_sd > 0 and
            math.isfinite(threshold) and threshold >= 0):
        raise ValueError('positive finite Ridge multiplier/noise scale and nonnegative threshold required')

    numerical_floor = np.finfo(float).eps * max(1., float(np.max(np.abs(values)))) * 64
    ridge = max(multiplier * math.sqrt(d) * noise_sd, numerical_floor)
    curvature = max(0., -float(values.min()))
    denominators = values + curvature + ridge
    raw = Q @ ((Q.T @ h) / denominators)
    soft = np.sign(raw) * np.maximum(np.abs(raw) - threshold, 0.)
    beta = core.project_l1_ball(soft, 1.)
    if not np.isfinite(beta).all():
        raise FloatingPointError('nonfinite Ridge postprocessing')
    return beta, dict(estimator_variant='SampleSDP-Ridge', solver='uniform_shift_ridge',
        ridge_noise_multiplier=float(multiplier), ridge_lambda=float(ridge),
        negative_curvature_shift=curvature, diagonal_shift_total=curvature + ridge,
        regularized_min_eigenvalue=float(denominators.min()),
        regularized_condition_number=float(denominators.max() / denominators.min()),
        raw_min_eigenvalue=float(values.min()), raw_max_eigenvalue=float(values.max()),
        raw_beta_l1=float(np.linalg.norm(raw, 1)), raw_beta_l2=float(np.linalg.norm(raw)),
        thresholded_beta_l1=float(np.linalg.norm(soft, 1)), final_beta_l1=float(np.linalg.norm(beta, 1)),
        l1_projection_active=bool(np.linalg.norm(soft, 1) > 1. + 1e-12),
        eigenvalues_floored=0, postprivate_solver_fallback=False,
        privacy_postprocessing_only=True, original_utility_theorem_preserved=False)


def fit_from_sum(rounded, statistic_meta, epsilon, config, rng, multiplier):
    G, h, meta = release_statistics(rounded, statistic_meta, epsilon, config, rng)
    values, Q = np.linalg.eigh(G)
    beta, details = ridge_from_eigh(values, Q, h, meta['soft_threshold'],
                                   meta['coordinate_noise_sd'], multiplier)
    return beta, {**meta, **details}
