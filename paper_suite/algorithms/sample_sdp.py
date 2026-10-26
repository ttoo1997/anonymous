from __future__ import annotations
import math
import numpy as np
from .sample_sdp_core import Config,sdp2_precision,prepare_exact_sample_statistics,fit_sdp2_from_sum
from ..config import SAMPLE_SDP_CLIP_CONSTANT, SAMPLE_SDP_THRESHOLD_CONSTANT
from dp_safety.parameters import APPROXIMATE_DP_DELTA

def fit_sample_sdp(user_x: np.ndarray, user_y: np.ndarray, epsilon: float,
                   seed: int, prepared=None, *, delta: float = APPROXIMATE_DP_DELTA, clip_constant: float = SAMPLE_SDP_CLIP_CONSTANT, threshold_constant: float = SAMPLE_SDP_THRESHOLD_CONSTANT, noise_scale: float = 0.25, kappa_infinity: float = 1.0) -> tuple[np.ndarray, dict]:
    'Existing sample-level SDP2 on all n*m rows, as in the original experiment.'
    x, y = user_x.reshape(-1, user_x.shape[-1]), user_y.ravel()
    n, d = x.shape
    config = Config(delta=delta,
                    sdp2_clip=clip_constant*d*math.log(n), sdp2_clip_constant=clip_constant, sdp2_threshold_constant=threshold_constant,
                    sdp2_dimension=d, l1_radius=1.0,
                    sdp2_noise_scale=noise_scale, sdp2_kappa_infinity=kappa_infinity)
    chunk = max(1, min(64, 2_000_000 // (d*d)))
    precision = sdp2_precision(n, epsilon, config)
    if prepared is None:
        prepared=prepare_exact_sample_statistics(x,y,config.sdp2_clip,precision,np.random.default_rng(seed+1),chunk_size=chunk)
    meta=prepared['meta'];rounded=np.asarray(prepared['rounded'])
    expected_dim=d*(d+1)//2+d
    if (meta['D']!=expected_dim or rounded.shape!=(expected_dim,) or
            not np.issubdtype(rounded.dtype,np.signedinteger)):
        raise ValueError('prepared rounded sums must be a signed integer statistic vector')
    if (meta['n'],meta['d'])!=(n,d):raise ValueError('prepared statistics shape mismatch')
    if prepared['precision']!=precision or prepared['clip']!=config.sdp2_clip:raise ValueError('prepared statistics parameter mismatch')
    stat=np.zeros(meta['D'])
    beta, details = fit_sdp2_from_sum(stat, meta, epsilon, config,
                                      np.random.default_rng(seed + 2), rounded)
    return beta, {**details, "privacy_unit": "sample-level replace-one; all n*m rows",
                  "unused_rows_per_user": 0, "l1_constraint_radius": 1.0,
                  "clip_constant":clip_constant,"threshold_constant":threshold_constant,
                  "solver_variant":"appendix_direct",
                  "simulation_version":"sample_sdp_table2_threshold_20261021"}
