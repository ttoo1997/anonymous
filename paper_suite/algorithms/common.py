from __future__ import annotations

import math
import numpy as np
from dp_safety.rr import rr_keep_probability, rr_debias_factor


def hard_threshold(vector: np.ndarray, sparsity: int) -> np.ndarray:
    if sparsity < 1:
        raise ValueError("sparsity must be positive")
    if sparsity >= len(vector):
        return vector.copy()
    keep = np.argpartition(np.abs(vector), -sparsity)[-sparsity:]
    result = np.zeros_like(vector)
    result[keep] = vector[keep]
    return result


def project_l2(vector: np.ndarray, radius: float = 1.0) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm <= radius else vector * (radius / norm)


def privunit(vector: np.ndarray, radius: float, epsilon: float,
             rng: np.random.Generator) -> np.ndarray:
    'Unbiased pure-epsilon LDP vector randomizer based on a hemisphere mechanism.'
    vector = np.asarray(vector, dtype=float)
    if radius <= 0 or epsilon <= 0 or vector.ndim != 1:
        raise ValueError("positive radius/epsilon and a vector are required")
    norm = float(np.linalg.norm(vector))
    if norm > radius * (1 + 1e-10):
        raise ValueError("input exceeds the declared L2 radius")
    direction = vector / norm if norm else rng.normal(size=len(vector))
    direction /= np.linalg.norm(direction)
    u = rng.normal(size=len(vector))
    u /= np.linalg.norm(u)
    if np.dot(u, direction) < 0:
        u = -u
    data_sign = 1.0 if rng.random() < 0.5 * (1.0 + norm / radius) else -1.0
    privacy_prob = rr_keep_probability(epsilon)
    privacy_sign = 1.0 if rng.random() < privacy_prob else -1.0
    log_c = math.lgamma(len(vector) / 2) - 0.5 * math.log(math.pi) - math.lgamma((len(vector) + 1) / 2)
    hemisphere_mean = math.exp(log_c)
    scale = radius * rr_debias_factor(epsilon) / hemisphere_mean
    return scale * data_sign * privacy_sign * u


def privunit_batch(vectors, radius, epsilon, rng):
    'Independent hemisphere reports, vectorized; same law as privunit.'
    v=np.asarray(vectors,dtype=float)
    if v.ndim!=2 or v.shape[1]<1 or not np.isfinite(v).all() or not math.isfinite(radius) or not math.isfinite(epsilon) or radius<=0 or epsilon<=0:
        raise ValueError('finite matrix and positive radius/epsilon required')
    norms=np.linalg.norm(v,axis=1)
    if np.any(norms>radius*(1+1e-10)):raise ValueError('input exceeds radius')
    direction=v/np.where(norms>0,norms,1.)[:,None]
    direction[norms==0,0]=1.
    u=rng.normal(size=v.shape);u/=np.linalg.norm(u,axis=1)[:,None]
    u*=np.where(np.sum(u*direction,axis=1)<0,-1.,1.)[:,None]
    signs=np.where(rng.random(len(v))<.5*(1+norms/radius),1.,-1.)
    signs*=np.where(rng.random(len(v))<rr_keep_probability(epsilon),1.,-1.)
    d=v.shape[1]
    c=math.exp(math.lgamma(d/2)-.5*math.log(math.pi)-math.lgamma((d+1)/2))
    return u*(signs*radius*rr_debias_factor(epsilon)/c)[:,None]
