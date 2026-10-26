'Conservative RR probabilities on the 53-bit NumPy uniform grid.'
from __future__ import annotations
from decimal import Decimal,localcontext,ROUND_CEILING,ROUND_FLOOR
from functools import lru_cache
import math

@lru_cache(maxsize=512)
def rr_keep_probability(epsilon:float)->float:
    epsilon=float(epsilon)
    if not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('finite positive epsilon required')
    Q=1<<53
    with localcontext() as c:
        c.prec=100
        E=Decimal.from_float(epsilon)

        t_upper=(-E).exp().next_plus()
        c.rounding=ROUND_CEILING
        denominator=Decimal(1)+t_upper
        c.rounding=ROUND_FLOOR
        p_lower=Decimal(1)/denominator
        cutoff=int((p_lower*Decimal(Q)).to_integral_value(rounding=ROUND_FLOOR))
    cutoff=max(Q//2,min(Q-1,cutoff))
    return cutoff/Q


def rr_debias_factor(epsilon: float) -> float:
    p = rr_keep_probability(epsilon)
    if p <= 0.5:
        raise ValueError("epsilon is too small for an informative 53-bit RR report")
    return 1.0 / (2.0*p - 1.0)

def round_flip_probability_up(gamma: float) -> float:
    'NumPy U<gamma realizes ceil(2^53 gamma)/2^53; use it for decoding too.'
    if not math.isfinite(gamma) or not 0 < gamma < 0.5:
        raise ValueError("flip probability must be in (0,1/2)")
    value = math.ceil(gamma*(1 << 53))/(1 << 53)
    if value >= 0.5:
        raise ValueError("unresolvable RR flip probability on the 53-bit grid")
    return value
