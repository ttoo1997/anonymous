'Public-parameter certificate for integer binomial noise plus IKOS.'
from __future__ import annotations
import math
from dataclasses import asdict,dataclass


def binomial_epsilon_bound(dimension:int, sensitivity_l2:float,
                           total_trials:int, delta_noise:float)->float:
    if not isinstance(dimension,int) or dimension<1:
        raise ValueError('dimension must be a positive integer')
    if not isinstance(total_trials,int) or total_trials<2 or total_trials%2:
        raise ValueError('centered integer noise requires a positive even trial count')
    if not (math.isfinite(sensitivity_l2) and sensitivity_l2>0 and 0<delta_noise<1):
        raise ValueError('positive finite sensitivity and delta in (0,1) required')
    D=dimension;D2=sensitivity_l2;D1=math.sqrt(D)*D2;Dinf=D2;V=total_trials/4
    if V<max(23*math.log(10*D/delta_noise),2*Dinf):
        return math.inf
    t1=D2*math.sqrt(2*math.log(1.25/delta_noise))/math.sqrt(V)
    t2=(2.5*D2*math.sqrt(math.log(10/delta_noise))+D1/3)/(V*(1-delta_noise/10))
    t3=((2/3)*Dinf*math.log(1.25/delta_noise)+(2/3)*Dinf*math.log(20*D/delta_noise)*math.log(10/delta_noise))/V
    return math.nextafter((t1+t2+t3)*(1.0+1e-12), math.inf)


@dataclass(frozen=True)
class Certificate:
    epsilon_target:float
    delta_target:float
    epsilon_noise_certified:float
    delta_noise:float
    shuffled_shares:int
    unshuffled_shares:int
    security_bits_actual:float
    tv_distance_bound:float
    delta_total_bound:float


def certify_existing_noise(*,n_users:int,dimension:int,modulus:int,
                          total_trials:int,sensitivity_l2:float,
                          epsilon:float,delta:float,
                          current_shares:int=3,
                          use_tighter_noise_epsilon:bool=False)->Certificate:
    'Keep p, radius, M, and q; only increase the number of shuffled shares.'
    if (not isinstance(n_users,int) or n_users<19 or not isinstance(modulus,int)
        or modulus<3 or not isinstance(current_shares,int) or current_shares<3
        or not math.isfinite(epsilon) or epsilon<=0 or not 0<delta<1):
        raise ValueError('invalid public aggregation parameters')
    dn=delta/2
    en=binomial_epsilon_bound(dimension,sensitivity_l2,total_trials,dn)
    if not math.isfinite(en) or en>epsilon:
        raise ValueError('existing binomial noise fails this sufficient certificate; increase M')
    ev=en if use_tighter_noise_epsilon else epsilon
    log1pexp=ev+math.log1p(math.exp(-ev))
    bits=math.ceil((math.log(2*dimension/delta)+log1pexp)/math.log(2))
    denominator=math.log2(n_users)-math.log2(math.e)
    shares=max(current_shares,3,math.ceil((2*bits+math.log2(modulus))/denominator)+1)
    actual=((shares-1)*denominator-math.log2(modulus))/2
    logmu=math.log(dimension)-actual*math.log(2)
    mu=math.exp(logmu)
    total=dn+math.exp(log1pexp+logmu)
    if total>delta:raise ArithmeticError('internal security-budget check failed')
    return Certificate(epsilon,delta,en,dn,shares,1,actual,mu,total)


def security_bits(epsilon: float, delta: float, dimension: int) -> int:
    'Conservative integer per-coordinate security parameter.'
    return max(1, math.ceil((math.log(2*dimension/delta)+epsilon+math.log1p(math.exp(-epsilon)))/math.log(2)))


def ensure_local_trials(n_users: int, dimension: int, sensitivity_l2: float,
                        epsilon: float, delta: float, initial: int) -> int:
    "Public-only calibration. Keep the table's N0 whenever it certifies."
    if (not isinstance(n_users,int) or n_users<19 or not isinstance(initial,int)
        or initial<2 or initial%2 or not math.isfinite(epsilon) or epsilon<=0
        or not 0<delta<1):raise ValueError('invalid public noise calibration input')
    cap=(2**53-1)//n_users//2
    start=initial//2
    if start>cap:raise ValueError('binomial trials exceed the safe sampling limit')
    def passes(h):
        return binomial_epsilon_bound(dimension,sensitivity_l2,n_users*2*h,delta/2)<=epsilon
    if passes(start):return initial
    lo=start;hi=start
    while not passes(hi):
        if hi>=cap:raise ValueError('privacy calibration exceeds the safe sampling limit')
        hi=min(cap,max(hi+1,hi*2))
    while hi-lo>1:
        mid=(hi+lo)//2
        if passes(mid):hi=mid
        else:lo=mid
    return 2*hi
