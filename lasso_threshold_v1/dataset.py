'Canonical synthetic signal protocol for the frozen comparison suite.'
from numbers import Integral
import numpy as np

COEFFICIENT_PROTOCOL='equal_positive_l1_v1'


def coefficient_values(k):
    if isinstance(k,(bool,np.bool_)) or not isinstance(k,Integral) or k<1:
        raise ValueError('positive integer k required')

    return np.full(int(k),1.0/float(k),dtype=float)


def public_signal_lower_bound(k):
    'Public min |beta_j| under equal_positive_l1_v1.'
    return 1.0/float(k)


def public_parameter_l2_norm(k):
    'Public ||beta||_2 under equal_positive_l1_v1.'
    return 1.0/np.sqrt(float(k))


def public_parameter_l1_norm(k):
    'Public ||beta||_1 under equal_positive_l1_v1.'
    coefficient_values(k)
    return 1.0


def synthetic(n,m,d,k,seed,test_samples,noise_std=0.25):
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) or v<1 for v in (n,m,d,k,test_samples)) or k>d:
        raise ValueError('invalid synthetic dimensions')
    if not np.isfinite(noise_std) or noise_std<0:raise ValueError('invalid noise_std')
    rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    support=np.sort(rng[0].choice(d,k,replace=False))
    beta=np.zeros(d);beta[support]=coefficient_values(k)
    x=rng[1].normal(size=(n,m,d));y=x@beta+rng[2].normal(scale=noise_std,size=(n,m))
    xt=rng[3].normal(size=(test_samples,d));yt=xt@beta+rng[4].normal(scale=noise_std,size=test_samples)
    return x,y,xt,yt,beta
