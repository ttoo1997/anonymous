"""Deterministic synthetic regression test cases; no real observations."""
import numpy as np


def case(seed=1, n=64, p=12, correlated=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n,p))
    if correlated:
        X[:,1] = .995*X[:,0] + .005*rng.normal(size=n)
        X[:,3] = .98*X[:,2] + .02*rng.normal(size=n)
    true = np.zeros(p)
    true[0] = .6
    true[2] = -.35
    y = X@true + .07*rng.normal(size=n)
    return X,y
