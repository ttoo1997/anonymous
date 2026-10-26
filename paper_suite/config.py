from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from dp_safety.parameters import APPROXIMATE_DP_DELTA


EPSILON_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
MAIN_SYNTHETIC_N = 15000
MAIN_SYNTHETIC_M = 50
MAIN_SYNTHETIC_D = 256
MAIN_SYNTHETIC_K = 8
MAIN_SYNTHETIC_NOISE_STD = 0.25
DIMENSION_GRID = (64, 128, 256, 512)


USER_GRID = (8000, 10000, 12000, 15000, 18000, 20000)
LOCAL_SAMPLE_GRID = (25, 50, 75, 100)
REAL_LOCAL_SAMPLE_GRID = (10, 50, 100)
REAL_EPSILON_GRID = (1.0, 4.0)
REAL_FORMAL_ALGORITHMS = ("2slr", "ldpiht", "lasso")
FIXED_EPSILON = 4.0


SAMPLE_SDP_CLIP_CONSTANT = 0.1
SAMPLE_SDP_THRESHOLD_CONSTANT = 2.0


@dataclass(frozen=True)
class SyntheticSetting:
    n: int = MAIN_SYNTHETIC_N
    m: int = MAIN_SYNTHETIC_M
    d: int = MAIN_SYNTHETIC_D
    k: int = MAIN_SYNTHETIC_K
    epsilon: float = FIXED_EPSILON
    delta: float = APPROXIMATE_DP_DELTA
    noise_std: float = MAIN_SYNTHETIC_NOISE_STD

    def validate(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (self.n, self.m, self.d, self.k)):
            raise ValueError("n, m, d and k must be integers")
        if min(self.n, self.m, self.d, self.k) < 1 or self.k > self.d:
            raise ValueError("n, m, d and k must be positive, with k <= d")
        if not all(math.isfinite(v) for v in (self.epsilon, self.delta, self.noise_std)) or self.epsilon <= 0 or not 0 < self.delta < 1 or self.noise_std < 0:
            raise ValueError("invalid privacy or noise parameter")


@dataclass(frozen=True)
class RealSetting:
    'Algorithm-facing setting for Exp5 real-data runs.'
    n: int
    m: int
    d: int
    k: int
    epsilon: float
    delta: float = APPROXIMATE_DP_DELTA

    def validate(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (self.n, self.m, self.d, self.k)):
            raise ValueError("n, m, d and k must be integers")
        if min(self.n, self.m, self.d, self.k) < 1 or self.k > self.d:
            raise ValueError("n, m, d and k must be positive, with k <= d")
        if not all(math.isfinite(v) for v in (self.epsilon, self.delta)) or self.epsilon <= 0 or not 0 < self.delta < 1:
            raise ValueError("invalid privacy parameter")


def synthetic_grid(experiment: str) -> list[SyntheticSetting]:
    if experiment == "exp1_epsilon":
        return [SyntheticSetting(epsilon=e) for e in EPSILON_GRID]
    if experiment == "exp2_dimension":

        return [SyntheticSetting(d=d, k=MAIN_SYNTHETIC_K, epsilon=FIXED_EPSILON)
                for d in DIMENSION_GRID]
    if experiment == "exp3_users":
        return [SyntheticSetting(n=n, epsilon=FIXED_EPSILON) for n in USER_GRID]
    if experiment == "exp4a_recall_m":
        return [SyntheticSetting(m=m, epsilon=FIXED_EPSILON) for m in LOCAL_SAMPLE_GRID]
    if experiment == "exp4b_recall_n":
        return [SyntheticSetting(n=n, epsilon=FIXED_EPSILON) for n in USER_GRID]
    raise ValueError(f"unknown experiment: {experiment}")


def validate_protocol_grid(experiment: str, settings: list[SyntheticSetting]) -> None:
    'Enforce the cross-panel canonical synthetic protocol.'
    for setting in settings:
        setting.validate()
    if not settings:
        raise AssertionError("experiment grid must be non-empty")
    if any(s.k != MAIN_SYNTHETIC_K or s.noise_std != MAIN_SYNTHETIC_NOISE_STD or s.delta != APPROXIMATE_DP_DELTA
           for s in settings):
        raise AssertionError("k=8, sigma=0.25 and delta=1e-5 must be common across Exp1--Exp4")
    if experiment != "exp1_epsilon" and any(s.epsilon != FIXED_EPSILON for s in settings):
        raise AssertionError("epsilon must stay fixed at 4 when another variable changes")

    if experiment == "exp1_epsilon":
        if any((s.n, s.m, s.d) != (MAIN_SYNTHETIC_N, MAIN_SYNTHETIC_M, MAIN_SYNTHETIC_D) for s in settings):
            raise AssertionError("Exp1 may vary epsilon only")
    elif experiment == "exp2_dimension":
        if any((s.n, s.m) != (MAIN_SYNTHETIC_N, MAIN_SYNTHETIC_M) for s in settings):
            raise AssertionError("Exp2 may vary d only")
    elif experiment == "exp3_users":
        if any((s.m, s.d) != (MAIN_SYNTHETIC_M, MAIN_SYNTHETIC_D) for s in settings):
            raise AssertionError("Exp3 may vary n only")
    elif experiment == "exp4a_recall_m":
        if any((s.n, s.d) != (MAIN_SYNTHETIC_N, MAIN_SYNTHETIC_D) for s in settings):
            raise AssertionError("Exp4a may vary m only")
    elif experiment == "exp4b_recall_n":
        if any((s.m, s.d) != (MAIN_SYNTHETIC_M, MAIN_SYNTHETIC_D) for s in settings):
            raise AssertionError("Exp4b may vary n only")
    else:
        raise ValueError(f"unknown experiment: {experiment}")
