'Finite-sample privacy accountant for Appendix-B Algorithm 3.'
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

CHEU_REFERENCE = (
    "Cheu, Smith, Ullman, Zeber, Zhilyaev, Distributed Differential Privacy "
    "via Shuffling, EUROCRYPT 2019, Algorithm 1 and Corollary 4.7"
)
LOCAL_REFERENCE = "exact pure-LDP likelihood-ratio bound for the two changed RR bits"
CERTIFICATION_RELATIVE_MARGIN = 1e-10


@dataclass(frozen=True)
class Stage1PrivacyCertificate:
    n_selection: int
    epsilon: float
    delta: float
    appendix_c_gamma: float
    appendix_gamma_floor: float
    certified_gamma_floor: float
    gamma: float
    certificate_mode: str
    certified_total_epsilon_bound: float
    certified_total_delta_bound: float
    per_stream_epsilon_bound: float | None
    per_stream_delta: float | None
    cheu_lambda: float | None
    local_vector_epsilon_bound: float
    equivalent_c_gamma: float
    certificate_source: str

    def as_dict(self) -> dict:
        return asdict(self)


def _validate(n_selection: int, epsilon: float, delta: float) -> None:
    if isinstance(n_selection, bool) or not isinstance(n_selection, int) or n_selection < 2:
        raise ValueError("n_selection must be an integer >= 2")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")


def appendix_gamma_floor(n_selection: int, epsilon: float, delta: float, c_gamma: float) -> float:
    _validate(n_selection, epsilon, delta)
    if not math.isfinite(c_gamma) or c_gamma <= 0:
        raise ValueError("c_gamma must be positive and finite")
    return c_gamma * math.log(1.0 / delta) / (n_selection * epsilon**2)


def local_vector_epsilon(gamma: float) -> float:
    if not math.isfinite(gamma) or not 0 < gamma < 0.5:
        raise ValueError("gamma must lie in (0,1/2)")
    return 2.0 * math.log((1.0 - gamma) / gamma)


def local_vector_gamma_floor(epsilon: float) -> float:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    z = math.exp(-epsilon / 2.0)
    return z / (1.0 + z)


def cheu_stream_epsilon_bound(gamma: float, n_selection: int, delta_stream: float) -> float:
    'Corollary 4.7 bound for one independently shuffled binary RR stream.'
    if not math.isfinite(gamma) or not 0 < gamma < 0.5:
        return math.inf
    if isinstance(n_selection, bool) or not isinstance(n_selection, int) or n_selection < 2:
        return math.inf
    if not math.isfinite(delta_stream) or not 0 < delta_stream < 1:
        return math.inf
    lam = 2.0 * n_selection * gamma
    lam_min = 14.0 * math.log(4.0 / delta_stream)
    if lam < lam_min or lam > n_selection:
        return math.inf
    root = math.sqrt(2.0 * lam * math.log(2.0 / delta_stream))
    effective = lam - root
    if effective <= 0:
        return math.inf
    multiplier = 1.0 - effective / n_selection
    if multiplier < 0:
        return math.inf
    return math.sqrt(32.0 * math.log(4.0 / delta_stream) / effective) * multiplier


def _cheu_composed_gamma_floor(n_selection: int, epsilon: float, delta: float) -> tuple[float, float, float] | None:
    'Smallest gamma certified by Cor. 4.7 + two-stream basic composition.'
    eps_stream = epsilon / 2.0
    delta_stream = delta / 2.0
    lam_min = 14.0 * math.log(4.0 / delta_stream)
    if lam_min >= n_selection:
        return None
    lo = lam_min
    hi = math.nextafter(float(n_selection), 0.0)
    eps_hi = cheu_stream_epsilon_bound(hi / (2.0 * n_selection), n_selection, delta_stream)
    if eps_hi > eps_stream:
        return None
    eps_lo = cheu_stream_epsilon_bound(lo / (2.0 * n_selection), n_selection, delta_stream)
    if eps_lo <= eps_stream:
        lam = lo
    else:
        for _ in range(100):
            mid = (lo + hi) / 2.0
            bound = cheu_stream_epsilon_bound(mid / (2.0 * n_selection), n_selection, delta_stream)
            if bound <= eps_stream:
                hi = mid
            else:
                lo = mid
        lam = hi


    lam = min(math.nextafter(float(n_selection), 0.0), lam * (1.0 + CERTIFICATION_RELATIVE_MARGIN))
    gamma = lam / (2.0 * n_selection)
    bound = cheu_stream_epsilon_bound(gamma, n_selection, delta_stream)
    if not math.isfinite(bound) or bound > eps_stream * (1.0 + 1e-10):
        raise RuntimeError("Cheu Stage-1 privacy calibration failed its own certificate")
    return gamma, bound, lam


def certify_stage1_gamma(
    n_selection: int,
    epsilon: float,
    delta: float,
    *,
    appendix_c_gamma: float = 32.0,
) -> Stage1PrivacyCertificate:
    'Return a privacy-certified running gamma for Algorithm 3.'
    _validate(n_selection, epsilon, delta)
    legacy = appendix_gamma_floor(n_selection, epsilon, delta, appendix_c_gamma)
    if not 0 < legacy < 0.5:
        raise ValueError(
            "Appendix-B C_gamma reference window infeasible: legacy gamma floor must lie in (0,1/2)"
        )

    local_floor = local_vector_gamma_floor(epsilon)
    local_floor = min(math.nextafter(0.5, 0.0), local_floor * (1.0 + CERTIFICATION_RELATIVE_MARGIN))
    candidates: list[tuple[float, str, float | None, float | None, float | None, str]] = [
        (local_floor, "pure_local_vector", None, None, None, LOCAL_REFERENCE)
    ]
    cheu = _cheu_composed_gamma_floor(n_selection, epsilon, delta)
    if cheu is not None:
        g, eps_stream_bound, lam = cheu
        candidates.append((g, "cheu_corollary_4_7_two_stream", eps_stream_bound, delta / 2.0, lam, CHEU_REFERENCE))

    cert_floor, mode, stream_eps, stream_delta, lam, source = min(candidates, key=lambda x: x[0])
    gamma = max(legacy, cert_floor)
    if not 0 < gamma < 0.5:
        raise ValueError("certified Stage-1 gamma must lie in (0,1/2)")

    if mode == "cheu_corollary_4_7_two_stream":
        actual_stream_eps = cheu_stream_epsilon_bound(gamma, n_selection, delta / 2.0)
        total_eps_bound = 2.0 * actual_stream_eps
        total_delta_bound = delta
        if not math.isfinite(total_eps_bound) or total_eps_bound > epsilon * (1.0 + 1e-10):
            raise RuntimeError("running gamma does not satisfy the Cheu two-stream certificate")
        stream_eps = actual_stream_eps
        stream_delta = delta / 2.0
        lam = 2.0 * n_selection * gamma
    else:
        total_eps_bound = local_vector_epsilon(gamma)
        total_delta_bound = 0.0
        if total_eps_bound > epsilon * (1.0 + 1e-10):
            raise RuntimeError("running gamma does not satisfy the pure-local certificate")
        stream_eps = None
        stream_delta = None
        lam = None

    equiv = gamma * n_selection * epsilon**2 / math.log(1.0 / delta)
    return Stage1PrivacyCertificate(
        n_selection=n_selection,
        epsilon=float(epsilon),
        delta=float(delta),
        appendix_c_gamma=float(appendix_c_gamma),
        appendix_gamma_floor=float(legacy),
        certified_gamma_floor=float(cert_floor),
        gamma=float(gamma),
        certificate_mode=mode,
        certified_total_epsilon_bound=float(total_eps_bound),
        certified_total_delta_bound=float(total_delta_bound),
        per_stream_epsilon_bound=None if stream_eps is None else float(stream_eps),
        per_stream_delta=None if stream_delta is None else float(stream_delta),
        cheu_lambda=None if lam is None else float(lam),
        local_vector_epsilon_bound=float(local_vector_epsilon(gamma)),
        equivalent_c_gamma=float(equiv),
        certificate_source=source,
    )
