'Finite-parameter checks for the mechanism.'
from dataclasses import replace
import math

from .core import make_table_aggregation_parameters


def rr_zero_event_check(n, epsilon, delta, gamma):
    'Neighboring one-hot populations: zero vs one vote for a fixed coordinate.'
    log_p = n * math.log1p(-gamma)
    log_ratio = epsilon + math.log(gamma) - math.log1p(-gamma)
    violation = math.exp(log_p) * (-math.expm1(log_ratio)) if log_ratio < 0 else 0.0
    return {"event": "zero count for a coordinate; zero vs one contributing vote",
            "delta_lower_bound": violation, "requested_delta": delta,
            "status": "violated" if violation > delta else "not_refuted_by_this_event"}


def checked_aggregation_parameters(n_users, dimension, radius, n_total, budget, precision_constant):
    "Keep reference constants without undercutting Table 5's unsimplified bound."
    params = make_table_aggregation_parameters(
        n_users, dimension, radius, n_total, budget, precision_constant)
    sensitivity = max(params.integer_sensitivity,
                      2*params.precision*radius + math.sqrt(dimension) + 2)
    requested = max(2, math.ceil(32*(sensitivity/budget.epsilon)**2 * math.log(4/budget.delta)))
    local = 2*((requested+2*n_users-1)//(2*n_users))
    total = n_users*local
    modulus = 2*(n_users*math.ceil(params.precision*radius)+total//2)+1
    shares = max(3, math.ceil((2*math.log2(8*dimension/budget.delta)+math.log2(modulus))/
                              (math.log2(n_users)-math.log2(math.e)))+1)
    result = replace(params, integer_sensitivity=sensitivity, requested_trials=requested,
                     local_trials=local, total_trials=total, modulus=modulus, shares=shares)
    result.validate()
    return result
