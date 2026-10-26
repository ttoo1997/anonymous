'Public, prelocked LDPIHT stabilization policy; not the original utility theorem.'
import math
PROFILE = "ldpiht_gradient_clipped_stabilization_v1"

def stabilized_parameters(k: int, participant_count: int, epsilon: float) -> dict:
    if k < 1 or participant_count < 2 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive k, epsilon and two records required")
    low = epsilon < 4.0
    return dict(iterations=2, step_size=1.5 if low else 2.0,
                x_clip=1.5 if low else 1.0, y_clip=.75 if low else .5,
                sparsity=10 if low else 8, vectorized=True,
                gradient_clip=4.0, radius_mode="public_iterate")
