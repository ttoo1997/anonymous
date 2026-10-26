'Fixed LDP-IHT comparison parameter profile.'

PROFILE = "zhu24_algorithm2_ma24_grid_fixed_T2_eta0p01_tau2_k10_v3_unscaled"


_SELECTED = dict(iterations=2, step_size=0.01, x_clip=2.0, y_clip=2.0,
                 sparsity=10, vectorized=True)


def selected_parameters(k: int, participant_count: int, sigma_scale: float = 1.0) -> dict:
    if k < 1 or participant_count < 2:
        raise ValueError("positive k and at least two participants required")


    return dict(_SELECTED)


def profile_name(k: int | None = None) -> str:
    return PROFILE
