from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .config import synthetic_grid, validate_protocol_grid, USER_GRID
from .storage import ResultStore
from .data import make_synthetic
from .runner import ALGORITHMS, DEFAULT_ALGORITHMS, evaluate, validate_request


EXPERIMENTS = ("exp1_epsilon", "exp2_dimension", "exp3_users",
               "exp4a_recall_m", "exp4b_recall_n")


def run(experiments: list[str], algorithms: list[str], repeats: int, out: Path) -> None:
    validate_request(algorithms, repeats, out)
    if not experiments or len(set(experiments)) != len(experiments):
        raise ValueError("distinct experiments are required")
    if any(e not in EXPERIMENTS for e in experiments):
        raise ValueError("unknown experiment")
    if any(e.startswith("exp4") for e in experiments) and not any(
            a in ("proposed", "2slr") for a in algorithms):
        raise ValueError("Exp4 requires at least one stage-1 algorithm")
    grids = {experiment: synthetic_grid(experiment) for experiment in experiments}
    for experiment, settings in grids.items():
        validate_protocol_grid(experiment, settings)
    stage1_names = [a for a in algorithms if a in ("proposed", "2slr")]
    expected = sum(len(settings)*repeats*(len(stage1_names) if e.startswith("exp4") else len(algorithms))
                   for e,settings in grids.items())
    protocol = {"source": "SDP_exp.pdf", "repeats": repeats,
                "settings": {e:[asdict(s) for s in grid] for e,grid in grids.items()},
                "algorithms": algorithms, "expected_cells": expected, "plot_generation": False,
                "epsilon_rule": "epsilon=4 whenever n, m, or d changes",
                "canonical_anchor": {"n": 15000, "m": 50, "d": 256, "k": 8, "epsilon": 4.0, "noise_std": 0.25},
                "user_grid": list(USER_GRID),
                "coefficient_rule": "equal_positive_l1_v1: exactly k active coordinates, each +1/k in every synthetic panel",
                "notes": {"ldpiht": "Zhu24 Algorithm 2; sample-level LDP on all n*m rows; fixed Ma24 Appendix-E-grid profile T=2, k_prime=10, eta=0.01, tau1=tau2=2; optional 8-point synthetic diagnostic grid at epsilon=4; does not modify the fixed profile; Exp4 excludes LDP-IHT because it has no separate support-selection stage",
                          "2slr": "Ma24 Algorithm 1 + Algorithms 7-9 paper-aligned: Appendix-E Lasso after 64-feature positive marginal screening; frozen synthetic signals are positive; one RR report/user in HeavyHitter; disjoint Range/Mean groups; project-before-Laplace; paper E.1 parameter grid exposed; fixed reference profile uses HeavyHitter alpha=0.05, chosen before final benchmark after a separate Stage-1 diagnostic",
                          "proposed": "Appendix B v5 DP-certified Stage 1: main n=15000; n1=n2=7500; C_gamma=32 reference floor plus finite-sample certified gamma; Table-5 full sensitivity",
                          "sample_sdp": "all n*m rows; Algorithm 2 direct solve; c0=0.1, c1=2; equal_positive_l1_v1 already satisfies unit-L1 bound; no response scaling",
                          "noise_std": "frozen synthetic residual standard deviation 0.25; Proposed Table-4/5 sigma uses the same value"}}
    with ResultStore(out, protocol) as store:
        for experiment, settings in grids.items():
            recall_only = experiment.startswith("exp4")
            names = stage1_names if recall_only else algorithms
            for setting in settings:
                for repeat in range(repeats):
                    seed = 20260915 + repeat
                    data = make_synthetic(setting, seed)
                    for name in names:
                        base = {"experiment": experiment, "algorithm": name, "repeat": repeat,
                                "seed": seed, **asdict(setting)}
                        store.append({**base, **evaluate(name, data, setting, seed+100000, recall_only)})
    return store.protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDP_exp.pdf synthetic experiments; writes tables only.")
    parser.add_argument("--experiment", choices=EXPERIMENTS, action="append")
    parser.add_argument("--algorithm", choices=ALGORITHMS, action="append")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment or list(EXPERIMENTS), args.algorithm or list(DEFAULT_ALGORITHMS), args.repeats, args.out)


if __name__ == "__main__":
    main()
