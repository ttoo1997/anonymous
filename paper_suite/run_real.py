from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .config import REAL_EPSILON_GRID, REAL_LOCAL_SAMPLE_GRID, REAL_FORMAL_ALGORITHMS, RealSetting
from .data import load_tabular_csv
from .datasets import DATASETS, load_registered, digest
from .storage import ResultStore
from .runner import fit_lasso, metric_row, metadata_row, evaluate, validate_request


def run(dataset: str, csv_path: Path, target: str, algorithms: list[str],
        repeats: int, out: Path) -> None:
    validate_request(algorithms, repeats, out)
    if any(name not in REAL_FORMAL_ALGORITHMS for name in algorithms):
        raise ValueError(f"Exp5 supports only {REAL_FORMAL_ALGORITHMS}; Proposed/Sample SDP require public real-data bounds not supplied by this interface")
    if csv_path is None:
        if dataset not in DATASETS:
            raise ValueError("registered dataset or --csv and --target required")
        source=DATASETS[dataset]["path"]
        target=DATASETS[dataset]["target"]
    else:
        source=csv_path
        if not target:
            raise ValueError("--target is required with --csv")
    protocol = {"source": "optional_generic_tabular_interface", "dataset": dataset, "data_path": str(source),
                "data_checksum": digest(source), "target": target,
                "m_grid": REAL_LOCAL_SAMPLE_GRID, "epsilon_grid": REAL_EPSILON_GRID,
                "repeats": repeats, "algorithms": algorithms, "formal_algorithms": list(REAL_FORMAL_ALGORITHMS),
                "expected_cells": len(REAL_LOCAL_SAMPLE_GRID)*len(REAL_EPSILON_GRID)*repeats*len(algorithms),
                "plot_generation": False, "data_preparation": [],
                "grouping_warning": "Pseudo-user grouping should be assessed separately from real subject-level privacy"}
    with ResultStore(out, protocol) as store:
        for m in REAL_LOCAL_SAMPLE_GRID:
            for repeat in range(repeats):
                seed = 20260915+repeat
                data = (load_tabular_csv(csv_path, target, m, seed) if csv_path is not None
                        else load_registered(dataset,m,seed))
                lasso_beta, lasso_meta = fit_lasso(data)
                lasso_metrics = metric_row(lasso_beta,data)
                lasso_mse = lasso_metrics["test_mse"]
                if not np.isfinite(lasso_mse) or lasso_mse <= 0:
                    raise ValueError("MSE Ratio requires a positive finite Lasso denominator")
                store.protocol["data_preparation"].append({"m":m,"repeat":repeat,**(data.metadata or {})})
                for epsilon in REAL_EPSILON_GRID:
                    setting = RealSetting(n=len(data.user_x), m=m, d=data.user_x.shape[-1],
                                            k=min(8,data.user_x.shape[-1]),epsilon=epsilon)
                    for name in algorithms:
                        if name == "lasso":
                            details = json.dumps(lasso_meta, ensure_ascii=False, allow_nan=False)
                            metrics = {"status":"ok", **lasso_metrics, **metadata_row(lasso_meta),
                                       "details_json": details,
                                       "convergence_warning": bool(lasso_meta.get("convergence_warning", False)),
                                       "convergence_warning_count": int(lasso_meta.get("convergence_warning_count", 0))}
                        else:
                            metrics = evaluate(name,data,setting,seed+100000)
                        ratio = {"mse_ratio":metrics["test_mse"]/lasso_mse} if metrics["status"]=="ok" else {}
                        store.append({"experiment":"exp5_real","dataset":dataset,"algorithm":name,
                                      "repeat":repeat,"seed":seed,"n":setting.n,"m":m,"d":setting.d,
                                      "k":setting.k,"delta":setting.delta,"epsilon":epsilon,"setting_kind":"real_standardized",
                                      **metrics,**ratio,"lasso_test_mse":lasso_mse,
                                      "lasso_denominator_convergence_warning":bool(lasso_meta.get("convergence_warning",False)),
                                      "lasso_denominator_convergence_warning_count":int(lasso_meta.get("convergence_warning_count",0))})
    return store.protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real tabular dataset grid; writes CSV only.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--algorithm", choices=REAL_FORMAL_ALGORITHMS, action="append")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.csv, args.target, args.algorithm or list(REAL_FORMAL_ALGORITHMS), args.repeats, args.out)


if __name__ == "__main__":
    main()
