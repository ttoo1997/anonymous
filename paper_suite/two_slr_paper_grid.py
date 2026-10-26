'Run Ma24 Appendix-E.1 2-SLR parameter reporting on a synthetic setting.'
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .algorithms.two_slr_paper import (
    PAPER_B_GRID,
    PAPER_BIN_GRID,
    PAPER_LOCAL_SELECTED_GRID,
    TwoSLRProfile,
    _local_ols_coefficients,
    _uldp_mean,
    select_support,
)
from .config import SyntheticSetting
from .data import make_synthetic
from .metrics import mse, relative_coefficient_error


def run(setting: SyntheticSetting, repeats: int, out: Path, seed0: int = 20261101) -> dict:
    setting.validate()
    if repeats < 30:
        raise ValueError("Ma24 Appendix E.1 reporting uses at least 30 replications")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows=[]
    for repeat in range(repeats):
        data_seed=seed0+repeat
        alg_seed=data_seed+100000
        data=make_synthetic(setting,data_seed)
        for selected in PAPER_LOCAL_SELECTED_GRID:
            if selected>setting.d:
                continue
            base=TwoSLRProfile(local_selected=selected,hh_target=setting.k,
                screening_features=min(64,setting.d),B=1.0,num_bins=2)
            support,sel_meta=select_support(data.user_x,data.user_y,setting.epsilon,
                setting.k,alg_seed,base)
            local_beta=_local_ols_coefficients(data.user_x[setting.n//2:],
                data.user_y[setting.n//2:],support)
            for B in PAPER_B_GRID:
                for bins in PAPER_BIN_GRID:


                    rng=np.random.default_rng(alg_seed+35023)
                    low_beta,mean_meta=_uldp_mean(local_beta,setting.epsilon,B,bins,rng)
                    beta=np.zeros(setting.d,float);beta[support]=low_beta
                    rows.append({
                        "repeat":repeat,"data_seed":data_seed,"algorithm_seed":alg_seed,
                        **asdict(setting),"local_selected":selected,"hh_target":setting.k,
                        "screening_features":min(64,setting.d),"B":B,"num_bins":bins,"hh_alpha":float(base.hh_alpha),
                        "selected_count":len(support),
                        "relative_coefficient_error":relative_coefficient_error(beta,data.beta_true),
                        "test_mse":mse(beta,data.x_test,data.y_test),
                        "stage1_support_recall":len(np.intersect1d(support,np.flatnonzero(data.beta_true)))/setting.k,
                        "hh_one_report_per_user":sel_meta["hh_one_report_per_user"],
                        "stage2_groups_disjoint":mean_meta["stage2_groups_disjoint"],
                    })
    frame=pd.DataFrame(rows)
    keys=["local_selected","hh_target","screening_features","B","num_bins","hh_alpha"]
    summary=frame.groupby(keys,as_index=False).agg(
        relative_coefficient_error_mean=("relative_coefficient_error","mean"),
        relative_coefficient_error_std=("relative_coefficient_error","std"),
        test_mse_mean=("test_mse","mean"),
        stage1_support_recall_mean=("stage1_support_recall","mean"),
        count=("repeat","count"),
    )
    best=summary.sort_values(["relative_coefficient_error_mean","test_mse_mean"],kind="mergesort").iloc[0].to_dict()
    frame.to_csv(out/"rows.csv",index=False)
    summary.to_csv(out/"summary.csv",index=False)
    protocol={
        "source":"Ma, Jia & Yang, ICML 2024, Appendix E.1",
        "reporting_rule":"best average over parameter grid; at least 30 replications; no held-out validation/CV set",
        "setting":asdict(setting),"repeats":repeats,"seed0":seed0,
        "grid":{"local_selected":list(PAPER_LOCAL_SELECTED_GRID),"B":list(PAPER_B_GRID),"num_bins":list(PAPER_BIN_GRID),"hh_alpha":[0.05]},
        "heavy_hitter_caveat":"Appendix E.1 does not publish numerical rho; public-author target-list heuristic is used with hh_target=k and frozen HeavyHitter alpha=0.05",
        "best_by_mean_relative_coefficient_error":best,
    }
    (out/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    return protocol


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--seed0",type=int,default=20261101)
    p.add_argument("--n",type=int,default=800);p.add_argument("--m",type=int,default=50)
    p.add_argument("--d",type=int,default=256);p.add_argument("--k",type=int,default=8)
    p.add_argument("--epsilon",type=float,default=4.0);p.add_argument("--noise-std",type=float,default=.25)
    a=p.parse_args()
    setting=SyntheticSetting(n=a.n,m=a.m,d=a.d,k=a.k,epsilon=a.epsilon,noise_std=a.noise_std)
    print(json.dumps(run(setting,a.repeats,a.out,a.seed0),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
