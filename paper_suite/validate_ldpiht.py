'Optional synthetic parameter-grid diagnostics for LDP-IHT.'
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .algorithms.ldp_iht import (paper_parameter_grid, ma24_empirical_parameter_grid,
                                  qualification_parameter_grid)
from .config import SyntheticSetting
from .data import make_synthetic
from .runner import run_algorithm, metric_row


def _grid(kind: str, n_participants: int, k: int):
    if kind == 'ma24':
        return ma24_empirical_parameter_grid()
    if kind == 'theorem':
        return paper_parameter_grid(n_participants, k)
    if kind == 'qualification':
        return qualification_parameter_grid()
    raise ValueError(kind)


def run(out: Path, seeds: list[int], epsilon: float = 4.0, grid_kind: str = 'ma24',
        first: int = 0, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if out.exists():
        raise FileExistsError(out)
    setting = SyntheticSetting(epsilon=epsilon)
    n_participants = setting.n * setting.m
    full = _grid(grid_kind, n_participants, setting.k)
    grid = full[first:] if limit is None else full[first:first+limit]
    rows=[]
    for seed in seeds:
        data=make_synthetic(setting, seed)
        for idx,cfg in enumerate(grid, start=first):
            call={k:v for k,v in cfg.items() if not k.startswith('theorem_')}
            beta,meta=run_algorithm('ldpiht',data,setting,seed+100000,ldpiht_parameters=call)
            metrics=metric_row(beta,data)
            rows.append({
                'validation_seed':seed,'config_id':idx,**asdict(setting),
                **cfg,**metrics,
                'x_clip_fraction':meta['x_clip_fraction'],
                'y_clip_fraction':meta['y_clip_fraction'],
                'projection_rate':meta['projection_rate'],
                'gradient_radius':meta['gradient_radius'],
                'parameter_profile':f'{grid_kind}_validation_override',
            })
    frame=pd.DataFrame(rows)
    group=['config_id','iterations','step_size','x_clip','y_clip','sparsity']
    summary=frame.groupby(group,dropna=False).agg(
        relative_coefficient_error_mean=('relative_coefficient_error','mean'),
        relative_coefficient_error_std=('relative_coefficient_error','std'),
        topk_support_recall_mean=('topk_support_recall','mean'),
        final_support_recall_mean=('final_support_recall','mean'),
        x_clip_fraction_mean=('x_clip_fraction','mean'),
        y_clip_fraction_mean=('y_clip_fraction','mean'),
        projection_rate_mean=('projection_rate','mean'),
        gradient_radius_mean=('gradient_radius','mean'),
    ).reset_index().sort_values(['relative_coefficient_error_mean','config_id'])
    out.mkdir(parents=True)
    frame.to_csv(out/'rows.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    protocol={
        'core_source':'Zhu et al., ICLR 2024, Algorithm 2',
        'grid_kind':grid_kind,
        'ma24_grid_source':'Ma et al., ICML 2024, Appendix E.1' if grid_kind in ('ma24','qualification') else None,
        'qualification_grid_frozen': bool(grid_kind == 'qualification'),
        'formal_profile_auto_update': False,
        'comparison_metric':'relative L2 coefficient error on synthetic evaluation samples',
        'diagnostic_only':['topk/final support recall','x/y clip fractions','projection rate'],
        'seeds':seeds,'epsilon':epsilon,'setting':asdict(setting),
        'grid_slice':{'first':first,'limit':limit,'full_grid_size':len(full)},
        'automatic_default_update':False,
    }
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    return frame,summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,action='append')
    p.add_argument('--epsilon',type=float,default=4.0)
    p.add_argument('--grid',choices=('ma24','theorem','qualification'),default='qualification')
    p.add_argument('--first',type=int,default=0)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    run(args.out,args.seed or [20261015,20261016],args.epsilon,args.grid,args.first,args.limit)


if __name__=='__main__':
    main()
