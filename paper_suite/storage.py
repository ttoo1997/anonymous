'Incremental, crash-readable results with an explicit incomplete state.'
from __future__ import annotations

from .checksum import checksum_file
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def json_default(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: dict):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False,default=json_default),encoding='utf-8')
    temp.replace(path)


class ResultStore:
    def __init__(self,out: Path,protocol: dict):
        self.out=out
        self.protocol=dict(protocol)
        self.rows=[]
        self.keys=set()

    def __enter__(self):
        self.out.mkdir(parents=True,exist_ok=False)
        code=Path(__file__).resolve().parent
        sources=list(code.rglob('*.py'))+list((code.parent/'lasso_threshold_v1').rglob('*.py'))+list((code.parent/'dp_safety').rglob('*.py'))
        checksums={}
        for path in sources:
            relative=path.relative_to(code.parent)
            destination=self.out/'source_snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
            checksums[str(relative)]=checksum_file(destination)
        self.protocol.update(source_checksums=checksums,execution_status='running',all_requested_cells_ok=False)
        self.protocol.update(completion_semantics='numerical execution only; not algorithm or privacy certification',
                             end_to_end_privacy_certified=False,
                             release_scope='internal research diagnostics; CSV/JSON may include nonprivate data statistics')
        atomic_json(self.out/'protocol.json',self.protocol)
        return self

    def append(self,row: dict):
        key=tuple(row.get(k) for k in ('experiment','dataset','algorithm','repeat','n','m','d','k','epsilon'))
        if key in self.keys:
            raise ValueError('duplicate result cell')
        if row.get('status') not in ('ok','infeasible','not_implemented'):
            raise ValueError('unknown result status')
        if row['status']=='ok':
            required=(['support_recall'] if str(row['experiment']).startswith('exp4') else
                      ['test_mse','mse_ratio'] if row['experiment']=='exp5_real' else
                      ['test_mse','relative_coefficient_error'])
            if any(k not in row or not np.isfinite(row[k]) for k in required):
                raise ValueError('successful cell is missing finite required metrics')

        text=json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_default)
        with (self.out/'results.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(text+'\n')
            handle.flush()
        self.rows.append(row)
        self.keys.add(key)
        atomic_json(self.out/'progress.json',{'completed_cells':len(self.rows),'last_status':row['status']})

    def __exit__(self,kind,value,traceback):
        frame=pd.DataFrame(self.rows)
        if not frame.empty:
            frame.to_csv(self.out/'results.csv',index=False)
            keys=[k for k in ('experiment','dataset','algorithm','n','m','d','k','epsilon') if k in frame]
            metrics=[k for k in ('relative_coefficient_error','support_recall','stage1_support_recall','stage1_support_precision','stage1_support_f1','final_support_recall','final_support_precision','final_support_f1','topk_support_recall','topk_support_precision','support_precision','support_f1','estimated_nonzero','selected_count','test_mse','mse_ratio') if k in frame]
            ok=frame[frame.status=='ok']
            if metrics and len(ok):
                summary=ok.groupby(keys,dropna=False)[metrics].agg(['mean','std','count'])
                summary.columns=['_'.join(c) for c in summary.columns]
                summary.to_csv(self.out/'summary.csv')
        expected=self.protocol.get('expected_cells',len(self.rows))
        self.protocol.update(execution_status='failed' if kind else 'finished',completed_cells=len(self.rows),
                             all_requested_cells_ok=not kind and len(self.rows)==expected and
                             all(r['status']=='ok' for r in self.rows))
        if kind:
            self.protocol['failure']=f'{kind.__name__}: {value}'
        atomic_json(self.out/'protocol.json',self.protocol)
        return False
