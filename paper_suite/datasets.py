"""Optional external dataset registry and generic tabular adapters.

The published reproduction uses scripts.run_real with its pinned publisher data.
This module supports extra, explicitly configured experiments and ships no
personal data paths, default dataset choices, or source records.
"""
from pathlib import Path
from .checksum import checksum_file
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .data import RegressionData,load_tabular_csv


def _load_registry():
    location = os.environ.get('SDP_DATASET_REGISTRY')
    if not location:
        return {}
    data = json.loads(Path(location).expanduser().read_text(encoding='utf-8'))
    if not isinstance(data,dict):
        raise ValueError('SDP_DATASET_REGISTRY must be a JSON object')
    registry={}
    for name,item in data.items():
        if not isinstance(item,dict) or item.get('format') not in ('csv','household_npz'):
            raise ValueError('Unsupported registered dataset format')
        if not item.get('path') or not item.get('target'):
            raise ValueError('Each registered dataset requires path and target')
        registry[name]={**item,'path':Path(item['path']).expanduser().resolve()}
    return registry


DATASETS=_load_registry()


def digest(path):
    return checksum_file(path)


def load_household_npz(path:Path,m:int,seed:int) -> RegressionData:
    with np.load(path,allow_pickle=False) as archive:
        x,y,ids=archive['x'],archive['y'],archive['ids']
    if x.ndim!=3 or y.shape!=x.shape[:2] or len(ids)!=len(x) or len(np.unique(ids))!=len(ids):
        raise ValueError('invalid household tensors or duplicate household IDs')
    if m<1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid finite inputs')
    train,test=train_test_split(np.arange(len(x)),test_size=.2,random_state=seed)
    d=x.shape[-1]
    raw_train=x[train].reshape(-1,d)
    scaler=StandardScaler().fit(raw_train)
    xt=scaler.transform(raw_train).astype(float)
    xv=scaler.transform(x[test].reshape(-1,d)).astype(float)
    yt=y[train].ravel().astype(float); yv=y[test].ravel().astype(float)
    mean,scale=float(yt.mean()),float(yt.std())
    if scale<=0:
        raise ValueError('constant target')
    yt=(yt-mean)/scale; yv=(yv-mean)/scale
    n=len(xt)//m
    if not n:
        raise ValueError('no complete groups')
    return RegressionData(xt[:n*m].reshape(n,m,d),yt[:n*m].reshape(n,m),xv,yv,None,scale,xt,yt,
        {'train_rows':len(xt),'test_rows':len(xv),'used_private_rows':n*m,
         'discarded_group_tail':len(xt)-n*m,'train_household_ids':ids[train].tolist(),
         'test_household_ids':ids[test].tolist(),'source_households':len(ids),
         'grouping':'pseudo-users after disjoint subject split; groups can cross subject boundaries',
         'target_transform':'train mean/std only; no test-dependent clipping'})


def load_registered(name:str,m:int,seed:int) -> RegressionData:
    spec=DATASETS[name]
    if spec['format']=='csv':
        return load_tabular_csv(spec['path'],spec['target'],m,seed)
    return load_household_npz(spec['path'],m,seed)
