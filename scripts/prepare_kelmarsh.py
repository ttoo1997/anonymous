"""Prepare the documented Kelmarsh split from publisher SCADA archives."""
import argparse
from pathlib import Path
import numpy as np
from data_preparation import kelmarsh_prepare as prep
from data_preparation.kelmarsh_transform import transform
from exp5.common import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True,help='Directory containing raw/ with six SCADA year archives and signal mapping')
    args=p.parse_args(); prep.ROOT=args.work_dir.resolve(); prep.ROOT.mkdir(parents=True,exist_ok=True)
    out=prep.ROOT/'nacelle_temperature_v2'
    if (out/'manifest.json').exists():raise FileExistsError('Prepared data already exist; use another work directory')
    prep.assemble(); prep.prepare('nacelle_temperature')
    meta=load(out/'manifest.json'); raw=np.load(prep.ROOT/'physical_raw_v2.npz')
    names=list(raw['names']); target=raw['X'][:,names.index(meta['target'])].astype(float)
    time=raw['time']; station=raw['station']; ids=raw['uid']; days=time//86400
    chosen=np.flatnonzero(np.isfinite(target)&((days*2654435761+20260925)%10==9))
    train_ids=np.load(out/'train_ids.npy'); train_days=np.load(out/'train_days.npy')
    if np.isin(ids[chosen],train_ids).any() or np.isin(days[chosen],train_days).any():raise ValueError('Train/test overlap')
    X=np.lib.format.open_memmap(out/'test_x.npy',mode='w+',dtype=np.float64,shape=(len(chosen),meta['d']))
    values=raw['X']
    for start in range(0,len(chosen),8192):
        ix=chosen[start:start+8192]
        X[start:start+len(ix)]=transform(values[ix],time[ix],station[ix],names,meta)
    X.flush()
    np.save(out/'test_y.npy',(target[chosen]-meta['y_mean'])/meta['y_scale'])
    np.save(out/'test_ids.npy',ids[chosen]); np.save(out/'test_days.npy',days[chosen])
    print('Prepared data:',out)


if __name__=='__main__':
    main()
