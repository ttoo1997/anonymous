"""Paired Kelmarsh evaluation with fixed profiles and no score-based exclusions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from exp5.algorithms import fit
from exp5.common import dump, load, digest
from exp5.runtime_checks import validate_fit
from paper_suite.frozen import verify
from paper_suite.checksum import checksum_file,checksum_bytes

ROOT = Path(__file__).resolve().parents[1]


def checked_data(path):
    """Check prepared arrays and record their file sizes and CRC32 values."""
    path=Path(path)
    layout=load(ROOT/'configs/data_files.json')
    files=[layout['metadata_file']]+layout['arrays']['train']+layout['arrays']['test']
    checksums={}
    for name in files:
        file=path/name
        if not file.is_file():
            raise ValueError('Missing prepared input file: '+name)
        checksums[name]=checksum_file(file)

    meta=load(path/layout['metadata_file'])
    if meta['data_id']!='kelmarsh-16807551-2017to2022-nacelle_temperature-925001-v2' or meta['d']!=90:
        raise ValueError('Dataset protocol differs')

    arrays={}
    for split in ('train','test'):
        x_name,y_name,ids_name,days_name=layout['arrays'][split]
        x=np.load(path/x_name,mmap_mode='r',allow_pickle=False)
        y=np.load(path/y_name,mmap_mode='r',allow_pickle=False)
        ids=np.load(path/ids_name,mmap_mode='r',allow_pickle=False)
        days=np.load(path/days_name,mmap_mode='r',allow_pickle=False)
        if (x.ndim!=2 or x.shape[1]!=meta['d'] or
            y.shape!=(len(x),) or ids.shape!=(len(x),) or
            days.shape!=(len(x),) or len(x)==0):
            raise ValueError('Prepared '+split+' array dimensions differ')
        if len(np.unique(ids))!=len(ids):
            raise ValueError('Duplicate '+split+' row identities')
        expected_count=meta.get('N_pool' if split=='train' else 'N_test_reserved')
        if expected_count is not None and len(x)!=expected_count:
            raise ValueError('Prepared '+split+' row count differs')
        expected_buckets=np.arange(2,9) if split=='train' else np.array([9])
        if not np.all(np.isin((days*2654435761+20260925)%10,expected_buckets)):
            raise ValueError('Prepared '+split+' dates differ from protocol')
        arrays[split]=(ids,days)
    train_ids,train_days=arrays['train']
    test_ids,test_days=arrays['test']
    if np.intersect1d(train_ids,test_ids).size or np.intersect1d(train_days,test_days).size:
        raise ValueError('Training and evaluation splits overlap')
    return meta,checksums


def metrics(beta, x, y, scale):
    error = x @ beta - y
    mse = float(np.mean(error*error))
    if not np.isfinite(mse):
        raise FloatingPointError('Nonfinite evaluation')
    return dict(mse=mse, mse_original=mse*scale**2, rmse_original=np.sqrt(mse)*scale,
                mae_original=float(np.mean(abs(error)))*scale, r2=1-mse/float(np.var(y)))


def fit_seed(seed, path, out, spec, profiles, meta):
    path, out = Path(path), Path(out)
    with threadpool_limits(limits=1):
        x0=np.load(path/'train_x.npy', mmap_mode='r'); y0=np.load(path/'train_y.npy', mmap_mode='r')
        day0=np.load(path/'train_days.npy', mmap_mode='r'); ids0=np.load(path/'train_ids.npy', mmap_mode='r')
        order=np.random.default_rng(seed+spec['group_seed_offset']).permutation(len(x0))
        order=order[:spec['n']*max(spec['m'])]
        X=np.asarray(x0[order]); y=np.asarray(y0[order]); days=np.asarray(day0[order]); ids=np.asarray(ids0[order])
        test_x=np.load(path/'test_x.npy',mmap_mode='r'); test_y=np.load(path/'test_y.npy',mmap_mode='r')
        cache=out/'cache'/str(seed); cache.mkdir(parents=True,exist_ok=True)
        data_id=meta['data_id']+'-formal-regroup-'+str(seed+spec['group_seed_offset'])
        for m in spec['m']:
            N=spec['n']*m
            xx,yy=X[:N],y[:N]
            draw_checksum=checksum_bytes(ids[:N].tobytes())
            reference_path=out/'references'/f'{seed}_{m}.json'
            if reference_path.exists():
                reference=load(reference_path)
                if reference['draw_checksum']!=draw_checksum:raise ValueError('Reference data differ')
            else:
                beta,details=fit('lasso',xx,yy,m,1.,'native',{},seed,cache,data_id,meta,days=days[:N])
                validate_fit('lasso',beta,details,N,meta['d'])
                reference=dict(seed=seed,m=m,draw_checksum=draw_checksum,beta=beta,details=details,
                               metrics=metrics(beta,test_x,test_y,meta['y_scale']))
                dump(reference_path,reference)
            denominator=reference['metrics']['mse']
            if not np.isfinite(denominator) or denominator<=0:raise ValueError('Invalid Lasso denominator')
            for epsilon in spec['epsilon']:
                for method in spec['methods']:
                    dest=out/'cells'/f'{seed}_{m}_{epsilon:g}_{method}.json'
                    if dest.exists():
                        saved=load(dest)
                        if saved['draw_checksum']!=draw_checksum or saved['status']!='ok':raise ValueError('Invalid saved cell')
                        continue
                    if method=='lasso':
                        beta,details,score=reference['beta'],reference['details'],reference['metrics']
                    else:
                        beta,details=fit(method,xx,yy,m,epsilon,'native',profiles[method],seed,cache,data_id,meta,days=days[:N])
                        validate_fit(method,beta,details,N,meta['d'])
                        score=metrics(beta,test_x,test_y,meta['y_scale'])
                    dump(dest,dict(seed=seed,group_seed=seed+spec['group_seed_offset'],m=m,n=spec['n'],
                                   epsilon=epsilon,method=method,draw_checksum=draw_checksum,status='ok',
                                   **score,mse_ratio=score['mse']/denominator,beta=beta,details=details))
                    print(f'{seed} m={m} epsilon={epsilon:g} {method} saved',flush=True)
    return seed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1); p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--plan-only',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    verify(); spec=load(ROOT/'configs/experiments.json')['real']; profiles=load(ROOT/'FROZEN_PARAMETERS.json')['profiles']
    if args.seeds:spec['seeds']=args.seeds
    if len(set(spec['seeds']))!=len(spec['seeds']):raise ValueError('Duplicate seeds')
    meta,data_checksums=checked_data(args.data_dir)
    spec={**spec,'data_id':meta['data_id'],'data_checksums':data_checksums,'profiles':profiles,'code_lock':digest(load(ROOT/'VERSION_LOCK.json'))}
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/'manifest.json'
    if manifest.exists():
        if not args.resume or load(manifest)!=spec:raise ValueError('Use --resume with the identical protocol')
    else:dump(manifest,spec)
    if args.plan_only:
        print(json.dumps({'seeds':len(spec['seeds']),'cells':len(spec['seeds'])*len(spec['m'])*len(spec['epsilon'])*len(spec['methods']),'fitting_started':False}))
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(fit_seed,s,str(args.data_dir),str(args.out),spec,profiles,meta) for s in spec['seeds']]
        for future in as_completed(tasks):future.result()
    rows=[{k:v for k,v in load(p).items() if k not in ('details','beta')} for p in sorted((args.out/'cells').glob('*.json'))]
    frame=pd.DataFrame(rows); expected=len(spec['seeds'])*len(spec['m'])*len(spec['epsilon'])*len(spec['methods'])
    if len(frame)!=expected or frame.duplicated(['seed','m','epsilon','method']).any():raise ValueError('Incomplete or duplicate result cells')
    frame.to_csv(args.out/'results.csv',index=False)
    summary=frame.groupby(['m','epsilon','method'])[['mse_original','rmse_original','mae_original','mse_ratio','r2']].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]; summary.to_csv(args.out/'summary.csv')
    dump(args.out/'completion.json',{'complete':True,'cells':len(frame),'seed_exclusions':[]})


if __name__=='__main__':
    main()
