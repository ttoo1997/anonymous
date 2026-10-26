'Domain-defined contemporaneous sensor regression, not forecasting.'
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import json, csv, zipfile, re, argparse, warnings, sys
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
ROOT=Path(__file__).resolve().parent/'kelmarsh'
from exp5.lasso_gram import fit as gram_lasso
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def norm(name):return name.strip().lower()

def mapped_columns(cols,allowed):
    result={}
    for col in cols:
        if ',' in col:continue
        label=norm(col)
        matches=[name for name in allowed if label==name or label.startswith(name+' (')]
        if not matches:continue
        name=max(matches,key=len)
        assert name not in result,('Duplicate physical signal',name,col)
        result[name]=col
    return result

def assemble():
    dest=ROOT/'physical_raw_v2.npz'
    if dest.exists():return
    mapping=pd.read_csv(ROOT/'raw/Kelmarsh_WT_dataSignalMapping.csv')
    allowed={norm(s) for s in mapping['Greenbyte Title']}
    plan=dict(years=list(range(2017,2023)),tasks=['nacelle_temperature','power'],
              task_target={'nacelle_temperature':'nacelle temperature','power':'power'},
              features='All publisher-mapped mean physical signals, no per-window replicated min/max/std or calculated accounting quantities.',
              power_exclusions='Active/apparent/reactive power, power factor, energy, current, voltage; avoid near-algebraic reconstruction from electrical output.',
              split='Whole UTC dates; hash modulo 10: 0,1 public; 2..8 training; 9 final holdout.',
              seed=925001)
    save(ROOT/'processing_plan.json',plan)
    arrays=[];times=[];stations=[];schemas=[];keys=[];names=None;entries=[]
    for year in plan['years']:
        paths=list((ROOT/'raw').glob(f'Kelmarsh_SCADA_{year}_*.zip'))
        assert len(paths)==1,f'Missing year {year}'
        with zipfile.ZipFile(paths[0]) as z:
            members=sorted(x for x in z.namelist() if x.startswith('Turbine_Data_') and x.endswith('.csv'))
            assert len(members)==6,(year,members)
            entries.extend((paths[0],member,year) for member in members)
    def read_entry(entry):
        archive,member,year=entry
        turbine=int(re.search(r'Kelmarsh_(\d)_',member).group(1))
        with zipfile.ZipFile(archive) as z:
            with z.open(member) as f:
                    while True:
                        line=f.readline()
                        if not line:raise ValueError('Missing header')
                        if line.startswith(b'# Date and time'):break
                    cols=next(csv.reader([line.decode('latin1').strip().lstrip('# ')]))
                    byname=mapped_columns(cols[1:],allowed)
                    selected=list(byname.values())
                    frame=pd.read_csv(f,header=None,names=cols,usecols=[cols[0]]+selected,encoding='latin1',low_memory=False)
        ordered=sorted(byname)
        values=frame[[byname[n] for n in ordered]].to_numpy(dtype=np.float32)
        dt=pd.to_datetime(frame[cols[0]],utc=True).astype('int64').to_numpy()//10**9
        assert (np.diff(dt)>0).all(),member
        schema=dict(file=member,raw_columns=len(cols),kept_mean_signals=len(selected),rows=len(frame),excluded=[c for c in cols[1:] if c not in selected])
        return values,dt,turbine,ordered,schema
    with ThreadPoolExecutor(max_workers=6) as pool:
        for values,dt,turbine,ordered,schema in pool.map(read_entry,entries):
            if names is None:names=ordered
            assert names==ordered,'Schema change across years'
            arrays.append(values);times.append(dt);stations.append(np.full(len(dt),turbine,dtype=np.int8));keys.append(dt*10+turbine)
            schemas.append(schema)
            print('read',schema['file'],len(dt),len(ordered),flush=True)
    X=np.concatenate(arrays);time=np.concatenate(times);station=np.concatenate(stations)
    uid=np.concatenate(keys);assert len(np.unique(uid))==len(uid)
    np.savez(dest,X=X,time=time,station=station,names=np.asarray(names),uid=uid)
    save(ROOT/'schema_audit.json',dict(signals=names,files=schemas,unique_rows=len(uid)))

def prepare(task):
    out=ROOT/(task+'_v2');out.mkdir(exist_ok=True)
    a=np.load(ROOT/'physical_raw_v2.npz');names=list(a['names']);raw=a['X'];time=a['time'];station=a['station'];uid=a['uid']
    target={'nacelle_temperature':'nacelle temperature','power':'power'}[task]
    yi=names.index(target);y=raw[:,yi].astype(float)
    selected=[];excluded={}
    for i,name in enumerate(names):
        reason=None
        if i==yi:reason='target'
        elif task=='power' and any(t in name for t in ('power','energy','current','voltage')):reason='electrical outcome or near-algebraic proxy'
        if reason:excluded[name]=reason
        else:selected.append(i)
    valid=np.isfinite(y)
    X=raw[valid][:,selected];y=y[valid];time=time[valid];station=station[valid];uid=uid[valid]
    names=[names[i] for i in selected];days=time//86400
    hour=(time%86400)/3600
    doy=pd.to_datetime(time,unit='s',utc=True).dayofyear.to_numpy()
    X=np.column_stack([X,np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24),np.sin(doy*2*np.pi/365.25),np.cos(doy*2*np.pi/365.25)])
    names+=['hour_sin','hour_cos','annual_sin','annual_cos']
    bucket=(days*2654435761+20260925)%10
    public=np.flatnonzero(bucket<2);pool=np.flatnonzero((bucket>=2)&(bucket<9));holdout=np.flatnonzero(bucket==9)
    fit_i,val_i=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=925001).split(public,groups=days[public]))
    rng=np.random.default_rng(925001)
    fit=rng.choice(public[fit_i],min(80000,len(fit_i)),replace=False)
    val=rng.choice(public[val_i],min(40000,len(val_i)),replace=False)
    assert not set(days[fit])&set(days[val]) and not set(days[pool])&set(days[public])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(X[fit],axis=0)
    all_missing=~np.isfinite(med)
    missing=~np.isfinite(X);X=np.where(missing,np.where(all_missing,0,med),X)
    sd=X[fit].std(0);keep=(sd>1e-8)&~all_missing
    dropped=[dict(feature=n,reason='all missing or constant on public fit') for n,k in zip(names,keep) if not k]
    X=X[:,keep];miss=missing[:,keep];names=[n for n,k in zip(names,keep) if k]
    mu=X[fit].mean(0);sd=X[fit].std(0);X=(X-mu)/sd;physical_dim=X.shape[1]

    indicators=np.column_stack([miss.astype(float),*(station==j for j in range(1,6))])
    indicator_names=['missing_'+n for n in names]+[f'turbine_{j}' for j in range(1,6)]
    im=indicators[fit].mean(0);ik=indicators[fit].std(0)>1e-8
    X=np.column_stack([X,indicators[:,ik]-im[ik]]).astype(np.float32)
    names += [n for n,k in zip(indicator_names,ik) if k]
    ym=float(y[fit].mean());scale=float(y[fit].std()/.25);Y=(y-ym)/scale


    reference,reference_diag=gram_lasso(X[fit],Y[fit],days[fit],alphas=60,tol=1e-5)
    reference_diag['cv']='Whole-day GroupKFold, 60 alphas, public preparation only'
    inflate=max(1.,1.25*float(np.linalg.norm(reference,1)));scale*=inflate;Y/=inflate;coef=reference/inflate
    V=np.asarray(X[val],float);W=Y[val];loss=float(np.mean((V@coef-W)**2))
    order=np.argsort(-np.abs(coef));curve=[]
    for k in (1,2,4,8,16):
        cols=order[:k];b=np.linalg.lstsq(np.asarray(X[fit][:,cols],float),Y[fit],rcond=None)[0]
        mse=float(np.mean((V[:,cols]@b-W)**2));curve.append(dict(k=k,ratio=mse/loss,r2=1-mse/float(np.var(W))))
    meta=dict(task=task,target=target,N_pool=len(pool),N_public=len(public),N_test_reserved=len(holdout),d=X.shape[1],
              physical_dim=physical_dim,numeric_dim=X.shape[1],categorical_low=[],categorical_high=[],features=names,
              excluded_physical=excluded,dropped_public_constants=dropped,imputation_medians=np.where(all_missing,0,med).tolist(),
              feature_means=mu.tolist(),feature_std=sd.tolist(),indicator_means=im[ik].tolist(),
              y_mean=ym,y_scale=scale,reference_l1=float(np.linalg.norm(coef,1)),reference_l2=float(np.linalg.norm(coef)),
              public_fit_rows=len(fit),public_validation_rows=len(val),reference_diagnostics=reference_diag,lasso_validation_r2=1-loss/float(np.var(W)),
              lasso_validation_mse=loss,sparsity_curve=curve,
              data_id='kelmarsh-16807551-2017to2022-'+task+'-925001-v2',
              privacy_scope='simulated random user slots; native privacy units; public-fitted preprocessing',
              top_features=[dict(name=names[i],coefficient=float(coef[i])) for i in order[:12]])
    ids=rng.permutation(pool)
    np.save(out/'train_x.npy',X[ids]);np.save(out/'train_y.npy',Y[ids]);np.save(out/'train_days.npy',days[ids]);np.save(out/'train_ids.npy',uid[ids])
    np.savez(out/'development.npz',X=X[fit],Y=Y[fit],V=X[val],W=W,fit_days=days[fit],val_days=days[val],reference=coef)
    save(out/'manifest.json',meta)
    print(json.dumps({k:meta[k] for k in ('task','N_pool','d','lasso_validation_r2','sparsity_curve','top_features')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tasks',nargs='+',default=['nacelle_temperature','power']);args=parser.parse_args()
    assemble()
    for task in args.tasks:prepare(task)
