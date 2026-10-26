'Validate the frozen Table 2 / Table 5 formula-aligned release.'
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,math,platform,sys,time,warnings
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import scipy,sklearn
from threadpoolctl import threadpool_limits
from .storage import atomic_json,json_default
from .config import SyntheticSetting,synthetic_grid
from .run_synthetic import EXPERIMENTS
from .data import make_synthetic,RegressionData
from .runner import run_algorithm,metric_row,DEFAULT_ALGORITHMS
from .frozen import verify

def certificates(out):
    from lasso_threshold_v1.mechanism.proposed import partition,PROTOCOL
    from lasso_threshold_v1.mechanism.appendix import aggregation_parameters,selection_parameters
    from lasso_threshold_v1.mechanism.core import StageBudget
    from .algorithms.sample_sdp_core import sdp2_table_parameters
    settings={tuple(asdict(s).values()):s for e in EXPERIMENTS for s in synthetic_grid(e)}
    stage=[];sample=[];selection=[]
    for st in settings.values():
        a,b,c=partition(st.n);n3,n4=len(b),len(c)
        sel=selection_parameters(st.n,st.m,st.d,st.k,st.epsilon,st.delta,n1=len(a))
        sc=sel['stage1_privacy_certificate']
        assert sc['certified_total_epsilon_bound']<=st.epsilon+1e-12
        assert sc['certified_total_delta_bound']<=st.delta
        selection.append({**asdict(st),'gamma_used':sel['gamma'],**sc})
        for s in range(1,st.d+1):
            tb=PROTOCOL.sigma*math.sqrt(s*math.log(st.n)/st.m)
            R0=1/math.sqrt(st.k)+tb
            Rf=tb+PROTOCOL.sigma*math.sqrt(s*math.log(st.n)/(n3*st.m))+R0*math.sqrt(s*math.log(st.n)*math.log(4*s/st.delta))/(n3*st.epsilon)
            for label,ng,R in [('coarse',n3,R0),('fine',n4,Rf)]:
                p=aggregation_parameters(ng,s,R,st.n,StageBudget(st.epsilon,st.delta),1.)
                table_requested=math.ceil(32*(p.integer_sensitivity/st.epsilon)**2*math.log(4*s/st.delta))
                table_local=2*math.ceil(table_requested/(2*ng))
                cert=p.privacy_certificate
                assert cert['delta_total_bound']<=st.delta
                assert cert['epsilon_noise_certified']<=st.epsilon
                stage.append({**asdict(st),'stage':label,'support_size':s,'group_size':ng,
                              'local_trials':p.local_trials,'table_local_trials':table_local,
                              'noise_changed':p.local_trials!=table_local,**cert})
        t=sdp2_table_parameters(st.n*st.m,st.d,st.epsilon,st.delta,noise_scale=st.noise_std,kappa_infinity=1.)
        sample.append({**asdict(st),'local_trials':t['local_trials'],
                       'noise_changed':t['noise_recalibrated'],**t['privacy_certificate']})
    pd.DataFrame(stage).to_csv(out/'stage2_all_supports.csv',index=False)
    pd.DataFrame(sample).to_csv(out/'sample_sdp.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'stage1.csv',index=False)
    result={'unique_settings':len(settings),'stage1_certificates':len(selection),
            'stage2_certificates':len(stage),'sample_certificates':len(sample),
            'all_certificates_passed':True,'formal_proposed_noise_changes':sum(r['noise_changed'] for r in stage),
            'formal_sample_noise_changes':sum(r['noise_changed'] for r in sample)}
    atomic_json(out/'summary.json',result);return result

def numerical(out,anchor=False):
    rows=[];private_names=[a for a in DEFAULT_ALGORITHMS if a!='lasso']
    cases=[('synthetic_anchor',SyntheticSetting(),20260915)] if anchor else [
        ('synthetic_quick',SyntheticSetting(n=256,m=50,d=16,k=3,epsilon=e),73291) for e in (4.,32.)]
    if not anchor:cases += [('synthetic_correlated',SyntheticSetting(n=512,m=50,d=10,k=3,epsilon=e),73292) for e in (4.,32.)]
    for label,st,seed in cases:
        if label == 'synthetic_correlated':
            rng=np.random.default_rng(seed)
            base=rng.normal(size=(1200,st.d))
            base[:,1]=.85*base[:,0]+.15*base[:,1]
            idx=rng.integers(len(base),size=st.n*st.m+2000)
            x=base[idx[:st.n*st.m]].reshape(st.n,st.m,st.d);test=base[idx[st.n*st.m:]]
            truth=np.zeros(st.d);truth[rng.choice(st.d,st.k,replace=False)]=1/st.k
            y=x@truth+rng.normal(0,st.noise_std,size=(st.n,st.m))
            yt=test@truth+rng.normal(0,st.noise_std,size=len(test))
            data=RegressionData(x,y,test,yt,truth,metadata={'coefficient_protocol':'equal_positive_l1_v1',
                'test_only_design':'seeded correlated Gaussian design with synthetic response and grouping'})
        else:data=make_synthetic(st,seed,test_samples=2000)
        for name in (private_names if anchor else DEFAULT_ALGORITHMS):
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');beta,meta=run_algorithm(name,data,st,seed+100000)
            assert beta.shape==(st.d,) and np.isfinite(beta).all()
            item={'case':label,'algorithm':name,'data_seed':seed,**asdict(st),**metric_row(beta,data),
                  'seconds':time.perf_counter()-start,'status':'ok','external_warnings':len(caught),
                  'local_ols_fallback_count':meta.get('local_ols_fallback_count',0),
                  'local_lasso_warning_count':meta.get('local_lasso_warning_count',0)}
            rows.append(item);tag=f'{label}_eps{st.epsilon}_{name}'
            np.save(out/(tag+'.npy'),beta);atomic_json(out/(tag+'.metadata.json'),meta)
            pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
            print(json.dumps(item,default=json_default),flush=True)
        del data
    result={'completed_runs':len(rows),'all_finite':True,'full_formal_grid_rerun':False}
    atomic_json(out/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['quick','certificates','anchor'],default='quick');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();version=verify();args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/'environment.json',{'version':version,'python':sys.version,'platform':platform.platform(),
                'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__})
    with threadpool_limits(limits=1):result=certificates(args.out) if args.mode=='certificates' else numerical(args.out,args.mode=='anchor')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
