'Original mechanism primitives with transparent parameter and cache adapters.'
from pathlib import Path
import math,warnings
import numpy as np
from .common import ROOT,load,dump,digest
from .privacy import budgets
from dp_safety.parameters import APPROXIMATE_DP_DELTA
from .names import canonical
from . import sample_fast
from . import sample_ridge
from . import lasso_gram
from . import two_local
from paper_suite.algorithms import two_slr_paper as two
from paper_suite.algorithms.ldp_iht import fit_ldp_iht
from paper_suite.algorithms.sample_sdp_core import Config,sdp2_precision,fit_sdp2_from_sum
from lasso_threshold_v1.mechanism import proposed as u2r_core
from lasso_threshold_v1.mechanism.local import candidates
from sklearn.linear_model import LassoCV,Lasso
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning

def profiles(method):
    method=canonical(method)
    if method=='u2r':return [dict(k=k,rho=rho,norm=1.) for k in (1,2,4,8) for rho in (.025,.05,.125,.25)]
    if method=='2slr':return [dict(local_selected=s,k=4,B=B,num_bins=bins) for s in (2,4,8,16) for B in (1.,2.,3.) for bins in (2,4,8,16,32)]
    if method=='sample_sdp':return [dict(clip_constant=c,threshold_constant=t) for c in (.003,.01,.03,.1) for t in (1.,2.,4.)]
    if method=='ldpiht':return [dict(k=4,iterations=T,step_size=eta,x_clip=tau,y_clip=tau,sparsity=32) for T in (5,10,20) for eta in (.01,.1,1.) for tau in (2.,4.,8.)]
    if method=='lasso':return [{}]
    raise ValueError(method)

def default_profile(method):
    method=canonical(method)

    return load(ROOT/'FROZEN_PARAMETERS.json')['profiles'][method]

def selection_cache(X,y,m,method,p,seed,cache_dir,data_id):
    method=canonical(method)
    N=len(X)//m*m;n=N//m;d=X.shape[1];n1=n//2
    base=dict(data=data_id,N=N,m=m,method=method,seed=seed,version='selection-v3-certified-fallback')
    if method=='u2r':base.update(rho=p['rho'])
    else:base.update(local_selected=p['local_selected'],screening_features=min(64,d))
    path=Path(cache_dir)/(digest(base)+'.json')
    if path.exists():return load(path)
    rng=np.random.default_rng(seed);selections=[];totals={};weights=0
    profile=two.TwoSLRProfile(p.get('local_selected',2),p['k'],min(64,d),p.get('B',1.),p.get('num_bins',4))
    for i in range(0,n1,256):
        count=min(256,n1-i);xx=np.asarray(X[i*m:(i+count)*m],float).reshape(count,m,d);yy=np.asarray(y[i*m:(i+count)*m]).reshape(count,m)
        if method=='u2r':s,meta=candidates(xx,yy,p['k'],total_users=n,rho=p['rho'])
        else:s,meta=two_local.local_votes(xx,yy,profile,rng)
        selections.extend([v.tolist() if isinstance(v,np.ndarray) else int(v) for v in s]);weights+=count
        for key,value in meta.items():
            if isinstance(value,(int,float)):
                if key.endswith('_rate') or key.endswith('_mean'):totals[key]=totals.get(key,0.)+value*count
                elif key.endswith('_max') or key.endswith('_max_kkt') or key.endswith('_max_kkt_ratio'):totals[key]=max(totals.get(key,value),value)
                elif key.endswith('_min'):totals[key]=min(totals.get(key,value),value)
                elif 'count' in key:totals[key]=totals.get(key,0)+value
    for key in totals:
        if key.endswith('_rate') or key.endswith('_mean'):totals[key]/=weights
    result=dict(selections=selections,metadata=totals,key=base);dump(path,result)
    return result

def cap_u2r_support(support,debiased,m,k,policy):
    'Public cardinality rule on already-DP counts; never inspect raw designs.'
    if policy!='min_k_m_minus_2' or m<3:
        raise ValueError('Invalid U2R public support cap policy or m<3')
    support=np.asarray(support,dtype=int);cap=min(int(k),int(m)-2)
    if cap<1:raise ValueError('U2R support cap must be positive')
    before=len(support)
    if before>cap:
        order=np.lexsort((support,-np.asarray(debiased)[support]))
        support=np.sort(support[order[:cap]])
    return support,dict(support_cap_policy=policy,public_support_cap=cap,
        threshold_selected_count=before,selected_count=len(support),support_cap_applied=before>cap,
        support_cap_dp_postprocessing=True,local_sample_count=int(m),
        estimator_variant='U2R-DP-count-cap',original_utility_theorem_claimed=False)


def fit_u2r(X,y,m,epsilon,p,seed,cache,data_id):
    n=len(X)//m;d=X.shape[1];N=n*m
    selection=selection_cache(X,y,m,'u2r',p,seed,cache,data_id)
    first,coarse,fine=u2r_core.partition(n)
    table=u2r_core.selection_parameters(n,m,d,p['k'],epsilon,u2r_core.PROTOCOL.delta,n1=len(first),rho=p['rho'])
    streams=u2r_core.RandomStreams(seed)
    counts,_=u2r_core._pick_and_randomize_counts(selection['selections'],d,table['gamma'],streams,materialize_parallel_shuffle=False,include_diagnostics=False)
    support,debiased=u2r_core.select_support_by_threshold(counts,len(first),table['gamma'],len(first)/(4*p['k']))
    cap_meta={}
    if p.get('support_cap_policy') is not None:
        support,cap_meta=cap_u2r_support(support,debiased,m,p['k'],p['support_cap_policy'])
    meta={**table,**selection['metadata'],**cap_meta,'support':support,'release_mode':'research_simulation','private_release_safe':False}

    if not len(support):return np.zeros(d),{**meta,'algorithm_status':'empty_global_support'}
    low=np.empty((n,m,len(support)),float)
    for i in range(0,N,8192):low.reshape(N,-1)[i:i+8192]=np.asarray(X[i:min(i+8192,N)])[:,support]
    beta,details=u2r_core._fit_after_selection(low,np.asarray(y[:N]).reshape(n,m),epsilon,np.arange(len(support)),
        (first,coarse,fine),streams,parameter_l2_norm=p['norm'],private_release=False,meta=meta)
    out=np.zeros(d);out[support]=beta;details['support']=support
    return out,details

def fit_two(X,y,m,epsilon,p,seed,cache,data_id):
    n=len(X)//m;d=X.shape[1];n1=n//2;N=n*m
    selection=selection_cache(X,y,m,'2slr',p,seed,cache,data_id)
    profile=two.TwoSLRProfile(p['local_selected'],p['k'],min(64,d),p['B'],p['num_bins']);profile.validate(d)
    support,hh=two._select_support_from_votes(selection['selections'],d,epsilon,profile,seed+17011)
    meta={**selection['metadata'],**hh,'support':support,'release_mode':'research_simulation','private_release_safe':False,'profile':p,
          'local_solver_policy':'Ma24 selector unchanged; normal path preserved; certified same-objective fallback; unresolved research failure aborts'}
    out=np.zeros(d)
    if not len(support):return out,{**meta,'algorithm_status':'empty_global_support'}
    if two._next_power_of_two(len(support))>(n-n1)-(n-n1)//2:return out,{**meta,'algorithm_status':'public_group_size_insufficient'}
    pieces=[]
    for i in range(n1,n,256):
        count=min(256,n-i);xx=np.asarray(X[i*m:(i+count)*m],float)[:,support].reshape(count,m,-1)
        pieces.append(two._local_ols_coefficients(xx,np.asarray(y[i*m:(i+count)*m]).reshape(count,m),np.arange(len(support))))
    beta,extra=two._uldp_mean(np.concatenate(pieces),epsilon,p['B'],p['num_bins'],np.random.default_rng(seed+35023))
    out[support]=beta
    return out,{**meta,**extra,'algorithm_status':'ok'}

def fit_sample(X,y,epsilon,delta,p,seed,cache,data_id,data_meta):
    solver=p.get('solver','appendix_direct')
    if solver not in ('appendix_direct','ridge'):
        raise ValueError('unknown SampleSDP server solver')
    if solver=='ridge':
        multiplier=p.get('ridge_noise_multiplier')
        if not isinstance(multiplier,(int,float)) or not math.isfinite(multiplier) or multiplier<=0:
            raise ValueError('Ridge requires a positive public ridge_noise_multiplier')
    N,d=X.shape;clip=p['clip_constant']*d*math.log(N)

    noise_scale=p.get('threshold_noise_scale',0.25)
    kappa_infinity=p.get('threshold_kappa_infinity',1.0)
    config=Config(delta=delta,sdp2_clip=clip,sdp2_dimension=d,sdp2_clip_constant=p['clip_constant'],sdp2_threshold_constant=p['threshold_constant'],
                  sdp2_noise_scale=noise_scale,sdp2_kappa_infinity=kappa_infinity)
    precision=sdp2_precision(N,epsilon,config)
    key=dict(data=data_id,N=N,clip=clip,precision=precision,seed=seed,version='exact-categorical-v2')
    path=Path(cache)/(digest(key)+'.npz');jpath=path.with_suffix('.json')
    if path.exists() and jpath.exists():prepared=load(jpath);prepared['rounded']=np.load(path)['rounded']
    else:
        prepared=sample_fast.prepare(X,y,clip,precision,np.random.default_rng(seed+1),data_meta['numeric_dim'],data_meta['categorical_low'],data_meta['categorical_high'])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,rounded=prepared['rounded']);dump(jpath,{k:v for k,v in prepared.items() if k!='rounded'})
    if solver=='ridge':
        beta,meta=sample_ridge.fit_from_sum(prepared['rounded'],prepared['meta'],epsilon,config,np.random.default_rng(seed+2),multiplier)
    else:
        beta,meta=fit_sdp2_from_sum(np.zeros(prepared['meta']['D']),prepared['meta'],epsilon,config,np.random.default_rng(seed+2),prepared['rounded'])
        meta['estimator_variant']='SampleSDP-original'
    meta.update(clip_constant=p['clip_constant'],threshold_constant=p['threshold_constant'],rounding_method=prepared['meta']['preparation'],
                threshold_scale_source=p.get('threshold_scale_source','public_empirical_reference'),
                threshold_noise_scale=noise_scale,threshold_kappa_infinity=kappa_infinity,
                threshold_curvature_bound_verified=False)
    return beta,meta

def fit(method,X,y,m,epsilon,track,p,seed,cache,data_id,data_meta,days=None):
    method=canonical(method)
    budget=budgets(method,epsilon,APPROXIMATE_DP_DELTA,m,track);N=len(X)//m*m;X=X[:N];y=y[:N]
    if method=='u2r':beta,meta=fit_u2r(X,y,m,epsilon,p,seed,cache,data_id)
    elif method=='2slr':beta,meta=fit_two(X,y,m,epsilon,p,seed,cache,data_id)
    elif method=='sample_sdp':beta,meta=fit_sample(X,y,budget['record_epsilon'],budget['record_delta'],p,seed,cache,data_id,data_meta)
    elif method=='ldpiht':
        beta,meta=fit_ldp_iht(X,y,budget['record_epsilon'],p['k'],seed,**{k:v for k,v in p.items() if k not in ('k','vectorized')},vectorized=True)
        meta['estimator_variant']='LDPIHT-gradient-clipped' if p.get('gradient_clip') is not None else 'LDPIHT-original'
        meta['parameter_transfer']='Fixed synthetic stabilization policy' if p.get('gradient_clip') is not None else 'fixed real-data profile'
    elif method=='lasso':
        if days is None:raise ValueError('Lasso grouped CV requires public day grouping')
        beta,meta=lasso_gram.fit(X,y,days[:N])
    else:raise ValueError(method)
    if not np.isfinite(beta).all():raise FloatingPointError('nonfinite coefficients')
    return beta,{**meta,'budget':budget,'parameters':p,'utility_theorem_assumptions_verified':False,'research_diagnostics_are_not_dp_release':True}
