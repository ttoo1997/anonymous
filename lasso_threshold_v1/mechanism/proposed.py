'Appendix-B executable implementation of the proposed user-level shuffled protocol.'
from dataclasses import dataclass,asdict
import math
import secrets
import numpy as np
from dp_safety.parameters import APPROXIMATE_DP_DELTA

from .core import (
    StageBudget,RandomStreams,
    select_support_by_threshold,
    local_ols_coefficients_appendix_b,local_ols_coefficients_private_totalized,
    distributed_private_mean_ikos,
)
from .local import candidates,candidates_private_totalized
from .appendix import selection_parameters, aggregation_parameters, C_LAMBDA, C_GAMMA


@dataclass(frozen=True)
class PdfProtocol:
    version:str='appendix_b_alg3_alg4_table5_aligned_20261021'
    selection_fraction:float=.5
    c_lambda:float=C_LAMBDA
    c_gamma:float=C_GAMMA
    c_beta:float=1.
    c1:float=1.
    c2:float=1.
    precision_constant:float=1.
    sigma:float=.25
    delta:float=APPROXIMATE_DP_DELTA
    mechanism_privacy_certified:bool=True
    simulation_end_to_end_private:bool=False

PROTOCOL=PdfProtocol()


PRIVATE_FORBIDDEN_METADATA_KEYS=frozenset({
    'picked_indices','rr_counts','debiased_counts','partition',
    'local_empty_rate','local_solver_fallback_count','local_solver_fallback_max_kkt',
    'clip_fraction','local_ols_norm_median','release_perturbation_l2',
    'seed','algorithm_seed','transport_seed','rng_state','local_shares','transcript',
    'offset','rounded','noise','signed_integer','local_ols_fallback_count','local_lasso_totalization_count',
})


def partition(n:int,seed:int|None=None):
    if not 0 < PROTOCOL.selection_fraction < 1:
        raise ValueError('selection_fraction must lie in (0,1)')
    n1=int(math.floor(n*PROTOCOL.selection_fraction))
    n3=(n-n1)//2
    if min(n1,n3,n-n1-n3)<19:
        raise ValueError('Appendix-B protocol needs at least 19 users in each group')


    ids=np.arange(n,dtype=int)
    return ids[:n1],ids[n1:n1+n3],ids[n1+n3:]


def _pick_and_randomize_counts(supports,d,gamma,streams,*,materialize_parallel_shuffle:bool,
                               include_diagnostics:bool):
    pick_rng,rr_rng=streams.rng('selection/pick'),streams.rng('selection/rr')
    coords=np.arange(d)
    picked=[] if include_diagnostics else None
    if materialize_parallel_shuffle:


        lanes=np.empty((len(supports),d),dtype=np.uint8)
        for i,support in enumerate(supports):
            support=np.asarray(support,dtype=int)
            v=int(pick_rng.choice(support)) if len(support) else int(pick_rng.integers(d))
            lanes[i]=((coords==v) ^ (rr_rng.random(d)<gamma)).astype(np.uint8)
            if picked is not None:
                picked.append(v)
        counts=np.empty(d,dtype=np.int64)
        for j in range(d):


            perm=streams.rng(f'selection/shuffle/{j}',transport=True).permutation(len(supports))
            shuffled_lane=lanes[perm,j]
            counts[j]=int(np.sum(shuffled_lane,dtype=np.int64))
        return counts,picked


    counts=np.zeros(d,dtype=np.int64)
    for support in supports:
        support=np.asarray(support,dtype=int)
        v=int(pick_rng.choice(support)) if len(support) else int(pick_rng.integers(d))
        bits=(coords==v) ^ (rr_rng.random(d)<gamma)
        counts += bits.astype(np.int64)
        if picked is not None:
            picked.append(v)
    return counts,picked


def _selection_public_meta(table,selected,first,coarse,fine,threshold,*,release_mode,transport_mode):

    return {
        'protocol_version':PROTOCOL.version,
        'release_mode':release_mode,
        'mechanism_privacy_certified':True,
        'private_release_safe':release_mode=='private',
        'support':np.asarray(selected,dtype=int).tolist(),
        'selected_count':int(len(selected)),
        'selection_users':int(len(first)),
        'coarse_users':int(len(coarse)),
        'fine_users':int(len(fine)),
        'gamma':float(table['gamma']),
        'gamma_table_lower':float(table['gamma_table_lower']),
        'gamma_certified_floor':float(table['gamma_certified_floor']),
        'gamma_choice':table['gamma_choice'],
        'stage1_privacy_certificate':table['stage1_privacy_certificate'],
        'lasso_alpha':float(table['lasso_alpha']),
        'coefficient_threshold':float(table['coefficient_threshold']),
        'rho':float(table['rho']),
        'hh_threshold':float(threshold),
        'privacy_unit':'user-level shuffled mechanism; three user groups are disjoint',
        'privacy_validation':'Stage 1 finite-sample certified; Stage 2 cpSGD binomial and Balle IKOS/TV certificates; disjoint-group parallel composition',
        'selection_privacy_basis':'Appendix-B Algorithm 3 RR with finite-sample certificate: Cheu19 Cor.4.7 two-stream composition or exact pure-local fallback',
        'selection_transport':transport_mode,
        'implementation':'Appendix-B Algorithms 3/4 and Tables 4/5; no D/E/F implementation substitutions',
        'appendix_b_formula_structure':True,
    }


def _select_support_impl(x,y,epsilon,k,local_seed,transport_seed,*,rho,private_release:bool,
                         materialize_parallel_shuffle:bool):
    n,m,d=x.shape
    if rho is None:
        raise ValueError('Appendix-B Algorithm 3 requires an explicit public signal-strength lower bound rho')
    rho=float(rho)
    first,coarse,fine=partition(n)
    table=selection_parameters(n,m,d,k,epsilon,PROTOCOL.delta,n1=len(first),rho=rho,
                               c_gamma=PROTOCOL.c_gamma,c_lambda=PROTOCOL.c_lambda,sigma=PROTOCOL.sigma)
    if private_release:
        supports=candidates_private_totalized(x[first],y[first],k,total_users=n,rho=rho,
                                              c_lambda=PROTOCOL.c_lambda,sigma=PROTOCOL.sigma)
        local_meta=None
    else:
        supports,local_meta=candidates(x[first],y[first],k,total_users=n,rho=rho,
                                       c_lambda=PROTOCOL.c_lambda,sigma=PROTOCOL.sigma)
    gamma=table['gamma']
    streams=RandomStreams(int(local_seed),None if transport_seed is None else int(transport_seed))
    counts,picked=_pick_and_randomize_counts(
        supports,d,gamma,streams,
        materialize_parallel_shuffle=materialize_parallel_shuffle,
        include_diagnostics=not private_release,
    )
    threshold=len(first)/(4*k)
    selected,debiased=select_support_by_threshold(counts,len(first),gamma,threshold)
    if private_release:
        meta=_selection_public_meta(
            table,selected,first,coarse,fine,threshold,
            release_mode='private',transport_mode='materialized_parallel_coordinate_shufflers',
        )
        meta.update({
            'fresh_secret_randomness':True,
            'seed_exported':False,
            'transcript_exported':False,
            'numerical_totalization':'local Lasso failures are mapped internally to Algorithm-3 empty-candidate fallback; no fallback flag is released',
        })
        return selected,meta,(first,coarse,fine),streams

    meta={
        'protocol':asdict(PROTOCOL),**table,**local_meta,'support':selected,
        'selection_users':len(first),'gamma':gamma,'rr_counts':counts,'debiased_counts':debiased,
        'picked_indices':picked,
        'rr_count_noise_sd':math.sqrt(len(first)*gamma*(1-gamma))/(1-2*gamma),
        'hh_threshold':threshold,
        'partition':{'selection':first.tolist(),'coarse':coarse.tolist(),'fine':fine.tolist()},
        'privacy_unit':'user-level shuffled mechanism; three user groups are disjoint',
        'privacy_validation':'Stage 1 finite-sample certified; Stage 2 cpSGD binomial and Balle IKOS/TV certificates; disjoint-group parallel composition',
        'selection_privacy_basis':'Appendix-B Algorithm 3 RR with finite-sample certificate: Cheu19 Cor.4.7 two-stream composition or exact pure-local fallback',
        'selection_transport':'count-equivalent research simulation; coordinate shuffles are not materialized because only counts are analyzed',
        'implementation':'Appendix-B Algorithms 3/4 and Tables 4/5; no D/E/F implementation substitutions',
        'appendix_b_formula_structure':True,
        'transcript_level_privacy_simulation':False,
        'release_mode':'research_simulation',
        'private_release_safe':False,
    }
    return selected,meta,(first,coarse,fine),streams


def select_support(x,y,epsilon,k,seed,*,rho=None):
    'Deterministic research-simulation Stage 1; metadata is NOT a DP release.'
    selected,meta,_,_=_select_support_impl(
        x,y,epsilon,k,int(seed),None,rho=rho,private_release=False,
        materialize_parallel_shuffle=False,
    )
    return selected,meta


def _fit_after_selection(x,y,epsilon,selected,group_ids,streams,*,parameter_l2_norm,
                         private_release:bool,meta:dict):
    n,m,d=x.shape
    beta=np.zeros(d)
    if not len(selected):
        return beta,{**meta,'algorithm_status':'empty_global_support','parameter_l2_norm':float(parameter_l2_norm)}
    s=len(selected)
    first,coarse_ids,fine_ids=group_ids
    n3,n4=len(coarse_ids),len(fine_ids)
    tau_beta=PROTOCOL.c_beta*PROTOCOL.sigma*math.sqrt(s*math.log(n)/m)
    R0=float(parameter_l2_norm)+tau_beta
    budget=StageBudget(float(epsilon),PROTOCOL.delta)

    coarse,coarse_local=local_ols_coefficients_private_totalized(
        x[coarse_ids],y[coarse_ids],selected,return_diagnostics=True)
    params_a=aggregation_parameters(n3,s,R0,n,budget,PROTOCOL.precision_constant)
    center,coarse_meta=distributed_private_mean_ikos(
        coarse,np.zeros(s),params_a,streams,'coarse',materialize_ikos=private_release,
    )
    radius=(tau_beta+PROTOCOL.c1*PROTOCOL.sigma*math.sqrt(s*math.log(n)/(n3*m))+
            PROTOCOL.c2*R0*math.sqrt(s*math.log(n)*math.log(4*s/PROTOCOL.delta))/(n3*epsilon))

    fine,fine_local=local_ols_coefficients_private_totalized(
        x[fine_ids],y[fine_ids],selected,return_diagnostics=True)
    params_b=aggregation_parameters(n4,s,radius,n,budget,PROTOCOL.precision_constant)
    estimate,fine_meta=distributed_private_mean_ikos(
        fine,center,params_b,streams,'fine',materialize_ikos=private_release,
    )
    beta[selected]=estimate

    if private_release:
        safe={**meta,
              'algorithm_status':'ok',
              'parameter_l2_norm':float(parameter_l2_norm),
              'parameter_l2_norm_role':'Appendix-B Table 5 public ||theta*||_2 used in R0',
              'tau_beta':float(tau_beta),'coarse_radius':float(R0),'fine_radius':float(radius),
              'coarse':coarse_meta,'fine':fine_meta,
              'local_solver':'Appendix-B exact normal-equation inverse when defined; privacy-safe internal zero totalization on singular/undefined local inputs',
              'transport_materialized':True,
              'end_to_end_release_boundary':'only beta plus this whitelisted metadata; no local diagnostics, RNG seed/state, raw RR counts, picked indices, shares, or transcripts are returned'}
        _assert_private_metadata_safe(safe)
        return beta,safe

    diagnostics={}
    for label,values,origin,params,released in (
        ('coarse',coarse,np.zeros(s),params_a,center),
        ('fine',fine,center,params_b,estimate)):
        offsets=values-origin
        norms=np.linalg.norm(offsets,axis=1)
        clipped=offsets*np.minimum(1.,params.radius/np.maximum(norms,1e-20))[:,None]
        diagnostics[label]={'clip_fraction':float(np.mean(norms>params.radius)),
            'local_ols_norm_median':float(np.median(np.linalg.norm(values,axis=1))),
            'noise_sd_per_coordinate':math.sqrt(params.total_trials)/(2*params.n_users*params.precision),
            'release_perturbation_l2':float(np.linalg.norm(released-origin-clipped.mean(axis=0)))}
    return beta,{**meta,'algorithm_status':'ok','parameter_l2_norm':float(parameter_l2_norm),
                 'parameter_l2_norm_role':'Appendix-B Table 5 public ||theta*||_2 used in R0',
                 'tau_beta':tau_beta,'coarse_radius':R0,'fine_radius':radius,
                 'coarse':coarse_meta,'fine':fine_meta,'estimation_diagnostics':diagnostics,
                 'local_ols_fallback_count':coarse_local['local_ols_fallback_count']+fine_local['local_ols_fallback_count'],
                 'local_solver':'Appendix-B exact normal-equation inverse when defined; internal zero totalization on singular/undefined inputs; no lstsq/pinv/Ridge',
                 'appendix_local_inverse_applicable':(coarse_local['local_ols_fallback_count']+fine_local['local_ols_fallback_count']==0)}


def fit(x,y,epsilon,k,seed,*,rho=None,parameter_l2_norm=None):
    'Deterministic paper-suite simulation of Algorithms 3 and 4.'
    if parameter_l2_norm is None or not math.isfinite(parameter_l2_norm) or parameter_l2_norm<=0:
        raise ValueError('Appendix-B Table 5 requires a positive public parameter_l2_norm for R0')
    selected,meta,groups,streams=_select_support_impl(
        x,y,epsilon,k,int(seed),None,rho=rho,private_release=False,
        materialize_parallel_shuffle=False,
    )
    return _fit_after_selection(x,y,epsilon,selected,groups,streams,
                                parameter_l2_norm=parameter_l2_norm,
                                private_release=False,meta=meta)


def _contains_forbidden_key(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if str(key) in PRIVATE_FORBIDDEN_METADATA_KEYS or _contains_forbidden_key(item):
                return True
    elif isinstance(value,(list,tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _assert_private_metadata_safe(meta):
    if _contains_forbidden_key(meta):
        raise RuntimeError('private release metadata contains a forbidden local/secret field')
    if meta.get('private_release_safe') is not True or meta.get('seed_exported') is not False:
        raise RuntimeError('private release metadata boundary is not marked safe')


def fit_private(x,y,epsilon,k,*,rho=None,parameter_l2_norm=None):
    'Release-safe materialized user-level shuffled-DP implementation.'
    if parameter_l2_norm is None or not math.isfinite(parameter_l2_norm) or parameter_l2_norm<=0:
        raise ValueError('Appendix-B Table 5 requires a positive public parameter_l2_norm for R0')
    local_seed=secrets.randbits(128)
    transport_seed=secrets.randbits(128)
    selected,meta,groups,streams=_select_support_impl(
        x,y,epsilon,k,local_seed,transport_seed,rho=rho,private_release=True,
        materialize_parallel_shuffle=True,
    )
    beta,meta=_fit_after_selection(x,y,epsilon,selected,groups,streams,
                                   parameter_l2_norm=parameter_l2_norm,
                                   private_release=True,meta=meta)
    _assert_private_metadata_safe(meta)
    return beta,meta
