'Fail closed on inconsistent per-fit privacy accounting; quality is separate.'
import math
import numpy as np


def validate_fit(method,beta,details,N,d):
    beta=np.asarray(beta)
    if beta.shape!=(d,) or not np.isfinite(beta).all():raise ValueError('Invalid estimator output')
    budget=details['budget'];certificates=[]
    if method=='u2r':
        if details.get('local_lasso_totalization_count',0):raise ValueError('Unresolved U2R local solves are not valid formal results')
        if details.get('local_ols_fallback_count',0) or details.get('appendix_local_inverse_applicable') is False:
            raise ValueError('U2R local OLS inverse inapplicable; research run stopped, seed retained')
        if details.get('support_cap_policy') is not None:
            m=details['local_sample_count'];cap=details['public_support_cap']
            if not 1<=cap<=m-2 or len(details['support'])>cap or not details['support_cap_dp_postprocessing']:
                raise ValueError('U2R public support cap violated')
        c=details['stage1_privacy_certificate']
        if c['certified_total_epsilon_bound']>budget['epsilon']*(1+1e-10) or c['certified_total_delta_bound']>budget['delta']*(1+1e-10):
            raise ValueError('U2R selection certificate exceeds target')
        for stage in ('coarse','fine'):
            if stage in details:certificates.append((details[stage]['parameters']['privacy_certificate'],budget['epsilon'],budget['delta']))
    elif method=='2slr':
        if details.get('local_lasso_totalization_count',0):raise ValueError('Unresolved 2-SLR local solves are not valid formal results')
        if not details['hh_one_report_per_user'] or details['hh_report_draws_max']>1:raise ValueError('Repeated HH privacy reports')
        if details['algorithm_status']=='ok':
            if not details['stage2_groups_disjoint']:raise ValueError('Stage-2 cohorts overlap')
            if details['range_budget_per_coordinate']*details['rotation_dimension']>budget['epsilon']*(1+1e-12):raise ValueError('Range budget exceeds target')
            if details['mean_budget_per_user']>budget['epsilon']*(1+1e-12):raise ValueError('Mean budget exceeds target')
    elif method=='sample_sdp':
        certificates.append((details['privacy_certificate'],budget['record_epsilon'],budget['record_delta']))
        if np.linalg.norm(beta,1)>1+1e-8:raise ValueError('SampleSDP unit L1 constraint violated')
        if details.get('estimator_variant')=='SampleSDP-Ridge':
            if not details['privacy_postprocessing_only'] or details['regularized_min_eigenvalue']<=0:raise ValueError('Invalid Ridge postprocessing')
    elif method=='ldpiht':
        if sum(details['cohort_sizes'])!=N or np.linalg.norm(beta)>1+1e-8:raise ValueError('LDP-IHT population/projection mismatch')
        radius=math.sqrt(d)*details['x_clip']*(math.sqrt(details['hard_threshold_k'])*details['x_clip']+details['y_clip'])
        cap=details.get('gradient_clip')
        if cap is None:
            if not math.isclose(radius,details['gradient_radius'],rel_tol=1e-12):raise ValueError('LDP-IHT sensitivity mismatch')
        else:
            if not math.isfinite(cap) or cap<=0 or details.get('radius_mode')!='public_iterate':raise ValueError('Invalid stabilized radius policy')
            if not math.isclose(radius,details['paper_constant_radius'],rel_tol=1e-12):raise ValueError('LDP-IHT paper bound mismatch')
            trace=details['iteration_diagnostics']
            if len(trace)!=details['iterations'] or not trace:raise ValueError('Missing radius history')
            for t in trace:
                legal=t['analytic_radius'];actual=t['report_radius']
                if not math.isfinite(legal) or not 0<legal<=radius*(1+1e-12):raise ValueError('Illegal conditional radius')
                if not math.isclose(actual,min(cap,legal),rel_tol=1e-12):raise ValueError('Clipping/randomization radii differ')
            first=math.sqrt(d)*details['x_clip']*details['y_clip']
            if not math.isclose(trace[0]['analytic_radius'],first,rel_tol=1e-12):raise ValueError('Wrong zero-iterate radius')
            if not math.isclose(details['gradient_radius'],max(t['report_radius'] for t in trace),rel_tol=1e-12):raise ValueError('Wrong radius summary')
            if details.get('gradient_unbiased_for')!='clipped gradient':raise ValueError('Incorrect unbiasedness claim')
            if details.get('original_utility_theorem_claimed') is not False:raise ValueError('Unsupported original utility theorem claim')
        if details.get('delta_ldp_target')!=0.0:raise ValueError('LDP-IHT must remain pure DP')
        if not math.isclose(details['epsilon_ldp_target'],budget['record_epsilon'],rel_tol=1e-12):raise ValueError('LDP-IHT budget mismatch')
    for c,e,de in certificates:
        if c['epsilon_noise_certified']>e*(1+1e-10) or c['delta_total_bound']>de*(1+1e-10):
            raise ValueError('Aggregation privacy certificate exceeds target')
