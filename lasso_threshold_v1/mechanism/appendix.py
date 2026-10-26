'Appendix-B formula instantiation plus a finite-sample Stage-1 DP certificate.'
from dataclasses import replace, asdict
from dp_safety.aggregation import certify_existing_noise, security_bits, ensure_local_trials
from dp_safety.rr import round_flip_probability_up
import math
from .core import make_table_aggregation_parameters
from .privacy_accountant import certify_stage1_gamma

C_GAMMA=32.0
C_LAMBDA=0.01
SIGMA=0.25
RHO=0.125
LOCAL_COEFFICIENT_THRESHOLD=RHO/2
KAPPA_REFERENCE=1.0

def selection_parameters(n,m,d,k,epsilon,delta,*,n1=None,rho=RHO,c_gamma=C_GAMMA,c_lambda=C_LAMBDA,sigma=SIGMA):
    'Instantiate Appendix-B Table 4 with an independent finite-sample DP floor.'
    if min(n,m,d,k)<1 or k>d or not math.isfinite(epsilon) or epsilon<=0 or not 0<delta<1:
        raise ValueError('invalid appendix parameters')
    if not math.isfinite(rho) or rho<=0:
        raise ValueError('rho must be a positive public signal-strength lower bound')
    if not math.isfinite(sigma) or sigma<=0:
        raise ValueError('sigma must be a positive public noise scale')
    n1=(n//2) if n1 is None else int(n1)
    if n1<1 or n1>=n:
        raise ValueError('n1 must satisfy 1 <= n1 < n')
    certificate=certify_stage1_gamma(n1,float(epsilon),float(delta),appendix_c_gamma=float(c_gamma))
    gamma=round_flip_probability_up(certificate.gamma)
    alpha=c_lambda*sigma*math.sqrt(math.log(d*n)/m)
    upper=KAPPA_REFERENCE*rho/(6*math.sqrt(k))
    sample_terms=(sigma**2*k*math.log(d*n)/(KAPPA_REFERENCE**2*rho**2),
                  (k*math.log(math.e*d/k)+math.log(n))/(KAPPA_REFERENCE**2),
                  k+math.log(n))
    n1_proxy=k*math.sqrt(math.log(d*n)*math.log(1/delta))/epsilon
    return {'gamma':gamma,
            'gamma_table_lower':certificate.appendix_gamma_floor,
            'gamma_certified_floor':certificate.certified_gamma_floor,
            'gamma_choice':'max(Appendix-B C_gamma=32 reference floor, independently certified finite-sample DP floor)',
            'stage1_privacy_certificate':certificate.as_dict(),
            'lasso_alpha':alpha,'coefficient_threshold':rho/2,
            'coefficient_threshold_source':'Appendix-B Algorithm 3: rho/2',
            'rho':rho,'c_gamma':c_gamma,'c_lambda':c_lambda,'sigma':sigma,'selection_users':n1,
            'lambda_upper_bound_reference':upper,
            'lambda_window_reference_satisfied':alpha<=upper,
            'lambda_upper_reference_satisfied':alpha<=upper,
            'lambda_window_label_scope':'legacy field checks upper bound only; not a utility-theorem certificate',
            'kappa_reference':KAPPA_REFERENCE,
            'proposition1_unit_constant_m_proxy':max(sample_terms),
            'proposition1_unit_constant_m_terms':sample_terms,
            'proposition1_m_above_unit_constant_proxy':m>=max(sample_terms),
            'proposition1_unit_constant_n1_proxy':n1_proxy,
            'proposition1_n1_above_unit_constant_proxy':n1>=n1_proxy,
            'constants_status':'C_gamma=32 retained as Appendix-B reference floor; privacy is independently certified and does not rely on proving C_gamma=32'}

def aggregation_parameters(n_users,dimension,radius,n_total,budget,precision_constant):
    base=make_table_aggregation_parameters(n_users,dimension,radius,n_total,budget,precision_constant)
    sensitivity=base.integer_sensitivity
    requested=base.requested_trials
    local=base.local_trials
    local=ensure_local_trials(int(n_users),int(dimension),sensitivity,
                              budget.epsilon,budget.delta,int(local))
    modulus=math.ceil(4*n_users*(base.precision*radius+local))
    modulus+=1-modulus%2

    shares=max(3,math.ceil((2*(math.log2(8*dimension/budget.delta)+budget.epsilon/math.log(2))+math.log2(modulus))/
                           (math.log2(n_users)-math.log2(math.e)))+1)
    result=replace(base,integer_sensitivity=sensitivity,requested_trials=requested,
                   local_trials=local,total_trials=n_users*local,modulus=modulus,shares=shares)
    cert=certify_existing_noise(n_users=n_users,dimension=dimension,modulus=modulus,
        total_trials=result.total_trials,sensitivity_l2=sensitivity,
        epsilon=budget.epsilon,delta=budget.delta,current_shares=shares)
    result=replace(result,shares=cert.shuffled_shares,
        security_bits=security_bits(budget.epsilon,budget.delta,dimension),
        delta_noise=budget.delta/2,delta_security=budget.delta/2,
        privacy_certificate=asdict(cert))
    result.validate()
    return result
