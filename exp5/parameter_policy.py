'Resolve the predeclared synthetic LDPIHT policy using public budgets only.'
from .common import ROOT
from .privacy import budgets
from paper_suite.algorithms.iht_stabilized_profile import stabilized_parameters
from dp_safety.parameters import APPROXIMATE_DP_DELTA

def resolve_parameters(method, profile, epsilon, m, track, N):
    p=dict(profile)
    policy=p.pop('parameter_policy',None)
    if policy is None:return p
    if method!='ldpiht' or policy!='frozen_stabilized_epsilon_v1':
        raise ValueError('Unknown algorithm parameter policy')
    if set(p)!={'k'}:raise ValueError('Unexpected overrides to frozen stabilization policy')
    budget=budgets(method,epsilon,APPROXIMATE_DP_DELTA,m,track)
    return dict(k=p['k'],**stabilized_parameters(p['k'],N,budget['record_epsilon']))
