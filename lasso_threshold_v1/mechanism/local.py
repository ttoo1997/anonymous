'Appendix Table 4 local Lasso shared by both two-stage methods.'
import warnings
import math
import numpy as np
from sklearn.linear_model import Lasso, LassoLars
from sklearn.exceptions import ConvergenceWarning

from .appendix import C_LAMBDA, SIGMA, RHO, LOCAL_COEFFICIENT_THRESHOLD


def _kkt_violation(x, y, beta, alpha):
    "Check the original Lasso objective's first-order conditions."
    grad=x.T @ (y-x @ beta)/len(y)
    active=np.abs(beta)>1e-8
    active_gap=np.max(np.abs(grad[active]-alpha*np.sign(beta[active])),initial=0.)
    inactive_gap=np.max(np.maximum(np.abs(grad[~active])-alpha,0.),initial=0.)
    return float(max(active_gap,inactive_gap))


def _fit_local(x,y,alpha):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        try:
            model=Lasso(alpha=alpha,fit_intercept=False,tol=1e-5,max_iter=200000).fit(x,y)
            return model.coef_, 'coordinate_descent', None
        except ConvergenceWarning:
            model=LassoLars(alpha=alpha,fit_intercept=False,max_iter=1000).fit(x,y)
    beta=model.coef_
    gap=_kkt_violation(x,y,beta,alpha)
    if not np.isfinite(gap) or gap>max(1e-8,alpha*1e-4):
        raise ValueError(f'fallback Lasso failed KKT check: {gap}')
    return beta,'lasso_lars_kkt_checked',gap


def _validate_inputs(x: np.ndarray,y: np.ndarray,rho: float,sigma: float):
    if x.ndim!=3 or y.shape!=x.shape[:2] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid local data')
    if not np.isfinite(rho) or rho<=0:
        raise ValueError('rho must be a positive public signal-strength lower bound')
    if not np.isfinite(sigma) or sigma<=0:
        raise ValueError('sigma must be a positive public noise scale')


def candidates(x: np.ndarray,y: np.ndarray,k:int, *, total_users=None, rho=RHO, c_lambda=C_LAMBDA, sigma=SIGMA):
    'Appendix-B Algorithm 3 local Lasso followed by the exact rho/2 threshold.'
    _validate_inputs(x,y,rho,sigma)
    total_users=len(x) if total_users is None else total_users
    alpha=c_lambda*sigma*math.sqrt(math.log(x.shape[2]*total_users)/x.shape[1])
    threshold=rho/2
    supports=[];fallback_count=0;fallback_max_kkt=0.;totalization_count=0
    for xi,yi in zip(x,y):
        try:
            beta,solver,gap=_fit_local(xi,yi,alpha)
            if not np.isfinite(beta).all():raise FloatingPointError('nonfinite local fit')
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning):
            supports.append(np.empty(0,dtype=int));totalization_count+=1
            continue
        if solver!='coordinate_descent':
            fallback_count+=1
            fallback_max_kkt=max(fallback_max_kkt,gap)
        supports.append(np.flatnonzero(np.abs(beta)>=threshold))
    return supports,{'lasso_alpha':alpha,'coefficient_threshold':threshold,
                     'local_empty_rate':float(np.mean([len(s)==0 for s in supports])),
                     'local_solver_fallback_count':fallback_count,
                     'local_lasso_totalization_count':totalization_count,
                     'local_solver_fallback_max_kkt':fallback_max_kkt,
                     'rho':rho,'sigma':sigma,'local_rule':'Appendix-B Algorithm 3: Lasso; |coef|>=rho/2; uniform index; empty->full domain'}


def candidates_private_totalized(x: np.ndarray,y: np.ndarray,k:int, *, total_users=None,
                                 rho=RHO,c_lambda=C_LAMBDA,sigma=SIGMA):
    'Private-release Stage-1 local preprocessing with a total data-domain map.'
    _validate_inputs(x,y,rho,sigma)
    total_users=len(x) if total_users is None else total_users
    alpha=c_lambda*sigma*math.sqrt(math.log(x.shape[2]*total_users)/x.shape[1])
    threshold=rho/2
    supports=[]
    for xi,yi in zip(x,y):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                beta,_,_=_fit_local(xi,yi,alpha)
            if not np.isfinite(beta).all():
                raise FloatingPointError('nonfinite private local Lasso coefficients')
            support=np.flatnonzero(np.abs(beta)>=threshold)
        except Exception:

            support=np.empty(0,dtype=int)
        supports.append(support)
    return supports
