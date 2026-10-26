'Ma24 selector with convergence retries; same screening, CV and alpha rule.'
import warnings
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.exceptions import ConvergenceWarning
from paper_suite.algorithms.two_slr_paper import _screen_indices
from .two_solver import LocalSolveError, fixed_alpha, cross_validated


def kkt_gap(X,y,model):
    residual=y-model.predict(X)
    gradient=X.T@residual/len(y)
    beta=model.coef_;active=np.abs(beta)>1e-12;alpha=float(model.alpha_ if hasattr(model,'alpha_') else model.alpha)
    return max(float(np.max(np.abs(gradient[active]-alpha*np.sign(beta[active])),initial=0.)),
               float(np.max(np.maximum(abs(gradient[~active])-alpha,0.),initial=0.)),
               abs(float(np.mean(residual))))


def _fit_cd_checked(X,y,*,alpha=None):
    initial_warnings=0
    for attempt,limit in enumerate((1000,8000,50000,200000)):
        model=LassoCV(max_iter=limit) if alpha is None else Lasso(alpha=alpha,max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(X,y)
        count=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
        initial_warnings+=count
        if not count and np.isfinite(model.coef_).all():break
    else:raise RuntimeError('2-SLR local solver did not converge after numerical retries')
    chosen=float(model.alpha_ if alpha is None else alpha)
    gap=kkt_gap(X,y,model)

    refined=False
    if gap>max(1e-8,.01*chosen):
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model=Lasso(alpha=chosen,max_iter=200000,tol=1e-7).fit(X,y)
        refined=True;gap=kkt_gap(X,y,model)
        if gap>max(1e-8,.01*chosen):raise RuntimeError('2-SLR final local fit failed KKT check')
    return model,chosen,dict(retry_count=attempt,initial_warning_count=initial_warnings,
                            kkt_refined=int(refined),kkt_over_alpha=gap/max(chosen,1e-15))


def fit_checked(X,y,*,alpha=None):

    try:
        model,chosen,diag=_fit_cd_checked(X,y,alpha=alpha)
        return model,chosen,{**diag,'robust_fallback_count':0,'robust_checked_fits':0,'conic_fit_count':0,'accuracy_warning_count':0}
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,ConvergenceWarning) as cause:
        try:
            if alpha is None:
                model,chosen,checked=cross_validated(X,y)
            else:
                model,cert=fixed_alpha(X,y,alpha);chosen=float(alpha)
                checked=dict(kkt=cert['kkt'],checked_fits=1,conic_fits=int(cert.get('used_conic',False)),
                             accuracy_warnings=int(cert.get('accepted_with_warning',False)))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            raise LocalSolveError('Unresolved local Lasso after '+type(cause).__name__+
                                  ': '+str(exc)) from exc
        return model,chosen,dict(retry_count=0,initial_warning_count=0,kkt_refined=1,
            kkt_over_alpha=checked['kkt']/max(chosen,1e-15),robust_fallback_count=1,
            robust_checked_fits=checked['checked_fits'],conic_fit_count=checked.get('conic_fits',0),
            accuracy_warning_count=checked.get('accuracy_warnings',0))


def candidates(X,y,target,screening_features):
    screened=_screen_indices(X,y,screening_features);Xs=X[:,screened]
    target=min(max(1,int(target)),Xs.shape[1])
    model,alpha,diag=fit_checked(Xs,y)
    totals=dict(diag)
    for _ in range(15):
        nz=np.flatnonzero(abs(model.coef_)>1e-12)
        if len(nz)<target and len(nz)<Xs.shape[1]:alpha/=2.
        elif len(nz)>2*target:alpha*=1.5
        else:break
        model,alpha,diag=fit_checked(Xs,y,alpha=alpha)
        for k in ('retry_count','initial_warning_count','kkt_refined','robust_fallback_count','robust_checked_fits','conic_fit_count','accuracy_warning_count'):totals[k]+=diag[k]
        totals['kkt_over_alpha']=max(totals['kkt_over_alpha'],diag['kkt_over_alpha'])
    nz=np.flatnonzero(abs(model.coef_)>1e-12)
    if not len(nz):nz=np.array([int(np.argmax(y@Xs))])
    return screened[nz],totals


def candidate_sets(x,y,profile):
    supports=[];sizes=[];warnings_seen=retries=refinements=robust=checked=conic=accuracy=0;max_gap=0.
    for user_index,(X,Y) in enumerate(zip(x,y)):
        try:
            support,diag=candidates(X,Y,profile.local_selected,profile.screening_features)
            warnings_seen+=diag['initial_warning_count'];retries+=diag['retry_count']
            refinements+=diag['kkt_refined'];max_gap=max(max_gap,diag['kkt_over_alpha'])
            robust+=diag['robust_fallback_count'];checked+=diag['robust_checked_fits'];conic+=diag['conic_fit_count']
            accuracy+=diag['accuracy_warning_count']
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,ConvergenceWarning) as exc:

            raise LocalSolveError(f'Local user {user_index}: {exc}') from exc
        sizes.append(len(support));supports.append(support)
    return supports,dict(local_lasso_warning_count=0,local_lasso_initial_warning_count=warnings_seen,
        local_solver_retry_count=retries,local_kkt_refinement_count=refinements,
        local_lasso_totalization_count=0,local_robust_fallback_count=robust,
        local_robust_checked_fit_count=checked,local_conic_fit_count=conic,local_lasso_max_kkt_ratio=max_gap,
        local_solver_accuracy_warning_count=accuracy,
        local_candidate_size_mean=float(np.mean(sizes)),local_candidate_size_min=int(min(sizes)),
        local_candidate_size_max=int(max(sizes)),local_solver_policy='unchanged converged path; certified same-objective fallback; unresolved failure aborts research run')


def local_votes(x,y,profile,rng):
    supports,meta=candidate_sets(x,y,profile)
    return [int(rng.choice(s)) for s in supports],meta
