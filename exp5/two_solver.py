'Rare-failure Lasso solver, preserving the objective and author CV rule.'
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
from sklearn.linear_model import LassoCV, LassoLars
from sklearn.model_selection import KFold


class LocalSolveError(RuntimeError):
    'Internal research abort: no default coordinate is substituted.'


@dataclass
class LinearFit:
    coef_: np.ndarray
    intercept_: float
    alpha: float

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_


def certificate(X, y, beta, alpha):
    'Primal/dual and KKT checks on the centered original Lasso objective.'
    if not np.isfinite(beta).all():
        return dict(certified=False, kkt=float('inf'), dual_gap=float('inf'))
    residual = y-X@beta
    gradient = X.T@residual/len(y)
    active = abs(beta)>1e-12
    kkt = max(float(np.max(abs(gradient[active]-alpha*np.sign(beta[active])),initial=0.)),
              float(np.max(np.maximum(abs(gradient[~active])-alpha,0.),initial=0.)))
    primal = .5*float(np.mean(residual**2))+alpha*float(np.sum(abs(beta)))
    factor = min(1.,alpha/max(float(np.max(abs(gradient),initial=0.)),1e-300))
    dual_residual = residual*factor
    dual = float(np.mean(y*dual_residual)-.5*np.mean(dual_residual**2))
    dual_gap = max(0.,primal-dual)
    good = (np.isfinite([kkt,primal,dual_gap]).all() and
            kkt<=max(1e-10,alpha*1e-5) and dual_gap<=max(1e-12,1e-7*primal))
    return dict(certified=bool(good),kkt=kkt,primal=primal,dual_gap=dual_gap)


def practical_certificate(X,y,beta,alpha):
    'Ordinary numerical accuracy, not a privacy/theorem certificate.'
    cert=certificate(X,y,beta,alpha)
    kkt_limit=max(1e-8,.01*alpha)
    gap_limit=max(1e-12,1e-4*float(np.mean(y*y)))
    accepted=(np.isfinite(beta).all() and np.isfinite(list(cert.values())).all()
              and cert['kkt']<=kkt_limit and cert['dual_gap']<=gap_limit)
    return dict(**cert,numerical_accepted=bool(accepted),accepted_with_warning=bool(accepted and not cert['certified']),
                numerical_policy='sklearn_default_dual_gap_and_existing_cd_kkt',
                kkt_limit=kkt_limit,dual_gap_limit=gap_limit)


def _certified_conic_lasso(A,t,alpha):
    'Exact residual/epigraph QP; accepted only by the unchanged Lasso certificate.'
    import clarabel
    n,p=A.shape
    P=sparse.diags(np.r_[np.zeros(2*p),np.full(n,1./n)],format='csc')
    q=np.r_[np.zeros(p),np.full(p,alpha),np.zeros(n)]
    identity=sparse.eye(p,format='csc');zero=sparse.csc_matrix((p,n))
    C=sparse.vstack([
        sparse.hstack([sparse.csc_matrix(A),sparse.csc_matrix((n,p)),-sparse.eye(n,format='csc')]),
        sparse.hstack([identity,-identity,zero]),
        sparse.hstack([-identity,-identity,zero])],format='csc')
    rhs=np.r_[t,np.zeros(2*p)]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=300
    settings.tol_gap_abs=1e-13;settings.tol_gap_rel=1e-12;settings.tol_feas=1e-12
    settings.reduced_tol_gap_abs=1e-13;settings.reduced_tol_gap_rel=1e-12;settings.reduced_tol_feas=1e-12
    settings.max_threads=1
    best=None;practical=[]


    for objective_scale in (1.,1e4,1e6):
        solution=clarabel.DefaultSolver(P*objective_scale,q*objective_scale,C,rhs,
            [clarabel.ZeroConeT(n),clarabel.NonnegativeConeT(2*p)],settings).solve()
        b=np.asarray(solution.x[:p],float)


        for dust in (0.,1e-12,1e-10,1e-8,1e-6):
            candidate=b.copy();candidate[abs(candidate)<=dust]=0.
            cert=certificate(A,t,candidate,alpha)
            info=dict(**cert,conic_status=str(solution.status),conic_iterations=solution.iterations,
                      conic_objective_scale=objective_scale)
            if cert['certified']:return candidate,info
            ordinary=practical_certificate(A,t,candidate,alpha)
            if ordinary['numerical_accepted']:
                practical.append((candidate,dict(**ordinary,conic_status=str(solution.status),
                    conic_iterations=solution.iterations,conic_objective_scale=objective_scale)))
            if best is None or cert['dual_gap']<best['dual_gap']:best=info
    if practical:


        return min(practical,key=lambda item:item[1]['kkt']/item[1]['kkt_limit']+
                   item[1]['dual_gap']/item[1]['dual_gap_limit'])
    raise LocalSolveError('Conic Lasso failed original numerical certificate; '+repr(best))


def fixed_alpha(X,y,alpha):
    X=np.asarray(X,float);y=np.asarray(y,float);alpha=float(alpha)
    if X.ndim!=2 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha<=0:
        raise LocalSolveError('Invalid local Lasso inputs')
    xm=X.mean(0);ym=float(y.mean());A=X-xm;t=y-ym;p=X.shape[1]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        try:b=LassoLars(alpha=alpha,max_iter=10000).fit(X,y).coef_.copy()
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError):b=np.zeros(p)
    cert=certificate(A,t,b,alpha);used_qp=False;used_conic=False
    if not cert['certified']:
        if not np.isfinite(b).all():b=np.zeros(p)

        def fg(z):
            r=A@(z[:p]-z[p:])-t;g=A.T@r/len(t)
            return .5*float(r@r)/len(t)+alpha*float(z.sum()),np.r_[g+alpha,-g+alpha]
        z0=np.r_[np.maximum(b,0),np.maximum(-b,0)]
        result=minimize(fg,z0,jac=True,method='L-BFGS-B',bounds=[(0.,None)]*(2*p),
            options=dict(maxiter=100000,gtol=1e-13,ftol=1e-16,maxls=50,maxcor=50))
        b=result.x[:p]-result.x[p:];cert=certificate(A,t,b,alpha);used_qp=True
        if not cert['certified']:
            result=minimize(fg,result.x,jac=True,method='SLSQP',bounds=[(0.,None)]*(2*p),
                options=dict(maxiter=10000,ftol=1e-14))
            b=result.x[:p]-result.x[p:];cert=certificate(A,t,b,alpha)
        if not cert['certified']:
            active=np.flatnonzero(abs(b)>1e-9)
            if len(active):
                V=A[:,active];rhs=V.T@t/len(t)-alpha*np.sign(b[active])
                face=np.linalg.lstsq(V.T@V/len(t),rhs,rcond=1e-14)[0]
                if np.all(face*np.sign(b[active])>=0):
                    polished=np.zeros(p);polished[active]=face
                    candidate=certificate(A,t,polished,alpha)
                    if candidate['certified']:b,cert=polished,candidate
        if not cert['certified']:
            b,cert=_certified_conic_lasso(A,t,alpha);used_conic=True
    return LinearFit(b,ym-float(xm@b),alpha),dict(**cert,used_qp=used_qp,used_conic=used_conic)


def cross_validated(X,y):


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grid=LassoCV(max_iter=1).fit(X,y).alphas_.copy()
    errors=np.empty((len(grid),5));certs=[]
    for fold,(train,valid) in enumerate(KFold(5).split(X)):
        for i,alpha in enumerate(grid):
            model,cert=fixed_alpha(X[train],y[train],float(alpha));certs.append(cert)
            errors[i,fold]=float(np.mean((model.predict(X[valid])-y[valid])**2))
    alpha=float(grid[np.argmin(errors.mean(axis=1))])
    model,cert=fixed_alpha(X,y,alpha);certs.append(cert)
    return model,alpha,dict(kkt=cert['kkt'],checked_fits=len(certs),
        qp_fits=sum(c['used_qp'] for c in certs),conic_fits=sum(c['used_conic'] for c in certs),
        accuracy_warnings=sum(bool(c.get('accepted_with_warning',False)) for c in certs),
        max_dual_gap=max(c['dual_gap'] for c in certs))
