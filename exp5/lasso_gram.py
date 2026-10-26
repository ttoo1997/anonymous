'Full-data, 300-alpha, grouped CV using sufficient statistics.'
import warnings
import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning

def path_from_gram(G,h,yy,n,alphas,tol=1e-4):
    eig,Q=np.linalg.eigh((G+G.T)/2)
    cut=max(float(eig.max()),1.)*1e-13
    if eig.min() < -cut*100:raise FloatingPointError('non-PSD nonprivate Gram matrix')
    positive=eig>cut;root=np.sqrt(np.maximum(eig,0))
    R=root[:,None]*Q.T
    z=np.zeros(len(h));z[positive]=(Q.T@h)[positive]/root[positive]
    residual=yy-z@z
    if residual < -max(yy,1.)*1e-7:raise FloatingPointError('invalid quadratic sufficient statistics')
    R=np.asfortranarray(np.vstack([R,np.zeros((1,len(h)))]));z=np.r_[z,np.sqrt(max(0.,residual))]
    _,coef,gaps=lasso_path(R,z,alphas=np.asarray(alphas)*n/len(R),precompute=R.T@R,Xy=R.T@z,max_iter=8000,tol=tol)
    return coef,gaps*len(R)/n

def fit(X,y,days,alphas=300,tol=1e-4):
    N,d=X.shape;folds=list(GroupKFold(n_splits=min(5,len(np.unique(days)))).split(X,y,days))
    stats=[]
    for _,ids in folds:
        G=np.zeros((d,d));h=np.zeros(d);yy=0.
        for i in range(0,len(ids),8192):
            ix=ids[i:i+8192];xx=np.asarray(X[ix],float);yb=np.asarray(y[ix],float)
            G+=xx.T@xx;h+=xx.T@yb;yy+=float(yb@yb)
        stats.append((G,h,yy,len(ids)))
    G=sum(a[0] for a in stats);h=sum(a[1] for a in stats);yy=sum(a[2] for a in stats)
    maximum=max(float(np.max(abs(h)))/N,np.finfo(float).resolution)
    grid=np.geomspace(maximum,maximum*.001,alphas)
    errors=[]
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning)
        for A,b,c,size in stats:
            coef,_=path_from_gram(G-A,h-b,yy-c,N-size,grid,tol)
            loss=(np.sum(coef*(A@coef),axis=0)-2*b@coef+c)/size
            errors.append(loss)
        mean=np.mean(errors,axis=0);index=int(np.argmin(mean));alpha=float(grid[index])
        coef,gaps=path_from_gram(G,h,yy,N,[alpha],tol)
    return coef[:,0],dict(alpha=alpha,cv='whole-day GroupKFold, full training population, 300 alphas',
        solver='sufficient-statistic Lasso objective; no feature reduction',training_rows=N,
        warning_count=sum(issubclass(w.category,ConvergenceWarning) for w in ws),
        cv_mse=float(mean[index]),dual_gap=float(gaps[0]),max_iter=8000)
