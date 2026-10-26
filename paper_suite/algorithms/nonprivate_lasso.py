'Non-private Lasso baseline using Ma24 Appendix-E.1 settings.'
import warnings
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning
from ..data import RegressionData


def fit_lasso(data: RegressionData) -> tuple[np.ndarray, dict]:
    x = data.lasso_x if data.lasso_x is not None else data.user_x.reshape(-1, data.user_x.shape[-1])
    y = data.lasso_y if data.lasso_y is not None else data.user_y.ravel()


    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)


        model = LassoCV(alphas=300, cv=5, fit_intercept=False,
                        max_iter=3000, tol=1e-4, n_jobs=1).fit(x, y)
    convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    coef = np.asarray(model.coef_, dtype=float)
    alpha = float(model.alpha_)
    if coef.shape != (x.shape[1],) or not np.isfinite(coef).all() or not np.isfinite(alpha):
        raise ValueError("LassoCV returned a non-finite or malformed solution")
    messages = list(dict.fromkeys(str(w.message) for w in convergence))
    return coef, {"alpha": alpha, "privacy": "none",
                  "convergence_warning": bool(convergence),
                  "convergence_warning_count": len(convergence),
                  "convergence_warning_messages": messages[:3],
                  "warning_policy": "ConvergenceWarning is recorded but non-fatal; genuine fit errors remain fatal",
                  "implementation":"Ma24 Appendix-E.1 LassoCV: 300 alphas, max_iter=3000, tol=1e-4"}
