'Thin reproducibility adapter for the unmodified Ma24 public ULDPFS code.'
import threading
import numpy as np
from ..vendor.ma2024.ULDPFS import ULDPFS

_LOCK=threading.Lock()
PROTOCOL='ma24_author_unmodified_screening_v3'

def fit(x,y,epsilon,k,seed):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=3 or y.shape!=x.shape[:2] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite aligned user data required')
    n,m,d=x.shape
    if n<80 or m<2 or d<2 or not 1<=k<=d or not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('unsupported author-code setting')
    parameters=dict(epsilon=epsilon,selector='screening',
        selector_params={'num_features':max(1,k//2),'screening_num_features':min(d,k*k//4)},
        heavyhitter_params={'min_hitters':k,'alpha':.1},
        regressor_params={'num_bins':16,'B':1})
    with _LOCK:
        state=np.random.get_state()
        try:
            np.random.seed(seed)
            model=ULDPFS(**parameters).fit(x,y)
        finally:
            np.random.set_state(state)
    beta=np.asarray(model.coef_,float)
    if beta.shape!=(d,) or not np.isfinite(beta).all():
        raise RuntimeError('author estimator produced invalid coefficients')
    return beta,dict(support=np.asarray(model.selected_indexes,dtype=int),
        protocol_version=PROTOCOL,implementation='Ma24 public ULDPFS; unmodified positive marginal Screening selector',
        parameters=parameters,selection_count=n//2,
        privacy_unit='author intended user-level; implementation not certified',
        privacy_validation='author code retained for provenance; formal suite uses two_slr_paper.py',
        local_rule='positive uncentered y@X correlation; author quantile rule and uniform index',
        coefficient_threshold=None)

def select_support(x,y,epsilon,k,seed):
    _,meta=fit(x,y,epsilon,k,seed)
    return meta['support'],meta
