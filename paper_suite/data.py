from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SyntheticSetting
from lasso_threshold_v1.dataset import synthetic, COEFFICIENT_PROTOCOL


@dataclass(frozen=True)
class RegressionData:
    user_x: np.ndarray
    user_y: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    beta_true: np.ndarray | None
    target_scale: float = 1.0
    lasso_x: np.ndarray | None = None
    lasso_y: np.ndarray | None = None
    metadata: dict | None = None


def make_synthetic(setting: SyntheticSetting, seed: int, test_samples: int = 4000) -> RegressionData:
    setting.validate()
    user_x,user_y,x_test,y_test,beta=synthetic(setting.n,setting.m,setting.d,setting.k,seed,test_samples,setting.noise_std)
    return RegressionData(user_x, user_y, x_test, y_test, beta,
        metadata={'coefficient_protocol':COEFFICIENT_PROTOCOL,
                  'true_nonzero_coefficients':beta[beta!=0].tolist()})


def load_tabular_csv(path: Path, target: str, m: int, seed: int,
                     test_size: float = 0.2, max_features: int | None = None) -> RegressionData:
    'Load a numeric regression CSV and group training rows into natural-size pseudo-users.'
    if m < 1 or not 0 < test_size < 1 or (max_features is not None and max_features < 1):
        raise ValueError("invalid grouping, split or feature count")
    frame = pd.read_csv(path)
    if target not in frame:
        raise ValueError(f"target column {target!r} is absent")
    numeric = frame.select_dtypes(include=[np.number]).dropna()
    if target not in numeric or len(numeric) < 2 * m:
        raise ValueError("target must be numeric and enough complete rows are required")
    y = numeric.pop(target).to_numpy(float)
    x = numeric.to_numpy(float)
    if x.shape[1] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite features and target are required")
    if max_features is not None:
        x = x[:, :max_features]
    train, test = train_test_split(np.arange(len(x)), test_size=test_size, random_state=seed)
    scaler = StandardScaler().fit(x[train])
    x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])
    y_mean, y_scale = float(y[train].mean()), float(y[train].std())
    if y_scale <= 0:
        raise ValueError("training target is constant")
    y_train, y_test = (y[train] - y_mean) / y_scale, (y[test] - y_mean) / y_scale
    n = len(train) // m
    if n < 1:
        raise ValueError("no complete users after grouping")
    return RegressionData(x_train[:n*m].reshape(n, m, -1), y_train[:n*m].reshape(n, m),
                          x_test, y_test, None, y_scale, x_train, y_train,
                          {"train_rows": len(train), "test_rows": len(test),
                           "used_private_rows": n*m, "discarded_group_tail": len(train)-n*m,
                           "dropped_missing_rows": len(frame)-len(numeric),
                           "excluded_nonnumeric_columns": list(frame.select_dtypes(exclude=[np.number]).columns),
                           "train_indices": train.tolist(), "test_indices": test.tolist()})
