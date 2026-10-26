from __future__ import annotations

import numpy as np


def relative_coefficient_error(beta: np.ndarray, truth: np.ndarray) -> float:
    denominator = np.linalg.norm(truth)
    if denominator == 0:
        raise ValueError("relative error is undefined for a zero truth vector")
    return float(np.linalg.norm(beta - truth) / denominator)


def support_recall(support: np.ndarray, truth: np.ndarray) -> float:
    actual = set(np.flatnonzero(truth))
    chosen = set(map(int, np.asarray(support).ravel()))
    return len(actual & chosen) / len(actual)


def mse(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x @ beta - y) ** 2))

