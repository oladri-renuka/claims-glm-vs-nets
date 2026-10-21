"""Actuarial evaluation metrics."""
from __future__ import annotations

import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean unit Poisson deviance per policy: 2 * mean[ y log(y/mu) - (y - mu) ]."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def gamma_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean unit Gamma deviance: 2 * mean[ (y - mu)/mu - log(y/mu) ]."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(2.0 * np.mean((y - mu) / mu - np.log(y / mu)))


def lorenz_curve(pred: np.ndarray, actual: np.ndarray, exposure: np.ndarray):
    """Ordered Lorenz curve: sort by predicted premium rate (ascending); x = cumulative exposure
    share, y = cumulative share of actual losses. Returns (x, y, gini)."""
    order = np.argsort(pred, kind="stable")
    x = np.concatenate([[0.0], np.cumsum(exposure[order]) / exposure.sum()])
    y = np.concatenate([[0.0], np.cumsum(actual[order]) / actual.sum()])
    gini = 1.0 - 2.0 * np.trapezoid(y, x)
    return x, y, float(gini)


def balance_ratio(y: np.ndarray, mu: np.ndarray) -> float:
    """Sum of predicted / sum of actual (1.0 = the balance property holds)."""
    return float(np.sum(mu) / np.sum(y))


def weighted_quantile_bins(score: np.ndarray, weight: np.ndarray, n: int) -> np.ndarray:
    """Assign each row to one of n bins of (approximately) equal total weight by score."""
    order = np.argsort(score, kind="stable")
    w = weight[order]
    cw = (np.cumsum(w) - w / 2) / w.sum()   # weight midpoint of each row avoids boundary rounding
    bins = np.empty(len(score), dtype=int)
    bins[order] = np.minimum((cw * n).astype(int), n - 1)
    return bins
