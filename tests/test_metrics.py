import numpy as np
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

from src.metrics import gamma_deviance, lorenz_curve, poisson_deviance, weighted_quantile_bins


def test_deviances_match_sklearn():
    rng = np.random.default_rng(0)
    y, mu = rng.poisson(0.3, 1000).astype(float), rng.uniform(0.1, 0.6, 1000)
    assert np.isclose(poisson_deviance(y, mu), mean_poisson_deviance(y, mu))
    ys, ms = rng.gamma(2, 500, 1000), rng.uniform(500, 1500, 1000)
    assert np.isclose(gamma_deviance(ys, ms), mean_gamma_deviance(ys, ms))


def test_gini_perfect_and_random():
    rng = np.random.default_rng(1)
    loss = rng.exponential(1, 20000) * (rng.random(20000) < 0.1)
    e = np.ones_like(loss)
    assert lorenz_curve(loss, loss, e)[2] > 0.85
    assert abs(lorenz_curve(rng.random(20000), loss, e)[2]) < 0.05


def test_weighted_bins_equal_weight():
    s, w = np.arange(1000.0), np.ones(1000)
    b = weighted_quantile_bins(s, w, 10)
    assert set(np.bincount(b)) == {100}
