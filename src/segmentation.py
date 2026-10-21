"""I5 / I6: risk segmentation from VAE latents vs raw features vs GLM deciles, and
autoencoder anomaly inspection. Clusterers are fitted on train; everything reported
is measured on the test split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.data.features import NNPreprocessor
from src.data.schema import CATEGORIES, NN_CATEGORICAL
from src.metrics import poisson_deviance, weighted_quantile_bins

FIT_SAMPLE = 100_000


def raw_features(pre: NNPreprocessor, df: pd.DataFrame) -> np.ndarray:
    """Standardized continuous features + one-hot categoricals (the 'no learning' space)."""
    xc, xk = pre.transform(df)
    onehots = [np.eye(len(CATEGORIES[c]))[xk[:, i]] for i, c in enumerate(NN_CATEGORICAL)]
    return np.concatenate([xc, *onehots], axis=1)


def segment_stats(labels: np.ndarray, df: pd.DataFrame) -> dict:
    """Observed-frequency spread across segments on the evaluation data."""
    g = pd.DataFrame({"s": labels, "n": df.ClaimNb.to_numpy(), "e": df.Exposure.to_numpy()}).groupby("s").sum()
    g["freq"] = g.n / g.e
    w = g.e / g.e.sum()
    overall = g.n.sum() / g.e.sum()
    # Deviance of pricing every policy at its segment's observed rate, relative to one rate.
    rate = pd.Series(labels).map(g.freq).to_numpy() * df.Exposure.to_numpy()
    dev_seg = poisson_deviance(df.ClaimNb, np.maximum(rate, 1e-12))
    dev_one = poisson_deviance(df.ClaimNb, overall * df.Exposure.to_numpy())
    return {
        "k": len(g), "max_min_ratio": float(g.freq.max() / max(g.freq.min(), 1e-12)),
        "freq_variance": float((w * (g.freq - overall) ** 2).sum()),
        "freq_cv": float(np.sqrt((w * (g.freq - overall) ** 2).sum()) / overall),
        "deviance_reduction_pct": float(100 * (dev_one - dev_seg) / dev_one),
        "min_exposure_share": float(w.min()),
    }


def _silhouette(x: np.ndarray, labels: np.ndarray, n: int, seed: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    idx = np.random.default_rng(seed).choice(len(x), size=min(n, len(x)), replace=False)
    return float(silhouette_score(x[idx], labels[idx]))


def run_segmentation(train: pd.DataFrame, test: pd.DataFrame, vaes: list, glm, seg_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Returns a long table (method, k, seed, metrics...) and fitted objects for serving."""
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(len(train), size=min(FIT_SAMPLE, len(train)), replace=False)
    fit_df = train.iloc[fit_idx]
    pre = vaes[0].pre
    raw_fit, raw_test = raw_features(pre, fit_df), raw_features(pre, test)
    glm_fit = np.log(glm.predict(fit_df.assign(Exposure=1.0)))
    glm_test = np.log(glm.predict(test.assign(Exposure=1.0)))
    n_sil = seg_cfg["silhouette_sample"]
    rows, labels, fitted = [], {}, {}

    for k in seg_cfg["k_range"]:
        for s, vae in enumerate(vaes):
            z_fit, z_test = vae.latent(fit_df), vae.latent(test)
            km = KMeans(k, n_init=5, random_state=s).fit(z_fit)
            gm = GaussianMixture(k, covariance_type="full", n_init=2, random_state=s).fit(z_fit)
            rkm = KMeans(k, n_init=5, random_state=s).fit(raw_fit)
            for method, model, xt in [("vae_kmeans", km, z_test), ("vae_gmm", gm, z_test), ("raw_kmeans", rkm, raw_test)]:
                lab = model.predict(xt)
                labels[(method, k, s)] = lab
                rows.append({"method": method, "k": k, "seed": s, **segment_stats(lab, test),
                             "silhouette": _silhouette(xt, lab, n_sil, s)})
            if s == 0:
                fitted[("vae_kmeans", k)] = km
        # GLM-predicted-frequency quantiles: k groups of equal training exposure (k=10 -> deciles).
        cuts = np.quantile(glm_fit, np.linspace(0, 1, k + 1)[1:-1])
        lab = np.searchsorted(cuts, glm_test)
        labels[("glm_quantile", k, 0)] = lab
        rows.append({"method": "glm_quantile", "k": k, "seed": 0, **segment_stats(lab, test),
                     "silhouette": _silhouette(glm_test[:, None], lab, n_sil, 0)})

    table = pd.DataFrame(rows)
    # Stability: mean pairwise ARI between seeds (VAE re-trained / k-means re-initialised).
    ari = []
    for (method, k), grp in table.groupby(["method", "k"]):
        seeds = sorted(grp.seed.unique())
        pairs = [adjusted_rand_score(labels[(method, k, a)], labels[(method, k, b)])
                 for i, a in enumerate(seeds) for b in seeds[i + 1:]]
        ari.append({"method": method, "k": k, "ari_across_seeds": float(np.mean(pairs)) if pairs else 1.0})
    table = table.merge(pd.DataFrame(ari), on=["method", "k"])
    return table, {"kmeans": fitted, "labels": labels}


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-segment exposure share, observed frequency and typical rating factors."""
    d = df.assign(segment=labels)
    agg = d.groupby("segment").agg(
        policies=("IDpol", "size"), exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"),
        DrivAge=("DrivAge", "median"), BonusMalus=("BonusMalus", "median"), VehAge=("VehAge", "median"),
        VehPower=("VehPower", "median"), Density=("Density", "median"),
        diesel_share=("VehGasDiesel", "mean"),
        top_area=("Area", lambda s: s.mode().iat[0]), top_brand=("VehBrand", lambda s: s.mode().iat[0]),
        top_region=("Region", lambda s: s.mode().iat[0]),
    )
    agg["exposure_share"] = agg.exposure / agg.exposure.sum()
    agg["frequency"] = agg.claims / agg.exposure
    return agg.sort_values("frequency")


def anomaly_report(test: pd.DataFrame, scores: np.ndarray, threshold: float, top: float = 0.01) -> dict:
    """Top-`top` share of test policies by reconstruction error vs the rest."""
    cut = np.quantile(scores, 1 - top)
    flag = scores >= cut
    d = test.assign(score=scores, flagged=flag)
    cont = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density", "Exposure"]
    compare = d.groupby("flagged")[cont].median().T.rename(columns={False: "rest_median", True: "top1pct_median"})
    cats = {}
    for c in ["VehBrand", "Region", "Area", "VehGas"]:
        share = pd.crosstab(d[c], d.flagged, normalize="columns")
        share["lift"] = share[True] / share[False]
        cats[c] = share.sort_values("lift", ascending=False).head(4)
    freq = d.groupby("flagged").apply(lambda g: g.ClaimNb.sum() / g.Exposure.sum(), include_groups=False)
    examples = d.sort_values("score", ascending=False).head(10)[
        ["IDpol", "DrivAge", "VehAge", "VehPower", "BonusMalus", "VehBrand", "VehGas", "Area", "Density",
         "Region", "Exposure", "ClaimNb", "score"]]
    return {"cutoff": float(cut), "train_threshold": threshold, "share_above_train_threshold": float((scores > threshold).mean()),
            "continuous": compare, "categorical": cats, "frequency": freq, "examples": examples}
