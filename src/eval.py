"""Final evaluation on the held-out test split. This is the only module that loads it.

    python -m src.eval

Writes reports/tables/*.md|csv, reports/figures/*.png, reports/results.json and
reports/segmentation_report.md, and fits the serving segmenter (artifacts/segmenter.json).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm  # noqa: F401  must load before torch (separate libomp copies on macOS)
import numpy as np
import pandas as pd
import torch

from src.data.load import load_claims, load_policies
from src.data.schema import CATEGORIES
from src.metrics import (balance_ratio, gamma_deviance, lorenz_curve, poisson_deviance,
                         weighted_quantile_bins)
from src.models.gbm import LGBMPoisson
from src.models.glm import GammaGLM, HomogeneousFrequency, PoissonGLM
from src.models.neural import NeuralFrequency, NeuralSeverity, TabularAE
from src.plotting import COLORS, INK2, SEQ, plt
from src import segmentation as seg
from src.utils import ROOT, load_config, read_json, write_json

REPORTS = ROOT / "reports"
FIG, TAB = REPORTS / "figures", REPORTS / "tables"
LOADERS = {("frequency", "homogeneous"): HomogeneousFrequency, ("frequency", "glm"): PoissonGLM,
           ("frequency", "lgbm"): LGBMPoisson, ("frequency", "ffnn"): NeuralFrequency,
           ("frequency", "cann"): NeuralFrequency, ("severity", "glm"): GammaGLM,
           ("severity", "ffnn"): NeuralSeverity, ("autoencoder", "ae"): TabularAE, ("vae", "vae"): TabularAE}
FREQ_RUNS = {"Homogeneous": "freq_homogeneous", "GLM": "freq_glm", "LightGBM": "freq_lgbm",
             "FFNN (one-hot)": "freq_ffnn_onehot", "FFNN (embeddings)": "freq_ffnn_emb", "CANN": "freq_cann"}
# I7 sensitivity: the same models trained/tested on a plain random split, where ~33% of test
# policies have an identical-profile twin in train.
RANDOM_RUNS = {"Homogeneous": "freq_homogeneous_random", "GLM": "freq_glm_random", "LightGBM": "freq_lgbm_random",
               "FFNN (embeddings)": "freq_ffnn_emb_random", "CANN": "freq_cann_random"}
N_BOOT = 1000


# LightGBM and PyTorch load separate libomp copies on macOS; with both in one process,
# torch must stay single-threaded (inference here is cheap, so this costs little).
torch.set_num_threads(1)


def load_run(cfg: dict, name: str) -> list:
    s = read_json(ROOT / cfg["artifacts_dir"] / name / "summary.json")
    loader = LOADERS[(s["config"]["task"], s["config"]["model"])]
    return [loader.load(ROOT / cfg["artifacts_dir"] / name / f"seed{k}") for k in s["config"]["seeds"]]


def md_table(df: pd.DataFrame, path: Path, floatfmt: str = ".4f") -> str:
    text = df.to_markdown(index=False, floatfmt=floatfmt)
    path.with_suffix(".md").write_text(text + "\n")
    df.to_csv(path.with_suffix(".csv"), index=False)
    return text


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """95% CI of mean(loss_a - loss_b) by resampling policies."""
    d = loss_a - loss_b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def unit_poisson(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))


def unit_gamma(y, mu):
    return 2 * ((y - mu) / mu - np.log(y / mu))


# ------------------------------------------------------------------ I1 / I2 frequency
def frequency_section(cfg, test, models, runs=FREQ_RUNS) -> tuple[pd.DataFrame, dict]:
    y = test.ClaimNb.to_numpy(float)
    preds, rows = {}, {}
    for label, ms in models.items():
        per_seed = [m.predict(test) for m in ms]
        preds[label] = np.mean(per_seed, axis=0)
        devs = np.array([poisson_deviance(y, p) for p in per_seed])
        summ = read_json(ROOT / cfg["artifacts_dir"] / runs[label] / "summary.json")
        rows[label] = {"model": label, "seeds": len(ms),
                       "val_dev": summ["val_metric"]["mean"] * 100,
                       "test_dev_mean": devs.mean() * 100, "test_dev_std": devs.std(ddof=1) * 100 if len(ms) > 1 else 0.0,
                       "test_dev_ensemble": poisson_deviance(y, preds[label]) * 100,
                       "balance": balance_ratio(y, preds[label])}
    glm_ens = rows["GLM"]["test_dev_ensemble"]
    glm_unit = unit_poisson(y, preds["GLM"])
    for label, r in rows.items():
        r["improvement_vs_glm_pct"] = 100 * (glm_ens - r["test_dev_ensemble"]) / glm_ens
        mean, lo, hi = paired_bootstrap(glm_unit, unit_poisson(y, preds[label]))
        r["delta_vs_glm_ci95"] = f"[{lo*100:.3f}, {hi*100:.3f}]"
    df = pd.DataFrame(rows.values())
    return df, preds


def freq_table_md(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Model": df.model, "Seeds": df.seeds,
        "Val deviance": df.val_dev.map("{:.3f}".format),
        "Test deviance (mean ± std over seeds)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(df.test_dev_mean, df.test_dev_std)],
        "Test deviance (seed ensemble)": df.test_dev_ensemble.map("{:.3f}".format),
        "Δ vs GLM (%)": df.improvement_vs_glm_pct.map("{:+.2f}".format),
        "GLM − model, 95% CI (×10⁻²)": df.delta_vs_glm_ci95,
        "Σpred / Σactual": df.balance.map("{:.4f}".format),
    })


# ------------------------------------------------------------------ I4 severity
def severity_section(cfg, claims_train, claims_test, glm, nns):
    y = claims_test.ClaimAmount.to_numpy(float)
    p_glm = glm.predict(claims_test)
    per_seed = [m.predict(claims_test) for m in nns]
    p_nn = np.mean(per_seed, axis=0)
    p_const = np.full_like(y, claims_train.ClaimAmount.mean())
    nn_devs = np.array([gamma_deviance(y, p) for p in per_seed])
    rows = []
    for label, p, devs in [("Constant mean", p_const, None), ("Gamma GLM", p_glm, None), ("Gamma NN", p_nn, nn_devs)]:
        mean, lo, hi = paired_bootstrap(unit_gamma(y, p_glm), unit_gamma(y, p))
        rows.append({"Model": label, "Seeds": len(nns) if devs is not None else 1,
                     "Test Gamma deviance (mean ± std)": f"{(devs.mean() if devs is not None else gamma_deviance(y, p)):.4f} ± {(devs.std(ddof=1) if devs is not None else 0):.4f}",
                     "Test Gamma deviance (ensemble)": f"{gamma_deviance(y, p):.4f}",
                     "GLM − model, 95% CI": f"[{lo:.4f}, {hi:.4f}]",
                     "Σpred / Σactual": f"{p.sum() / y.sum():.4f}"})
    return pd.DataFrame(rows), {"glm": gamma_deviance(y, p_glm), "nn": gamma_deviance(y, p_nn),
                                "const": gamma_deviance(y, p_const), "nn_seed_std": float(nn_devs.std(ddof=1)),
                                "n_test_claims": int(len(y))}


# ------------------------------------------------------------------ ranking + calibration
def lorenz_section(test, freq_preds, sev_preds, load):
    """Pure premium = E[N] x E[severity]; Lorenz / Gini vs capped actual losses."""
    mask = test.HasSeverity.to_numpy()
    t = test[mask]
    actual, expo = t.ClaimTotal.to_numpy(float), t.Exposure.to_numpy(float)
    combos = {"GLM": ("GLM", "glm"), "LightGBM": ("LightGBM", "glm"), "CANN": ("CANN", "nn"),
              "FFNN (embeddings)": ("FFNN (embeddings)", "nn")}
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    out = {}
    rng = np.random.default_rng(0)
    for label, (f, s) in combos.items():
        pp = freq_preds[f][mask] * sev_preds[s][mask] * load
        rate = pp / expo  # rank by premium per policy-year, not by exposure-scaled premium
        x, yv, g = lorenz_curve(rate, actual, expo)
        boots = []
        for _ in range(200):
            i = rng.integers(0, len(pp), len(pp))
            boots.append(lorenz_curve(rate[i], actual[i], expo[i])[2])
        out[label] = {"gini": g, "gini_boot_std": float(np.std(boots)),
                      "pure_premium_balance": float(pp.sum() / (actual.sum() * load))}
        step = max(1, len(x) // 2000)
        ax.plot(x[::step], yv[::step], color=COLORS[label], label=f"{label}  Gini {g:.3f}")
    ax.set(xlabel="Cumulative share of exposure (sorted by predicted pure premium)",
           ylabel="Cumulative share of actual losses", title="Ordered Lorenz curve, test set")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "lorenz_pure_premium.png")
    plt.close(fig)
    return out


def double_lift(actual, p_model, p_glm, expo, label, fname, what):
    """Deciles (equal exposure) of the model/GLM prediction ratio; each prediction is rescaled
    so its total equals the actual total, which isolates ranking from overall level."""
    p_model = p_model * actual.sum() / p_model.sum()
    p_glm = p_glm * actual.sum() / p_glm.sum()
    b = weighted_quantile_bins(p_model / p_glm, expo, 10)
    d = pd.DataFrame({"b": b, "y": actual, "m": p_model, "g": p_glm, "e": expo}).groupby("b").sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    x = d.index + 1
    ax.plot(x, d.y / d.e, color=COLORS["Actual"], marker="o", ms=5, label="Actual")
    ax.plot(x, d.m / d.e, color=COLORS[label], marker="o", ms=5, label=label)
    ax.plot(x, d.g / d.e, color=COLORS["GLM"], marker="o", ms=5, label="GLM")
    ax.set(xlabel=f"Decile of {label} / GLM predicted ratio", ylabel=f"{what} per policy-year",
           title=f"Double-lift, {what.split(' (')[0].lower()}: {label} vs GLM (test)", xticks=list(x))
    ax.legend()
    fig.savefig(FIG / fname)
    plt.close(fig)
    d["actual"], d["model"], d["glm"] = d.y / d.e, d.m / d.e, d.g / d.e
    d["model_abs_err"], d["glm_abs_err"] = (d.model - d.actual).abs(), (d.glm - d.actual).abs()
    return d[["actual", "model", "glm", "model_abs_err", "glm_abs_err"]].reset_index(names="decile")


def calibration_plot(test, preds: dict):
    expo, y = test.Exposure.to_numpy(float), test.ClaimNb.to_numpy(float)
    labels = ["GLM", "LightGBM", "CANN"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, label in zip(axes, labels):
        p = preds[label]
        b = weighted_quantile_bins(p / expo, expo, 10)
        d = pd.DataFrame({"b": b, "y": y, "p": p, "e": expo}).groupby("b").sum()
        ax.plot(d.p / d.e, d.y / d.e, color=COLORS[label], marker="o", ms=5)
        lim = [0, max((d.p / d.e).max(), (d.y / d.e).max()) * 1.05]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.set(title=label, xlabel="Predicted frequency (decile mean)")
    axes[0].set_ylabel("Observed frequency")
    fig.suptitle("Actual vs predicted by predicted-frequency decile (test)", fontweight="bold")
    fig.savefig(FIG / "calibration_deciles.png")
    plt.close(fig)


# ------------------------------------------------------------------ interpretability
def pdp_section(train, models: dict):
    sample = train.sample(10_000, random_state=0).assign(Exposure=1.0)
    grids = {"DrivAge": np.arange(18, 91), "BonusMalus": np.arange(50, 151, 2), "VehAge": np.arange(0, 21)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (feat, grid) in zip(axes, grids.items()):
        for label in ["GLM", "LightGBM", "FFNN (embeddings)", "CANN"]:
            ms = models[label]
            curve = [np.mean([m.predict(sample.assign(**{feat: v})).mean() for m in ms]) for v in grid]
            ax.plot(grid, curve, color=COLORS[label], label=label, lw=2 if label in ("GLM", "CANN") else 1.5)
        ax.set(title=feat, xlabel=feat)
    axes[0].set_ylabel("Mean predicted frequency")
    axes[0].legend()
    fig.suptitle("Partial dependence (10K training policies, exposure = 1; NNs averaged over seeds)", fontweight="bold")
    fig.savefig(FIG / "partial_dependence.png")
    plt.close(fig)


# ------------------------------------------------------------------ I3 embeddings
def embedding_section(train, ffnns) -> pd.DataFrame:
    freq = {c: train.groupby(c).apply(lambda g: g.ClaimNb.sum() / g.Exposure.sum(), include_groups=False)
            for c in ["VehBrand", "Region"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for ax, (ci, col) in zip(axes, [(1, "VehBrand"), (2, "Region")]):
        levels = CATEGORIES[col]
        f = freq[col].reindex(levels).to_numpy()
        for s, m in enumerate(ffnns):
            w = m.net.cat.emb[ci].weight.detach().numpy()
            # Does embedding distance track risk difference? Spearman over all level pairs.
            iu = np.triu_indices(len(levels), 1)
            dist = np.linalg.norm(w[:, None] - w[None], axis=2)[iu]
            dfreq = np.abs(np.log(f)[:, None] - np.log(f)[None])[iu]
            rho = pd.Series(dist).corr(pd.Series(dfreq), method="spearman")
            rows.append({"variable": col, "seed": s, "spearman_dist_vs_logfreq_gap": rho})
            if s == 0:
                sc = ax.scatter(w[:, 0], w[:, 1], c=f, cmap="Blues", s=70, edgecolor="#52514e", lw=0.6, vmin=f.min() * 0.8)
                for i, lv in enumerate(levels):
                    ax.annotate(lv, w[i], xytext=(4, 3), textcoords="offset points", fontsize=7.5, color="#0b0b0b")
                plt.colorbar(sc, ax=ax, label="Observed train frequency")
        ax.margins(x=0.25, y=0.1)  # room for direct labels at the edges
        ax.set(title=f"{col} embedding (FFNN, seed 0)", xlabel="dim 1", ylabel="dim 2")
    fig.savefig(FIG / "embeddings.png")
    plt.close(fig)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ failure analysis
def failure_section(test, p_nn, p_glm, label):
    y, e = test.ClaimNb.to_numpy(float), test.Exposure.to_numpy(float)
    d = test.assign(pred_nn=p_nn, pred_glm=p_glm, dev_nn=unit_poisson(y, p_nn), dev_glm=unit_poisson(y, p_glm))
    groups = {
        "DrivAge": pd.cut(d.DrivAge, [17, 20, 25, 30, 50, 70, 90]),
        "BonusMalus": pd.cut(d.BonusMalus, [49, 50, 75, 100, 125, 150]),
        "Exposure": pd.cut(d.Exposure, [0, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0]),
        "Region": d.Region, "VehBrand": d.VehBrand,
    }
    rows = []
    for name, g in groups.items():
        agg = d.groupby(g, observed=True).agg(policies=("IDpol", "size"), exposure=("Exposure", "sum"),
                                              claims=("ClaimNb", "sum"), nn=("pred_nn", "sum"), glm=("pred_glm", "sum"))
        for lv, r in agg.iterrows():
            rows.append({"factor": name, "level": str(lv), "policies": int(r.policies), "claims": int(r.claims),
                         f"A/E {label}": r.claims / r.nn, "A/E GLM": r.claims / r.glm})
    ae = pd.DataFrame(rows)
    by_claims = d.groupby(d.ClaimNb.clip(upper=2)).agg(policies=("IDpol", "size"), dev_share_nn=("dev_nn", "sum"),
                                                        mean_pred_nn=("pred_nn", "mean"))
    by_claims["dev_share_nn"] /= d.dev_nn.sum()
    worst = d.sort_values("dev_nn", ascending=False).head(8)[
        ["IDpol", "DrivAge", "BonusMalus", "VehAge", "VehBrand", "Region", "Exposure", "ClaimNb", "pred_nn", "pred_glm"]]
    return ae, by_claims.reset_index(), worst


# ------------------------------------------------------------------ main
def main(config_path: str) -> None:
    cfg = load_config(config_path)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    pol = load_policies(cfg, ("train", "val", "test"))
    train, test = pol[pol.split == "train"], pol[pol.split == "test"].reset_index(drop=True)
    claims = load_claims(cfg, ("train", "test"))
    c_train, c_test = claims[claims.split == "train"], claims[claims.split == "test"]
    report = read_json(ROOT / cfg["data"]["processed_dir"] / "cleaning_report.json")
    results: dict = {"data": report}

    print("I1/I2 frequency")
    fmodels = {label: load_run(cfg, run) for label, run in FREQ_RUNS.items()}
    fdf, fpreds = frequency_section(cfg, test, fmodels)
    md_table(freq_table_md(fdf), TAB / "frequency_results")
    results["frequency"] = fdf.to_dict(orient="records")

    print("I7 random-split sensitivity")
    rcfg = {**cfg, "split": {**cfg["split"], "column": "split"}}
    rpol = load_policies(rcfg, ("test",))
    rdf, _ = frequency_section(cfg, rpol, {l: load_run(cfg, r) for l, r in RANDOM_RUNS.items()}, RANDOM_RUNS)
    md_table(freq_table_md(rdf), TAB / "frequency_results_random_split")
    results["frequency_random_split"] = rdf.to_dict(orient="records")
    search = pd.read_csv(ROOT / cfg["artifacts_dir"] / "freq_lgbm" / "search.csv")
    search["val_poisson_deviance"] *= 100
    md_table(search.rename(columns={"val_poisson_deviance": "val deviance (x1e-2)"}), TAB / "lgbm_search", floatfmt=".4g")

    print("I4 severity")
    sev_glm, sev_nns = load_run(cfg, "sev_glm")[0], load_run(cfg, "sev_nn")
    sdf, sres = severity_section(cfg, c_train, c_test, sev_glm, sev_nns)
    md_table(sdf, TAB / "severity_results")
    results["severity"] = sres

    print("ranking / calibration")
    sev_pol = {"glm": sev_glm.predict(test), "nn": np.mean([m.predict(test) for m in sev_nns], axis=0)}
    load = report["severity_cap"]["large_loss_load"]
    results["lorenz"] = lorenz_section(test, fpreds, sev_pol, load)
    best_nn = min(["FFNN (embeddings)", "CANN"], key=lambda k: fdf.set_index("model").loc[k, "val_dev"])
    results["best_nn_by_val"] = best_nn
    expo = test.Exposure.to_numpy(float)
    md_table(double_lift(test.ClaimNb.to_numpy(float), fpreds[best_nn], fpreds["GLM"], expo, best_nn,
                         "double_lift.png", "Claim frequency"), TAB / "double_lift")
    # Pure premium: frequency NN x severity NN vs GLM x Gamma GLM, on policies with complete severity data.
    m = test.HasSeverity.to_numpy()
    pp_nn = fpreds[best_nn][m] * sev_pol["nn"][m] * load
    pp_glm = fpreds["GLM"][m] * sev_pol["glm"][m] * load
    md_table(double_lift(test.ClaimTotal.to_numpy(float)[m] * load, pp_nn, pp_glm, expo[m], best_nn,
                         "double_lift_pure_premium.png", "Pure premium (EUR)"), TAB / "double_lift_pure_premium",
             floatfmt=".2f")
    calibration_plot(test, fpreds)

    print("PDP / embeddings")
    pdp_section(train, fmodels)
    emb = embedding_section(train, fmodels["FFNN (embeddings)"])
    md_table(emb.groupby("variable").spearman_dist_vs_logfreq_gap.agg(["mean", "std"]).reset_index(), TAB / "embedding_spearman")
    results["embedding_spearman"] = emb.groupby("variable").spearman_dist_vs_logfreq_gap.mean().to_dict()

    print("failure analysis")
    ae, by_claims, worst = failure_section(test, fpreds[best_nn], fpreds["GLM"], best_nn)
    md_table(ae, TAB / "failure_actual_vs_expected", floatfmt=".3f")
    md_table(by_claims, TAB / "failure_deviance_by_claimnb", floatfmt=".4f")
    md_table(worst, TAB / "failure_worst_policies", floatfmt=".3f")

    print("I5 segmentation")
    vae_cfg = load_config(ROOT / "configs" / "vae.yaml")
    vaes = load_run(cfg, "vae")
    glm = fmodels["GLM"][0]
    seg_table, fitted = seg.run_segmentation(train, test, vaes, glm, vae_cfg["segmentation"])
    agg = (seg_table.groupby(["method", "k"])
           .agg(max_min_ratio=("max_min_ratio", "mean"), max_min_std=("max_min_ratio", "std"),
                freq_cv=("freq_cv", "mean"), deviance_reduction_pct=("deviance_reduction_pct", "mean"),
                silhouette=("silhouette", "mean"), ari=("ari_across_seeds", "first"),
                min_exposure_share=("min_exposure_share", "mean")).reset_index())
    md_table(agg, TAB / "segmentation", floatfmt=".3f")
    results["segmentation"] = agg.to_dict(orient="records")
    plot_segmentation(agg)
    k = vae_cfg["segmentation"]["serve_k"]
    km = fitted["kmeans"][("vae_kmeans", k)]
    prof = seg.profile_segments(test, fitted["labels"][("vae_kmeans", k, 0)])
    glm_prof = seg.profile_segments(test, fitted["labels"][("glm_quantile", k, 0)])
    # Serving segmenter: VAE seed-0 latent -> nearest k-means centre, with train frequencies.
    tr_lab = km.predict(vaes[0].latent(train))
    seg_freq = (pd.DataFrame({"s": tr_lab, "n": train.ClaimNb.values, "e": train.Exposure.values})
                .groupby("s").sum().pipe(lambda g: g.n / g.e))
    write_json({"vae_run": "vae/seed0", "k": k, "centers": km.cluster_centers_.tolist(),
                "train_frequency": seg_freq.reindex(range(k)).tolist()}, ROOT / cfg["artifacts_dir"] / "segmenter.json")

    print("I6 anomalies")
    aes = load_run(cfg, "ae")
    scores = [m.score(test) for m in aes]
    tops = [set(np.argsort(-s)[: len(s) // 100]) for s in scores]
    jacc = np.mean([len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]])
    anom = seg.anomaly_report(test, scores[0], aes[0].threshold)
    results["anomaly"] = {"top1pct_jaccard_across_seeds": float(jacc), "cutoff": anom["cutoff"],
                          "frequency_flagged": float(anom["frequency"][True]),
                          "frequency_rest": float(anom["frequency"][False]),
                          "share_above_train_threshold": anom["share_above_train_threshold"]}
    write_segmentation_report(agg, prof, glm_prof, k, anom, jacc)

    results["latency"] = read_json(TAB / "latency.json") if (TAB / "latency.json").exists() else None
    write_json(results, REPORTS / "results.json")
    print(freq_table_md(fdf).to_markdown(index=False))
    print(sdf.to_markdown(index=False))
    print(pd.DataFrame(results["lorenz"]).T)


def plot_segmentation(agg: pd.DataFrame):
    names = {"glm_quantile": ("GLM-frequency quantiles", COLORS["GLM"]), "vae_kmeans": ("VAE latent + k-means", COLORS["CANN"]),
             "vae_gmm": ("VAE latent + GMM", COLORS["LightGBM"]), "raw_kmeans": ("Raw features + k-means", COLORS["FFNN (embeddings)"])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m, (lbl, col) in names.items():
        g = agg[agg.method == m]
        axes[0].plot(g.k, g.max_min_ratio, color=col, marker="o", ms=5, label=lbl)
        axes[1].plot(g.k, g.deviance_reduction_pct, color=col, marker="o", ms=5, label=lbl)
    axes[0].set(title="Max / min segment claim frequency (test)", xlabel="number of segments k", ylabel="ratio")
    axes[1].set(title="Poisson deviance reduction vs one rate (test)", xlabel="number of segments k", ylabel="%")
    axes[0].legend()
    fig.savefig(FIG / "segmentation_spread.png")
    plt.close(fig)


def write_segmentation_report(agg, prof, glm_prof, k, anom, jacc):
    fmt = lambda df: df.to_markdown(floatfmt=".3f")  # noqa: E731
    sel = agg[agg.k == k].drop(columns="k")
    text = f"""# Segmentation report (I5) and anomaly inspection (I6)

All numbers are measured on the held-out **test** split; clusterers were fitted on a
100K-policy sample of the training split. VAE/raw-feature metrics are means over 3 seeds.

![segmentation spread](figures/segmentation_spread.png)

## Method comparison at k = {k}

{sel.to_markdown(index=False, floatfmt=".3f")}

Full sweep over k = 4..10: [tables/segmentation.md](tables/segmentation.md).
`ari` is the mean pairwise adjusted Rand index across seeds (GLM quantiles are deterministic, so 1 by construction).
`deviance_reduction_pct` is the Poisson-deviance reduction from pricing each policy at its segment's observed rate.

## VAE k-means segments (k = {k}, seed 0), sorted by observed frequency

{fmt(prof)}

## GLM-frequency segments (k = {k}) for comparison

{fmt(glm_prof)}

## Autoencoder anomalies: top 1% of test policies by reconstruction error

* Cut-off score {anom['cutoff']:.2f}; {100*anom['share_above_train_threshold']:.2f}% of test policies exceed the training 99th-percentile threshold used by the API.
* Overlap of the top-1% sets across 3 autoencoder seeds (mean Jaccard): {jacc:.2f}.
* Observed claim frequency: flagged {anom['frequency'][True]:.4f} vs rest {anom['frequency'][False]:.4f}.

### Continuous features (medians)

{fmt(anom['continuous'])}

### Over-represented categories among flagged policies (share and lift vs rest)

""" + "\n\n".join(f"**{c}**\n\n{fmt(t)}" for c, t in anom["categorical"].items()) + f"""

### Ten most unusual test policies

{anom['examples'].to_markdown(index=False, floatfmt='.2f')}
"""
    (REPORTS / "segmentation_report.md").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
