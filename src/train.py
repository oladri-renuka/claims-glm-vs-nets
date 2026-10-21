"""Train one experiment (all its seeds) from a config file.

    python -m src.train --config configs/freq_cann.yaml

Uses only the train and val splits. Writes artifacts/<name>/seed<k>/ (model + val metrics)
and artifacts/<name>/summary.json; logs params and metrics to MLflow.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.data.load import load_claims, load_policies
from src.metrics import gamma_deviance, poisson_deviance
from src.models.glm import GammaGLM, HomogeneousFrequency, PoissonGLM
from src.utils import ROOT, Tracker, load_config, nn_params, run_dir, set_seed, write_json


def build_model(cfg: dict, seed: int):
    task, kind = cfg["task"], cfg["model"]
    if not (task == "frequency" and kind == "lgbm"):
        # torch is imported only by non-LightGBM runs (see the libomp note below).
        from src.models.neural import NeuralFrequency, NeuralSeverity, TabularAE  # noqa: F401
    if task == "frequency":
        if kind == "homogeneous":
            return HomogeneousFrequency()
        if kind == "glm":
            return PoissonGLM()
        if kind == "lgbm":
            # Imported lazily: LightGBM and PyTorch ship separate libomp copies on macOS and
            # multi-threaded torch segfaults if both are loaded in one process.
            from src.models.gbm import LGBMPoisson

            return LGBMPoisson(cfg["params"], seed)
        if kind in ("ffnn", "cann"):
            glm = None
            if kind == "cann":
                glm = PoissonGLM.load(ROOT / cfg["artifacts_dir"] / cfg["params"]["glm_run"] / "seed0")
            from src.models.neural import NeuralFrequency

            return NeuralFrequency(nn_params(cfg), seed, glm)
    if task == "severity":
        if kind == "glm":
            return GammaGLM()
        from src.models.neural import NeuralSeverity

        return NeuralSeverity(nn_params(cfg), seed)
    if task in ("autoencoder", "vae"):
        from src.models.neural import TabularAE

        return TabularAE(nn_params(cfg), seed, variational=(task == "vae"))
    raise ValueError(f"unknown task/model {task}/{kind}")


def val_metrics(cfg: dict, model, train, val) -> dict:
    if cfg["task"] == "frequency":
        mu = model.predict(val)
        return {"val_poisson_deviance": poisson_deviance(val.ClaimNb, mu),
                "train_poisson_deviance": poisson_deviance(train.ClaimNb, model.predict(train)),
                "val_balance": float(mu.sum() / val.ClaimNb.sum())}
    if cfg["task"] == "severity":
        return {"val_gamma_deviance": gamma_deviance(val.ClaimAmount, model.predict(val)),
                "train_gamma_deviance": gamma_deviance(train.ClaimAmount, model.predict(train))}
    return {"val_loss": float(model.fit_info["best_val"])}


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = load_claims(cfg) if cfg["task"] == "severity" else load_policies(cfg)
    train, val = data[data.split == "train"], data[data.split == "val"]
    tracker = Tracker(cfg)
    summary = {"name": cfg["name"], "config": cfg, "seeds": {}}
    for seed in cfg["seeds"]:
        model = build_model(cfg, seed)  # imports torch if needed; weights are created in fit()
        set_seed(seed)
        with tracker.run(f"{cfg['name']}-seed{seed}", {**cfg.get("params", {}), "seed": seed,
                                                        "task": cfg["task"], "model": cfg["model"]}) as t:
            if type(model).__module__ == "src.models.neural":
                model.fit(train, val, log=t.log_metrics)
            else:
                model.fit(train, val)
            m = val_metrics(cfg, model, train, val)
            info = getattr(model, "fit_info", None)
            if info:
                m.update(epochs=info["epochs"], best_epoch=info["best_epoch"], train_seconds=info["seconds"])
            t.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        d = run_dir(cfg, seed)
        model.save(d)
        if info:
            write_json(info["history"], d / "history.json")
        write_json(m, d / "val_metrics.json")
        summary["seeds"][seed] = m
        print(f"[{cfg['name']}] seed={seed} " + " ".join(f"{k}={v:.5g}" for k, v in m.items()))
    key = next(k for k in summary["seeds"][cfg["seeds"][0]] if k.startswith("val_"))
    vals = np.array([s[key] for s in summary["seeds"].values()])
    summary["val_metric"] = {"name": key, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    write_json(summary, ROOT / cfg["artifacts_dir"] / cfg["name"] / "summary.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
