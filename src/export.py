"""Copy the models chosen on *validation* into a self-contained serving bundle.

    python -m src.export          ->  serve/model_bundle/
"""
from __future__ import annotations

import argparse
import shutil

from src.utils import ROOT, load_config, read_json, write_json

NN_CANDIDATES = ["freq_ffnn_emb", "freq_cann"]


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    art = ROOT / cfg["artifacts_dir"]
    out = ROOT / "serve" / "model_bundle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    # Pick the frequency NN by validation deviance (the test set plays no part in this choice).
    val = {r: read_json(art / r / "summary.json")["val_metric"]["mean"] for r in NN_CANDIDATES}
    freq_nn = min(val, key=val.get)
    for run in ["freq_glm", freq_nn, "sev_glm", "sev_nn"]:
        shutil.copytree(art / run, out / run, ignore=shutil.ignore_patterns("history.json"))
    for run in ["ae", "vae"]:
        shutil.copytree(art / run / "seed0", out / run / "seed0", ignore=shutil.ignore_patterns("history.json"))
    shutil.copy(art / "segmenter.json", out / "segmenter.json")
    report = read_json(ROOT / cfg["data"]["processed_dir"] / "cleaning_report.json")
    write_json({"freq_nn": freq_nn, "freq_nn_val_deviance": val, "large_loss_load": report["severity_cap"]["large_loss_load"],
                "severity_cap": report["severity_cap"]["value"]}, out / "manifest.json")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"bundle -> {out} ({size:.2f} MB), frequency NN = {freq_nn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
