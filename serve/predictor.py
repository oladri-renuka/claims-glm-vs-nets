"""Loads the serving bundle and prices policies (single or batched)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.glm import GammaGLM, PoissonGLM
from src.models.neural import NeuralFrequency, NeuralSeverity, TabularAE
from src.utils import read_json

BUNDLE = Path(__file__).resolve().parent / "model_bundle"


def _seeds(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("seed"))


class Predictor:
    def __init__(self, bundle: Path = BUNDLE):
        torch.set_num_threads(1)
        self.manifest = read_json(bundle / "manifest.json")
        self.freq_glm = PoissonGLM.load(bundle / "freq_glm" / "seed0")
        self.freq_nn = [NeuralFrequency.load(p) for p in _seeds(bundle / self.manifest["freq_nn"])]
        self.sev_glm = GammaGLM.load(bundle / "sev_glm" / "seed0")
        self.sev_nn = [NeuralSeverity.load(p) for p in _seeds(bundle / "sev_nn")]
        self.ae = TabularAE.load(bundle / "ae" / "seed0")
        self.vae = TabularAE.load(bundle / "vae" / "seed0")
        seg = read_json(bundle / "segmenter.json")
        self.centers, self.seg_freq = np.asarray(seg["centers"]), np.asarray(seg["train_frequency"])
        self.load = float(self.manifest["large_loss_load"])
        self.bundle = bundle

    def quote_frame(self, df: pd.DataFrame, model: str = "nn") -> pd.DataFrame:
        annual = df.assign(Exposure=1.0)
        if model == "nn":
            freq = np.mean([m.predict(annual) for m in self.freq_nn], axis=0)
            sev = np.mean([m.predict(annual) for m in self.sev_nn], axis=0)
        elif model == "glm":
            freq, sev = self.freq_glm.predict(annual), self.sev_glm.predict(annual)
        else:
            raise ValueError("model must be 'nn' or 'glm'")
        sev = sev * self.load
        z = self.vae.latent(df)
        segment = np.argmin(((z[:, None, :] - self.centers[None]) ** 2).sum(-1), axis=1)
        score = self.ae.score(df)
        expo = df["Exposure"].to_numpy(float)
        return pd.DataFrame({
            "model": model, "expected_frequency": freq, "expected_claims": freq * expo,
            "expected_severity": sev, "pure_premium": freq * sev, "pure_premium_for_exposure": freq * sev * expo,
            "risk_segment": segment, "segment_frequency": self.seg_freq[segment],
            "anomaly": score > self.ae.threshold, "anomaly_score": score,
        })

    def quote(self, policy: dict, model: str = "nn") -> dict:
        row = self.quote_frame(pd.DataFrame([policy]), model).iloc[0].to_dict()
        row["risk_segment"], row["anomaly"] = int(row["risk_segment"]), bool(row["anomaly"])
        return row

    def size_mb(self) -> float:
        return sum(p.stat().st_size for p in self.bundle.rglob("*") if p.is_file()) / 1e6
