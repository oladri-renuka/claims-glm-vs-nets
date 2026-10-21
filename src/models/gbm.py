"""LightGBM Poisson frequency model with log(Exposure) as an offset (init_score)."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.schema import AREAS, REGIONS, VEH_BRANDS, VEH_GAS
from src.utils import read_json, write_json

CATS = {"Area": AREAS, "VehBrand": VEH_BRANDS, "VehGas": VEH_GAS, "Region": REGIONS}
NUM = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]


def lgb_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "VehPower": df.VehPower.astype(float), "VehAge": np.minimum(df.VehAge, 20).astype(float),
        "DrivAge": np.minimum(df.DrivAge, 90).astype(float),
        "BonusMalus": np.minimum(df.BonusMalus, 150).astype(float),
        "LogDensity": np.log(df.Density.astype(float)),
    })
    for c, levels in CATS.items():
        out[c] = pd.Categorical(df[c], categories=levels)
    return out


class LGBMPoisson:
    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params, self.seed, self.booster = dict(params or {}), seed, None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "LGBMPoisson":
        p = dict(self.params)
        n_est, es = p.pop("n_estimators"), p.pop("early_stopping_rounds")
        p.update(seed=self.seed, verbose=-1, num_threads=p.pop("num_threads", 4), deterministic=True,
                 force_row_wise=True)
        dtr = lgb.Dataset(lgb_frame(train), train.ClaimNb, init_score=np.log(train.Exposure))
        dva = lgb.Dataset(lgb_frame(val), val.ClaimNb, init_score=np.log(val.Exposure), reference=dtr)
        self.history: dict = {}
        self.booster = lgb.train(p, dtr, num_boost_round=n_est, valid_sets=[dva], valid_names=["val"],
                                 callbacks=[lgb.early_stopping(es, verbose=False),
                                            lgb.record_evaluation(self.history)])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        bi = self.booster.best_iteration
        raw = self.booster.predict(lgb_frame(df), raw_score=True, num_iteration=bi if bi and bi > 0 else None)
        return np.exp(raw) * df["Exposure"].to_numpy(float)

    def save(self, d: Path) -> None:
        self.booster.save_model(str(d / "model.txt"), num_iteration=self.booster.best_iteration)
        write_json({"params": self.params, "seed": self.seed, "best_iteration": self.booster.best_iteration},
                   d / "meta.json")

    @classmethod
    def load(cls, d: Path) -> "LGBMPoisson":
        meta = read_json(d / "meta.json")
        m = cls(meta["params"], meta["seed"])
        m.booster = lgb.Booster(model_file=str(d / "model.txt"))
        return m
