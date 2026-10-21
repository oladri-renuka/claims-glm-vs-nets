"""Feature engineering for the GLMs (Noll et al. 2018 binning) and the neural nets.

Pure pandas/numpy so the same code runs in training and in the API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import AREAS, CATEGORIES, NN_CATEGORICAL, NN_CONTINUOUS, REGIONS, VEH_BRANDS

# ---- GLM design (Noll, Salzmann & Wuthrich 2018, model "GLM2") --------------------------
# Area: ordinal code (continuous); VehPower: factor, capped at 9; VehAge: 3 bins;
# DrivAge: 7 bins; BonusMalus: continuous (capped at 150); log(Density): continuous;
# VehBrand, VehGas, Region: factors. Reference level = the most populous level.
VEHAGE_BINS = [-np.inf, 0.5, 10.5, np.inf]                  # [0,1) [1,10] (10,inf)
VEHAGE_LABELS = ["0", "1-10", "11+"]
DRIVAGE_BINS = [-np.inf, 20.5, 25.5, 30.5, 40.5, 50.5, 70.5, np.inf]
DRIVAGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
VEHPOWER_LEVELS = ["4", "5", "6", "7", "8", "9"]

GLM_FACTORS = {
    "VehPowerGLM": (VEHPOWER_LEVELS, "6"),
    "VehAgeGLM": (VEHAGE_LABELS, "1-10"),
    "DrivAgeGLM": (DRIVAGE_LABELS, "41-50"),
    "VehBrand": (VEH_BRANDS, "B1"),
    "VehGas": (["Diesel", "Regular"], "Regular"),
    "Region": (REGIONS, "Centre"),
}
GLM_CONTINUOUS = ["AreaGLM", "BonusMalusGLM", "DensityGLM"]


def glm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["AreaGLM"] = df["Area"].map({a: i + 1 for i, a in enumerate(AREAS)}).astype(float)
    out["BonusMalusGLM"] = np.minimum(df["BonusMalus"], 150).astype(float)
    out["DensityGLM"] = np.log(df["Density"].astype(float))
    out["VehPowerGLM"] = np.minimum(df["VehPower"], 9).astype(int).astype(str)
    out["VehAgeGLM"] = pd.cut(df["VehAge"], VEHAGE_BINS, labels=VEHAGE_LABELS).astype(str)
    out["DrivAgeGLM"] = pd.cut(df["DrivAge"], DRIVAGE_BINS, labels=DRIVAGE_LABELS).astype(str)
    for c in ["VehBrand", "VehGas", "Region"]:
        out[c] = df[c].astype(str)
    return out


def glm_columns() -> list[str]:
    cols = ["const", *GLM_CONTINUOUS]
    for f, (levels, ref) in GLM_FACTORS.items():
        cols += [f"{f}[{lv}]" for lv in levels if lv != ref]
    return cols


def glm_design(df: pd.DataFrame) -> np.ndarray:
    """Dense treatment-coded design matrix with a fixed column order (see glm_columns)."""
    g = glm_frame(df)
    blocks = [np.ones((len(g), 1)), g[GLM_CONTINUOUS].to_numpy()]
    for f, (levels, ref) in GLM_FACTORS.items():
        vals = g[f].to_numpy()
        unknown = set(np.unique(vals)) - set(levels)
        if unknown:
            raise ValueError(f"unknown {f} levels: {sorted(unknown)}")
        blocks.append(np.stack([vals == lv for lv in levels if lv != ref], axis=1).astype(float))
    return np.concatenate(blocks, axis=1)


# ---- Neural-net preprocessing ---------------------------------------------------------
class NNPreprocessor:
    """Standardizes continuous inputs (train statistics) and integer-codes categoricals
    against the fixed vocabularies in schema.CATEGORIES."""

    def __init__(self, mean: dict | None = None, std: dict | None = None):
        self.mean, self.std = mean or {}, std or {}

    @staticmethod
    def _continuous(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["VehPower"] = df["VehPower"].astype(float)
        out["VehAge"] = np.minimum(df["VehAge"], 20).astype(float)
        out["DrivAge"] = np.minimum(df["DrivAge"], 90).astype(float)
        out["BonusMalus"] = np.minimum(df["BonusMalus"], 150).astype(float)
        out["LogDensity"] = np.log(df["Density"].astype(float))
        out["VehGasDiesel"] = (df["VehGas"] == "Diesel").astype(float)
        return out[NN_CONTINUOUS]

    def fit(self, df: pd.DataFrame) -> "NNPreprocessor":
        c = self._continuous(df)
        self.mean = c.mean().to_dict()
        self.std = c.std().replace(0, 1).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Plain numpy (not a pandas frame per call): this runs once per model per API request.
        dens = df["Density"].to_numpy(float)
        raw = np.column_stack([
            df["VehPower"].to_numpy(float), np.minimum(df["VehAge"].to_numpy(float), 20),
            np.minimum(df["DrivAge"].to_numpy(float), 90), np.minimum(df["BonusMalus"].to_numpy(float), 150),
            np.log(dens), (df["VehGas"].to_numpy() == "Diesel").astype(float),
        ])
        mean = np.array([self.mean[c] for c in NN_CONTINUOUS])
        std = np.array([self.std[c] for c in NN_CONTINUOUS])
        x_cont = ((raw - mean) / std).astype(np.float32)
        codes = []
        for col in NN_CATEGORICAL:
            c = pd.Categorical(df[col].to_numpy(), categories=CATEGORIES[col]).codes.astype(np.int64)
            if (c < 0).any():
                bad = sorted(set(df[col].to_numpy()[c < 0]))
                raise ValueError(f"unknown {col} values: {bad}")
            codes.append(c)
        return x_cont, np.stack(codes, axis=1)

    @property
    def cardinalities(self) -> list[int]:
        return [len(CATEGORIES[c]) for c in NN_CATEGORICAL]

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "NNPreprocessor":
        return cls(d["mean"], d["std"])
