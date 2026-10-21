"""Homogeneous model, Poisson GLM (frequency) and Gamma GLM (severity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.features import glm_columns, glm_design
from src.utils import read_json, write_json


class HomogeneousFrequency:
    """One claim rate for everyone: lambda = sum(ClaimNb) / sum(Exposure)."""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "HomogeneousFrequency":
        self.rate = float(train.ClaimNb.sum() / train.Exposure.sum())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.rate * df["Exposure"].to_numpy(float)

    def save(self, d: Path) -> None:
        write_json({"rate": self.rate}, d / "model.json")

    @classmethod
    def load(cls, d: Path) -> "HomogeneousFrequency":
        m = cls()
        m.rate = read_json(d / "model.json")["rate"]
        return m


class _GLM:
    family: sm.families.Family

    def __init__(self, coef: np.ndarray | None = None):
        self.coef = coef

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray:
        """log of the expected value per unit exposure (no offset)."""
        return glm_design(df) @ self.coef

    def save(self, d: Path) -> None:
        write_json({"columns": glm_columns(), "coef": self.coef.tolist(),
                    "bse": getattr(self, "bse", np.full_like(self.coef, np.nan)).tolist()}, d / "model.json")

    @classmethod
    def load(cls, d: Path):
        j = read_json(d / "model.json")
        assert j["columns"] == glm_columns(), "GLM design changed since this model was saved"
        m = cls(np.asarray(j["coef"]))
        m.bse = np.asarray(j["bse"])
        return m

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.coef, "se": self.bse}, index=glm_columns())


class PoissonGLM(_GLM):
    """log E[N] = log(Exposure) + x'beta, fitted by IRLS (statsmodels)."""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "PoissonGLM":
        res = sm.GLM(train.ClaimNb.to_numpy(float), glm_design(train), family=sm.families.Poisson(),
                     offset=np.log(train.Exposure.to_numpy(float))).fit()
        self.coef, self.bse = np.asarray(res.params), np.asarray(res.bse)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.linear_predictor(df)) * df["Exposure"].to_numpy(float)


class GammaGLM(_GLM):
    """log E[claim amount] = x'beta with a Gamma response, one row per claim."""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "GammaGLM":
        y = train.ClaimAmount.to_numpy(float)
        res = sm.GLM(y, glm_design(train), family=sm.families.Gamma(sm.families.links.Log())).fit(
            start_params=np.r_[np.log(y.mean()), np.zeros(len(glm_columns()) - 1)])
        self.coef, self.bse = np.asarray(res.params), np.asarray(res.bse)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.linear_predictor(df))
