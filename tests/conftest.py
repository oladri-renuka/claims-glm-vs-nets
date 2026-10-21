import numpy as np
import pandas as pd
import pytest

from src.data.schema import AREAS, REGIONS, VEH_BRANDS


@pytest.fixture
def policies() -> pd.DataFrame:
    """Small synthetic portfolio with the real schema (no download needed)."""
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({
        "IDpol": np.arange(n), "Area": rng.choice(AREAS, n), "VehPower": rng.integers(4, 16, n),
        "VehAge": rng.integers(0, 30, n), "DrivAge": rng.integers(18, 100, n),
        "BonusMalus": rng.integers(50, 200, n), "VehBrand": rng.choice(VEH_BRANDS, n),
        "VehGas": rng.choice(["Diesel", "Regular"], n), "Density": rng.integers(1, 27000, n),
        "Region": rng.choice(REGIONS, n), "Exposure": rng.uniform(0.05, 1.0, n),
    })
    lam = 0.08 * np.exp(0.01 * (df.BonusMalus - 50) - 0.01 * (df.DrivAge - 18).clip(upper=30))
    df["ClaimNb"] = rng.poisson(lam * df.Exposure)
    df["VehGasDiesel"] = (df.VehGas == "Diesel").astype(float)
    return df
