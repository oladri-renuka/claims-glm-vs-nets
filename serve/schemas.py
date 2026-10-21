"""Request / response models for the pricing API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.schema import RANGES, REGIONS, VEH_BRANDS


def _rng(name: str) -> dict:
    lo, hi = RANGES[name]
    return {"ge": lo, "le": hi}


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "Area": "C", "VehPower": 6, "VehAge": 5, "DrivAge": 40, "BonusMalus": 50, "VehBrand": "B12",
        "VehGas": "Regular", "Density": 1200, "Region": "Ile-de-France", "Exposure": 1.0}})

    Area: Literal["A", "B", "C", "D", "E", "F"]
    VehPower: int = Field(..., **_rng("VehPower"))
    VehAge: int = Field(..., **_rng("VehAge"))
    DrivAge: int = Field(..., **_rng("DrivAge"))
    BonusMalus: int = Field(..., **_rng("BonusMalus"))
    VehBrand: str
    VehGas: Literal["Diesel", "Regular"]
    Density: float = Field(..., **_rng("Density"))
    Region: str
    Exposure: float = Field(1.0, gt=0, le=1.0, description="Policy-years covered (capped at 1 in training)")

    @field_validator("VehBrand")
    @classmethod
    def _brand(cls, v: str) -> str:
        if v not in VEH_BRANDS:
            raise ValueError(f"unknown VehBrand '{v}'; expected one of {VEH_BRANDS}")
        return v

    @field_validator("Region")
    @classmethod
    def _region(cls, v: str) -> str:
        if v not in REGIONS:
            raise ValueError(f"unknown Region '{v}'; expected one of {REGIONS}")
        return v


class Quote(BaseModel):
    model: str = Field(description="'nn' (neural frequency + Gamma NN) or 'glm' (Poisson GLM + Gamma GLM)")
    expected_frequency: float = Field(description="Expected claims per policy-year")
    expected_claims: float = Field(description="Expected claims over the requested exposure")
    expected_severity: float = Field(description="Expected cost per claim (EUR, capped losses x large-loss load)")
    pure_premium: float = Field(description="Annual pure premium = frequency x severity (EUR)")
    pure_premium_for_exposure: float
    risk_segment: int = Field(description="VAE latent k-means segment id")
    segment_frequency: float = Field(description="Observed training claim frequency of that segment")
    anomaly: bool = Field(description="Autoencoder reconstruction error above the training 99th percentile")
    anomaly_score: float
