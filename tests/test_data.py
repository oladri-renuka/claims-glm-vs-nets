import numpy as np
import pandas as pd
import pytest

from src.data.features import NNPreprocessor, glm_columns, glm_design
from src.data.load import apply_caps, apply_consistency_rule, split

CAPS = {"ClaimNb": 4, "Exposure": 1.0, "VehAge": 20, "DrivAge": 90, "BonusMalus": 150}


def test_caps_and_log_density(policies):
    df = apply_caps(policies.assign(ClaimNb=7, Exposure=1.5), CAPS)
    for col, cap in CAPS.items():
        assert df[col].max() <= cap
    assert np.allclose(df.LogDensity, np.log(policies.Density))
    assert (df.ClaimNbRaw == 7).all()


def test_split_is_disjoint_stratified_and_reproducible(policies):
    a, b = split(policies, [0.8, 0.1, 0.1], 1), split(policies, [0.8, 0.1, 0.1], 1)
    assert (a == b).all()
    shares = a.value_counts(normalize=True)
    assert abs(shares["train"] - 0.8) < 0.01 and abs(shares["test"] - 0.1) < 0.01
    pos = policies.ClaimNb > 0
    rates = pos.groupby(a).mean()
    assert rates.max() - rates.min() < 0.01


def test_consistency_rule_counts():
    freq = pd.DataFrame({"IDpol": [1, 2, 3, 4], "ClaimNb": [1, 2, 0, 1]})
    sev = pd.DataFrame({"IDpol": [1, 2, 9, 9], "ClaimAmount": [100.0, 50.0, 10.0, 20.0]})
    kept, rep = apply_consistency_rule(freq, sev)
    assert kept.IDpol.tolist() == [1]                       # policy 2 has 1 of 2 claims -> dropped
    assert rep["severity_rows_orphan_policy"] == 2          # policy 9 not in frequency table
    assert rep["policies_with_claims_but_no_severity"] == 1  # policy 4
    assert rep["frequency_rows_dropped"] == 0


def test_glm_design_shape_and_reference_levels(policies):
    x = glm_design(policies)
    assert x.shape == (len(policies), len(glm_columns()))
    assert np.allclose(x[:, 0], 1)
    assert not any(c.startswith("DrivAgeGLM[41-50]") for c in glm_columns())


def test_preprocessor_rejects_unknown_category(policies):
    pre = NNPreprocessor().fit(policies)
    xc, xk = pre.transform(policies)
    assert np.allclose(xc.mean(0), 0, atol=1e-5)
    with pytest.raises(ValueError, match="unknown VehBrand"):
        pre.transform(policies.assign(VehBrand="B99"))
