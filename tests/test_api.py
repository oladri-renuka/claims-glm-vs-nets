import json
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "serve" / "model_bundle"
pytestmark = pytest.mark.skipif(not (BUNDLE / "manifest.json").exists(), reason="run `make all` to build the bundle")
POLICY = json.loads((BUNDLE.parent / "example_policy.json").read_text())


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from serve.app import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("model", ["nn", "glm"])
def test_quote_ok(client, model):
    r = client.post(f"/quote?model={model}", json=POLICY)
    assert r.status_code == 200, r.text
    q = r.json()
    assert 0 < q["expected_frequency"] < 2 and q["expected_severity"] > 0
    assert q["pure_premium"] == pytest.approx(q["expected_frequency"] * q["expected_severity"])
    assert isinstance(q["anomaly"], bool) and q["risk_segment"] >= 0


def test_unknown_brand_is_a_clear_422(client):
    r = client.post("/quote", json={**POLICY, "VehBrand": "B99"})
    assert r.status_code == 422 and "unknown VehBrand 'B99'" in r.text


def test_unknown_region_and_out_of_range(client):
    assert "unknown Region" in client.post("/quote", json={**POLICY, "Region": "Bavaria"}).text
    assert client.post("/quote", json={**POLICY, "DrivAge": 12}).status_code == 422
    assert client.post("/quote", json={**POLICY, "Exposure": 0}).status_code == 422


def test_bonus_malus_increases_premium(client):
    lo = client.post("/quote?model=glm", json={**POLICY, "BonusMalus": 50}).json()["pure_premium"]
    hi = client.post("/quote?model=glm", json={**POLICY, "BonusMalus": 120}).json()["pure_premium"]
    assert hi > lo
