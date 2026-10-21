"""FastAPI pricing service.

    uvicorn serve.app:app --port 8000
    curl -X POST localhost:8000/quote -H 'content-type: application/json' -d @serve/example_policy.json
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query

from serve.predictor import Predictor
from serve.schemas import Policy, Quote

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()
    yield
    state.clear()


app = FastAPI(title="freMTPL2 pricing API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    p: Predictor = state["predictor"]
    return {"status": "ok", "frequency_nn": p.manifest["freq_nn"], "bundle_mb": round(p.size_mb(), 3)}


@app.post("/quote", response_model=Quote)
def quote(policy: Policy, model: Literal["nn", "glm"] = Query("nn")) -> Quote:
    return Quote(**state["predictor"].quote(policy.model_dump(), model))
