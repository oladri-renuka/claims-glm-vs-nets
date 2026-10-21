"""Latency (p50 / p95, CPU, single request) and model size.

    python -m serve.benchmark            -> reports/tables/latency.json|md

Measures (a) in-process model inference and (b) end-to-end HTTP against a local uvicorn.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from serve.predictor import Predictor
from src.utils import ROOT, write_json

N = 1000
POLICY = json.loads((Path(__file__).parent / "example_policy.json").read_text())


def pct(ts) -> dict:
    a = np.asarray(ts) * 1000
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "n": len(a)}


def main() -> None:
    p = Predictor()
    res = {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
           or "unknown", "torch_threads": 1, "bundle_mb": p.size_mb()}
    for model in ["nn", "glm"]:
        for _ in range(50):
            p.quote(POLICY, model)
        ts = []
        for _ in range(N):
            t = time.perf_counter()
            p.quote(POLICY, model)
            ts.append(time.perf_counter() - t)
        res[f"inprocess_{model}"] = pct(ts)

    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "serve.app:app", "--port", "8765", "--log-level", "warning"],
                              cwd=ROOT)
    try:
        with httpx.Client(base_url="http://127.0.0.1:8765", timeout=10) as c:
            for _ in range(100):
                try:
                    c.get("/health")
                    break
                except httpx.TransportError:
                    time.sleep(0.2)
            for model in ["nn", "glm"]:
                for _ in range(50):
                    c.post(f"/quote?model={model}", json=POLICY)
                ts = []
                for _ in range(N):
                    t = time.perf_counter()
                    r = c.post(f"/quote?model={model}", json=POLICY)
                    ts.append(time.perf_counter() - t)
                    r.raise_for_status()
                res[f"http_{model}"] = pct(ts)
    finally:
        server.terminate()
        server.wait()
    out = ROOT / "reports" / "tables"
    write_json(res, out / "latency.json")
    rows = [f"| {k} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} |" for k, v in res.items() if isinstance(v, dict)]
    (out / "latency.md").write_text(f"CPU: {res['cpu']}, 1 torch thread, {N} sequential requests. Bundle size: "
                                    f"{res['bundle_mb']:.2f} MB.\n\n| path | p50 (ms) | p95 (ms) |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
