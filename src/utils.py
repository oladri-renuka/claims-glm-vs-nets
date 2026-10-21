"""Config loading, seeding and experiment tracking."""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving a single-level `inherit:` chain relative to the file."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg


def nn_params(cfg: dict) -> dict:
    """Neural-net hyper-parameters: base defaults overridden by the experiment's params."""
    return _merge(cfg.get("nn_defaults", {}), cfg.get("params", {}))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = sys.modules.get("torch")  # seed torch only if this run uses it (keeps LightGBM runs torch-free)
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def run_dir(cfg: dict, seed: int) -> Path:
    d = ROOT / cfg["artifacts_dir"] / cfg["name"] / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


class Tracker:
    """Thin MLflow wrapper; a no-op when tracking is disabled or MLflow is missing."""

    def __init__(self, cfg: dict):
        t = cfg.get("tracking", {})
        self.enabled = bool(t.get("enabled"))
        if self.enabled:
            try:
                import mlflow

                uri = t["uri"]
                if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
                    uri = f"sqlite:///{ROOT / uri.removeprefix('sqlite:///')}"
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(t["experiment"])
                self.mlflow = mlflow
            except ImportError:
                self.enabled = False

    @contextlib.contextmanager
    def run(self, name: str, params: dict) -> Iterator["Tracker"]:
        if not self.enabled:
            yield self
            return
        with self.mlflow.start_run(run_name=name):
            flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()}
            self.mlflow.log_params(flat)
            yield self

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
