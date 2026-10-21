"""Training wrappers for the neural models: FFNN / CANN frequency, Gamma NN severity,
tabular autoencoder and VAE. Each wrapper owns its preprocessor and saves a
self-contained directory (weights + preprocessing + any frozen GLM coefficients)."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.data.features import NNPreprocessor
from src.data.schema import NN_CONTINUOUS
from src.metrics import gamma_deviance, poisson_deviance
from src.models.glm import PoissonGLM
from src.models.nets import TabularAutoencoder, TabularRegressor
from src.utils import read_json, write_json


def _tensors(pre: NNPreprocessor, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    xc, xk = pre.transform(df)
    return torch.from_numpy(xc), torch.from_numpy(xk)


def fit_loop(model: torch.nn.Module, n: int, batch_loss: Callable[[torch.Tensor], torch.Tensor],
             val_metric: Callable[[], float], p: dict, seed: int, log=None) -> dict:
    """Adam + mini-batches + early stopping on a validation metric; restores the best weights."""
    torch.set_num_threads(p.get("num_threads", 2))
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    best, best_state, wait, hist = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(p["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, p["batch_size"]):
            idx = perm[i : i + p["batch_size"]]
            opt.zero_grad()
            loss = batch_loss(idx)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            v = val_metric()
        hist.append({"epoch": epoch, "train_loss": tot / n, "val": v})
        if log:
            log({"train_loss": tot / n, "val_metric": v}, epoch)
        if v < best - 1e-7:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= p["patience"]:
                break
    model.load_state_dict(best_state)
    return {"best_val": best, "epochs": len(hist), "best_epoch": len(hist) - 1 - wait,
            "seconds": time.time() - t0, "history": hist}


def _build_regressor(p: dict, pre: NNPreprocessor) -> TabularRegressor:
    return TabularRegressor(len(NN_CONTINUOUS), pre.cardinalities, p["categorical_encoding"],
                            p["embedding_dim"], p["hidden"], p["activation"])


class NeuralFrequency:
    """FFNN:  E[N] = Exposure * exp(s(x)).
    CANN:  E[N] = Exposure * exp(glm_lp(x) + s(x)), GLM frozen; the head of s starts at
           zero so the model is initialised exactly at the GLM."""

    def __init__(self, params: dict, seed: int = 0, glm: PoissonGLM | None = None):
        self.p, self.seed, self.glm = params, seed, glm
        self.cann = glm is not None

    def _offset(self, df: pd.DataFrame) -> torch.Tensor:
        off = np.log(df["Exposure"].to_numpy(float))
        if self.cann:
            off = off + self.glm.linear_predictor(df)
        return torch.from_numpy(off.astype(np.float32))

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, log=None) -> "NeuralFrequency":
        self.pre = NNPreprocessor().fit(train)
        self.net = _build_regressor(self.p, self.pre)
        if self.cann:
            self.net.init_output(0.0, zero_weights=True)
        else:
            self.net.init_output(float(np.log(train.ClaimNb.sum() / train.Exposure.sum())))
        xc, xk = _tensors(self.pre, train)
        off, y = self._offset(train), torch.tensor(train.ClaimNb.to_numpy(np.float32))
        vxc, vxk = _tensors(self.pre, val)
        voff, vy = self._offset(val), val.ClaimNb.to_numpy(float)

        def batch_loss(idx):
            eta = self.net(xc[idx], xk[idx]) + off[idx]
            return (torch.exp(eta) - y[idx] * eta).mean()   # Poisson NLL up to a constant

        def val_metric():
            return poisson_deviance(vy, torch.exp(self.net(vxc, vxk) + voff).numpy())

        self.fit_info = fit_loop(self.net, len(train), batch_loss, val_metric, self.p, self.seed, log)
        return self

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        xc, xk = _tensors(self.pre, df)
        return torch.exp(self.net(xc, xk) + self._offset(df)).numpy().astype(float)

    def save(self, d: Path) -> None:
        torch.save(self.net.state_dict(), d / "weights.pt")
        write_json({"params": self.p, "seed": self.seed, "pre": self.pre.to_dict(),
                    "glm_coef": self.glm.coef.tolist() if self.cann else None}, d / "model.json")

    @classmethod
    def load(cls, d: Path) -> "NeuralFrequency":
        j = read_json(d / "model.json")
        glm = PoissonGLM(np.asarray(j["glm_coef"])) if j["glm_coef"] is not None else None
        m = cls(j["params"], j["seed"], glm)
        m.pre = NNPreprocessor.from_dict(j["pre"])
        m.net = _build_regressor(m.p, m.pre)
        m.net.load_state_dict(torch.load(d / "weights.pt", weights_only=True))
        m.net.eval()
        return m


class NeuralSeverity:
    """E[claim amount] = exp(s(x)), trained with the Gamma deviance on claims only."""

    def __init__(self, params: dict, seed: int = 0):
        self.p, self.seed = params, seed

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, log=None) -> "NeuralSeverity":
        self.pre = NNPreprocessor().fit(train)
        self.net = _build_regressor(self.p, self.pre)
        self.net.init_output(float(np.log(train.ClaimAmount.mean())))
        xc, xk = _tensors(self.pre, train)
        y = torch.tensor(train.ClaimAmount.to_numpy(np.float32))
        vxc, vxk = _tensors(self.pre, val)
        vy = val.ClaimAmount.to_numpy(float)

        def batch_loss(idx):
            eta = self.net(xc[idx], xk[idx])
            return (y[idx] * torch.exp(-eta) + eta).mean()  # Gamma NLL up to a constant

        def val_metric():
            return gamma_deviance(vy, torch.exp(self.net(vxc, vxk)).numpy())

        self.fit_info = fit_loop(self.net, len(train), batch_loss, val_metric, self.p, self.seed, log)
        return self

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        xc, xk = _tensors(self.pre, df)
        return torch.exp(self.net(xc, xk)).numpy().astype(float)

    def save(self, d: Path) -> None:
        torch.save(self.net.state_dict(), d / "weights.pt")
        write_json({"params": self.p, "seed": self.seed, "pre": self.pre.to_dict()}, d / "model.json")

    @classmethod
    def load(cls, d: Path) -> "NeuralSeverity":
        j = read_json(d / "model.json")
        m = cls(j["params"], j["seed"])
        m.pre = NNPreprocessor.from_dict(j["pre"])
        m.net = _build_regressor(m.p, m.pre)
        m.net.load_state_dict(torch.load(d / "weights.pt", weights_only=True))
        m.net.eval()
        return m


class TabularAE:
    """Autoencoder (variational=False) or VAE (variational=True) over all rating factors."""

    def __init__(self, params: dict, seed: int = 0, variational: bool = False):
        self.p, self.seed, self.variational = params, seed, variational

    def _build(self) -> TabularAutoencoder:
        return TabularAutoencoder(len(NN_CONTINUOUS), self.pre.cardinalities, self.p["embedding_dim"],
                                  self.p["hidden"], self.p["latent_dim"], self.p["activation"],
                                  self.variational)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, log=None) -> "TabularAE":
        self.pre = NNPreprocessor().fit(train)
        self.net = self._build()
        xc, xk = _tensors(self.pre, train)
        vxc, vxk = _tensors(self.pre, val)
        beta = self.p.get("beta", 1.0)

        def batch_loss(idx):
            rec, kl = self.net.row_losses(xc[idx], xk[idx])
            return (rec + beta * kl).mean()

        def val_metric():
            # Fixed noise so the early-stopping signal is comparable across epochs, without
            # disturbing the training RNG stream.
            state = torch.get_rng_state()
            torch.manual_seed(0)
            rec, kl = self.net.row_losses(vxc, vxk)
            torch.set_rng_state(state)
            return float((rec + beta * kl).mean())

        self.fit_info = fit_loop(self.net, len(train), batch_loss, val_metric, self.p, self.seed, log)
        if not self.variational:
            self.threshold = float(np.quantile(self.score(train), self.p["anomaly_quantile"]))
        return self

    @torch.no_grad()
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Per-policy reconstruction error (deterministic: uses the latent mean)."""
        self.net.eval()
        rec, _ = self.net.row_losses(*_tensors(self.pre, df), sample=False)
        return rec.numpy().astype(float)

    @torch.no_grad()
    def latent(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        mu, _ = self.net.encode(*_tensors(self.pre, df))
        return mu.numpy().astype(float)

    def save(self, d: Path) -> None:
        torch.save(self.net.state_dict(), d / "weights.pt")
        write_json({"params": self.p, "seed": self.seed, "variational": self.variational,
                    "pre": self.pre.to_dict(), "threshold": getattr(self, "threshold", None)},
                   d / "model.json")

    @classmethod
    def load(cls, d: Path) -> "TabularAE":
        j = read_json(d / "model.json")
        m = cls(j["params"], j["seed"], j["variational"])
        m.pre = NNPreprocessor.from_dict(j["pre"])
        m.threshold = j["threshold"]
        m.net = m._build()
        m.net.load_state_dict(torch.load(d / "weights.pt", weights_only=True))
        m.net.eval()
        return m
