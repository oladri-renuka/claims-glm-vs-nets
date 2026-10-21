"""PyTorch modules: tabular MLP (embeddings or one-hot), autoencoder and VAE."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


class CategoricalEncoder(nn.Module):
    """Maps integer-coded categoricals to entity embeddings or one-hot vectors."""

    def __init__(self, cards: list[int], encoding: str = "embedding", emb_dim: int = 2):
        super().__init__()
        self.cards, self.encoding = list(cards), encoding
        if encoding == "embedding":
            self.emb = nn.ModuleList(nn.Embedding(c, emb_dim) for c in cards)
            self.out_dim = emb_dim * len(cards)
        elif encoding == "onehot":
            self.out_dim = sum(cards)
        else:
            raise ValueError(encoding)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if self.encoding == "embedding":
            return torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.emb)], dim=1)
        return torch.cat([F.one_hot(x_cat[:, i], c).float() for i, c in enumerate(self.cards)], dim=1)


def mlp(sizes: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), ACTIVATIONS[activation]()]
    return nn.Sequential(*layers)


class TabularRegressor(nn.Module):
    """Returns a log-scale score s(x); the wrapper adds offsets and applies exp(.)."""

    def __init__(self, n_cont: int, cards: list[int], encoding: str, emb_dim: int,
                 hidden: list[int], activation: str):
        super().__init__()
        self.cat = CategoricalEncoder(cards, encoding, emb_dim)
        self.body = mlp([n_cont + self.cat.out_dim, *hidden], activation)
        self.head = nn.Linear(hidden[-1], 1)

    def init_output(self, bias: float, zero_weights: bool = False) -> None:
        nn.init.constant_(self.head.bias, bias)
        if zero_weights:
            nn.init.zeros_(self.head.weight)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat([x_cont, self.cat(x_cat)], dim=1))).squeeze(1)


class TabularAutoencoder(nn.Module):
    """Encoder/decoder over continuous + categorical inputs. With variational=True the
    encoder outputs (mu, logvar) and sampling uses the reparameterization trick."""

    def __init__(self, n_cont: int, cards: list[int], emb_dim: int, hidden: list[int],
                 latent_dim: int, activation: str, variational: bool = False):
        super().__init__()
        self.cards, self.n_cont, self.variational = list(cards), n_cont, variational
        self.cat = CategoricalEncoder(cards, "embedding", emb_dim)
        self.enc = mlp([n_cont + self.cat.out_dim, *hidden], activation)
        self.to_latent = nn.Linear(hidden[-1], latent_dim * (2 if variational else 1))
        self.dec = mlp([latent_dim, *reversed(hidden)], activation)
        self.out = nn.Linear(hidden[0], n_cont + sum(cards))

    def encode(self, x_cont, x_cat):
        h = self.to_latent(self.enc(torch.cat([x_cont, self.cat(x_cat)], dim=1)))
        if self.variational:
            mu, logvar = h.chunk(2, dim=1)
            return mu, logvar
        return h, None

    def decode(self, z):
        o = self.out(self.dec(z))
        cont = o[:, : self.n_cont]
        logits = torch.split(o[:, self.n_cont:], self.cards, dim=1)
        return cont, logits

    def row_losses(self, x_cont, x_cat, sample: bool = True):
        """Per-row (reconstruction, KL) losses: summed MSE over continuous features plus
        summed cross-entropy over categoricals; KL is 0 for the plain autoencoder."""
        mu, logvar = self.encode(x_cont, x_cat)
        if self.variational and sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        cont, logits = self.decode(z)
        rec = ((cont - x_cont) ** 2).sum(1)
        for i, lg in enumerate(logits):
            rec = rec + F.cross_entropy(lg, x_cat[:, i], reduction="none")
        kl = (-0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1)) if self.variational else torch.zeros_like(rec)
        return rec, kl
