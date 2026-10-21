"""Download, clean and split freMTPL2.

Consistency rule (applied once, counts written to data/processed/cleaning_report.json):
  The frequency table is the source of truth for claim *counts*. Severity rows are kept
  only for policies whose number of severity records equals their raw ClaimNb. So:
    * frequency models use all 678,013 policies with ClaimNb as published (comparable to
      Noll, Salzmann & Wuthrich 2018);
    * severity rows of policies absent from the frequency table, or whose count does not
      match, are dropped from the severity model (assumed missing at random).

Run:  python -m src.data.load --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schema import FEATURES as FEATURES_KEY
from src.utils import ROOT, load_config, write_json


def download(cfg: dict) -> tuple[Path, Path]:
    raw = ROOT / cfg["data"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, fname in [("freq_url", "freMTPL2freq.csv"), ("sev_url", "freMTPL2sev.csv")]:
        p = raw / fname
        if not p.exists():
            print(f"downloading {cfg['data'][key]}")
            urllib.request.urlretrieve(cfg["data"][key], p)
        paths.append(p)
    return paths[0], paths[1]


def apply_consistency_rule(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the severity rows that fully account for their policy's ClaimNb, plus a report."""
    nsev = sev.groupby("IDpol").size()
    n_claims = freq.set_index("IDpol")["ClaimNb"]
    joined = pd.DataFrame({"ClaimNb": n_claims}).join(nsev.rename("nsev")).fillna({"nsev": 0})
    mismatched = joined[joined.ClaimNb != joined.nsev]
    orphan = ~sev.IDpol.isin(freq.IDpol)
    partial = sev.IDpol.isin(mismatched.index)
    keep = ~orphan & ~partial
    report = {
        "rule": "keep frequency ClaimNb as published; keep severity rows only where #rows == ClaimNb",
        "policies_total": int(len(freq)),
        "policies_count_mismatch": int(len(mismatched)),
        "policies_with_claims_but_no_severity": int(((joined.ClaimNb > 0) & (joined.nsev == 0)).sum()),
        "claims_without_severity": int((mismatched.ClaimNb - mismatched.nsev).clip(lower=0).sum()),
        "severity_rows_total": int(len(sev)),
        "severity_rows_orphan_policy": int(orphan.sum()),
        "severity_rows_partial_policy": int((partial & ~orphan).sum()),
        "severity_rows_kept": int(keep.sum()),
        "frequency_rows_dropped": 0,
    }
    return sev[keep].reset_index(drop=True), report


def apply_caps(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    df = df.copy()
    df["ClaimNbRaw"] = df["ClaimNb"]
    for col, cap in caps.items():
        df[col] = np.minimum(df[col], cap)
    df["LogDensity"] = np.log(df["Density"])
    df["VehGasDiesel"] = (df["VehGas"] == "Diesel").astype(float)
    return df


def split(df: pd.DataFrame, fractions: list[float], seed: int) -> pd.Series:
    """80/10/10 split stratified on ClaimNb > 0. Returns a 'train'/'val'/'test' label per row."""
    strat = (df["ClaimNb"] > 0).astype(int)
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, train_size=fractions[0], stratify=strat, random_state=seed)
    val_share = fractions[1] / (fractions[1] + fractions[2])
    va, te = train_test_split(rest, train_size=val_share, stratify=strat.iloc[rest], random_state=seed)
    labels = np.empty(len(df), dtype=object)
    labels[tr], labels[va], labels[te] = "train", "val", "test"
    return pd.Series(labels, index=df.index, name="split")


def grouped_split(df: pd.DataFrame, fractions: list[float], seed: int) -> pd.Series:
    """Main split: policies with identical rating factors (the same insured re-issued under
    a new IDpol, ~38% of rows) always land in the same split, so no test policy has a twin
    in train. Groups are stratified on 'any claim in the group' and assigned 80/10/10 by
    row count within each stratum."""
    from src.data.schema import FEATURES

    gid = df.groupby(FEATURES, observed=True).ngroup().to_numpy()
    g = pd.DataFrame({"gid": gid, "claim": df["ClaimNb"].to_numpy() > 0}).groupby("gid").agg(
        rows=("claim", "size"), claim=("claim", "any"))
    rng = np.random.default_rng(seed)
    label = pd.Series("", index=g.index, dtype=object)
    cut = np.cumsum(fractions)[:2]
    for _, stratum in g.groupby("claim"):
        order = stratum.index.to_numpy()[rng.permutation(len(stratum))]
        share = stratum.rows.loc[order].cumsum().to_numpy() / stratum.rows.sum()
        label.loc[order] = np.select([share <= cut[0], share <= cut[1]], ["train", "val"], "test")
    return pd.Series(label.loc[gid].to_numpy(), index=df.index, name="split_grouped")


def build(cfg: dict) -> None:
    fpath, spath = download(cfg)
    freq = pd.read_csv(fpath)
    sev = pd.read_csv(spath)
    sev_kept, report = apply_consistency_rule(freq, sev)

    caps = cfg["data"]["caps"]
    report["rows_capped"] = {c: int((freq[c] > v).sum()) for c, v in caps.items()}
    df = apply_caps(freq, caps)
    df["split"] = split(df, cfg["split"]["fractions"], cfg["split"]["seed"]).values
    df["split_grouped"] = grouped_split(df, cfg["split"]["fractions"], cfg["split"]["seed"]).values
    for col in ["split", "split_grouped"]:
        tr_profiles = set(df.loc[df[col] == "train", FEATURES_KEY].itertuples(index=False, name=None))
        te = df.loc[df[col] == "test", FEATURES_KEY].itertuples(index=False, name=None)
        report[f"test_rows_with_identical_train_profile[{col}]"] = float(np.mean([t in tr_profiles for t in te]))

    claims = sev_kept.merge(df.drop(columns=["ClaimNb", "ClaimNbRaw"]), on="IDpol", how="inner")
    cap_q = cfg["data"]["severity_cap_quantile"]
    main = cfg["split"].get("column", "split")
    sev_cap = float(claims.loc[claims[main] == "train", "ClaimAmount"].quantile(cap_q))
    claims["ClaimAmountRaw"] = claims["ClaimAmount"]
    claims["ClaimAmount"] = np.minimum(claims["ClaimAmount"], sev_cap)
    # Large-loss load: ratio of uncapped to capped total on train, applied to the pure premium.
    tr = claims[claims[main] == "train"]
    report["severity_cap"] = {
        "quantile": cap_q, "value": sev_cap,
        "claims_capped": int((claims.ClaimAmountRaw > sev_cap).sum()),
        "large_loss_load": float(tr.ClaimAmountRaw.sum() / tr.ClaimAmount.sum()),
    }
    # Per-policy (capped) losses, used for Lorenz curves on pure premium.
    loss = claims.groupby("IDpol")["ClaimAmount"].sum()
    df["ClaimTotal"] = df["IDpol"].map(loss).fillna(0.0)
    df["HasSeverity"] = df["IDpol"].isin(claims.IDpol) | (df["ClaimNbRaw"] == 0)

    report["main_split"] = main
    report["split_sizes"] = df["split"].value_counts().to_dict()
    report["split_claim_rate"] = df.groupby("split").apply(
        lambda g: g.ClaimNb.sum() / g.Exposure.sum(), include_groups=False).to_dict()
    report["severity_split_sizes"] = claims[main].value_counts().to_dict()
    report["grouped_split_sizes"] = df["split_grouped"].value_counts().to_dict()
    report["grouped_split_claim_rate"] = df.groupby("split_grouped").apply(
        lambda g: g.ClaimNb.sum() / g.Exposure.sum(), include_groups=False).to_dict()

    out = ROOT / cfg["data"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "policies.parquet", index=False)
    claims.to_parquet(out / "claims.parquet", index=False)
    write_json(report, out / "cleaning_report.json")
    print(pd.Series(report).to_string())


def _select(df: pd.DataFrame, cfg: dict, splits: tuple[str, ...]) -> pd.DataFrame:
    col = cfg["split"].get("column", "split")
    df = df.assign(split=df[col])
    return df[df.split.isin(splits)].reset_index(drop=True)


def load_policies(cfg: dict, splits: tuple[str, ...] = ("train", "val")) -> pd.DataFrame:
    """Load processed policies; `split` holds the labels of the configured split column.
    The test split must be requested explicitly (only src/eval.py does)."""
    return _select(pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "policies.parquet"), cfg, splits)


def load_claims(cfg: dict, splits: tuple[str, ...] = ("train", "val")) -> pd.DataFrame:
    return _select(pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "claims.parquet"), cfg, splits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    build(load_config(ap.parse_args().config))
