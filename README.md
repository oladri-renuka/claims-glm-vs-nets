# Neural claim pricing and risk segmentation on freMTPL2 (678K French motor policies)

**Result: on held-out test data, a 5-seed neural ensemble cuts Poisson deviance by 2.1–2.2% versus the industry-standard Poisson GLM (31.18 → 30.50 ×10⁻², 95% CI excludes 0), and the gain survives a leakage-free grouped split. A tuned LightGBM beats both (−4.0%). Neural nets do not help on severity, and VAE segments separate risk far worse than simple GLM-frequency bands.**

![Double-lift chart](reports/figures/double_lift.png)

*Double-lift on test: policies sorted by the FFNN/GLM prediction ratio. Where the two models disagree (deciles 1 and 10), the observed frequency (black) follows the neural net. The GLM over-prices decile 1 by 70% and under-prices decile 10 by 43%.*

## How to run (3 commands)

```bash
make setup     # pip install -r requirements.txt
make all       # download → clean/split → train every config × seeds → evaluate on test → serving bundle → latency benchmark
make serve     # FastAPI on :8000   (make app → Streamlit premium explorer; make test → 18 tests)
```

Example quote: `curl -X POST 'localhost:8000/quote?model=nn' -H 'content-type: application/json' -d @serve/example_policy.json`

## Results

All numbers are on the untouched **test** split (67,802 policies). Only `src/eval.py` reads it, and it ran once after every model had been chosen on validation. "± std" is across training seeds with a fixed data split. "Ensemble" averages the seed predictions. Deviance is the mean unit deviance ×10⁻² (lower is better).

### Q1 — Frequency (I1, I2)

| Model | Seeds | Test deviance (mean ± std) | Seed ensemble | Δ vs GLM | GLM − model, 95% bootstrap CI | Σpred/Σactual |
|---|---|---|---|---|---|---|
| Homogeneous (one rate) | 1 | 32.944 | 32.944 | −5.65% | [−1.99, −1.54] | 1.002 |
| **Poisson GLM** (Noll et al. binning) | 1* | 31.183 | 31.183 | — | — | 1.003 |
| LightGBM (Poisson, offset) | 5 | **29.959 ± 0.016** | **29.947** | **+3.96%** | [1.04, 1.45] | 1.004 |
| FFNN, one-hot categoricals | 5 | 30.629 ± 0.028 | 30.546 | +2.04% | [0.50, 0.79] | 0.995 |
| FFNN, entity embeddings | 5 | 30.597 ± 0.048 | 30.516 | +2.14% | [0.54, 0.81] | 1.007 |
| CANN (GLM skip connection) | 5 | 30.598 ± 0.043 | 30.500 | +2.19% | [0.56, 0.82] | 1.002 |

\*The GLM and the homogeneous model are fitted deterministically (IRLS / closed form), so their seed variance is 0 by construction.

**Does the gain survive a fair comparison? (I7)** 32.8% of test policies have an exact rating-factor twin in train, most likely the same insured re-issued under a new `IDpol`. Flexible models can memorise these twins. Every model was therefore retrained on `split_grouped`, which keeps twins in the same split:

| Grouped split (3 seeds) | Ensemble test deviance | Δ vs GLM |
|---|---|---|
| GLM | 31.331 | — |
| LightGBM | 30.338 | +3.17% (was +3.96%) |
| FFNN (embeddings) | 30.686 | +2.06% (was +2.14%) |
| CANN | 30.728 | +1.93% (was +2.19%) |

Twin leakage accounts for about a fifth of LightGBM's advantage but barely affects the neural nets. On both splits every model's gain over the GLM keeps a 95% CI above zero.

- **Embeddings vs one-hot (I2):** a small, consistent win for embeddings: 30.516 vs 30.546. That gap is about the size of the seed noise, so it is not a strong effect.
- **CANN vs FFNN:** CANN ties with the FFNN. Starting from the GLM is a useful safety property (the network only learns the residual), but it does not improve accuracy here.
- **Seed ensembling:** averaging seeds is worth 0.08–0.10 ×10⁻², about twice the single-seed std.
- **Calibration:** all models satisfy the balance property within 0.7% on test. Decile calibration plots: [reports/figures/calibration_deciles.png](reports/figures/calibration_deciles.png).

### Severity (I4): deep learning does not help with 21K claims

| Model | Seeds | Test Gamma deviance | Ensemble | GLM − model, 95% CI |
|---|---|---|---|---|
| Constant mean | 1 | 1.1638 | 1.1638 | [−0.022, 0.022] |
| Gamma GLM | 1 | 1.1641 | 1.1641 | — |
| Gamma NN | 5 | 1.1556 ± 0.0042 | 1.1544 | [−0.010, 0.030] |

The Gamma GLM does no better than a constant mean, and the NN's 0.8% edge has a CI that includes zero. The rating factors carry almost no severity signal at the 99.5% cap (€34,148; 133 claims capped, with a large-loss load of ×1.22 added back into the premium).

### Pure premium ranking

Ordered Lorenz curve on test, ranked by predicted pure premium per policy-year ([figure](reports/figures/lorenz_pure_premium.png)):

| Frequency × severity | Gini (± bootstrap sd) |
|---|---|
| GLM × Gamma GLM | 0.277 ± 0.026 |
| CANN × Gamma NN | 0.321 ± 0.025 |
| FFNN × Gamma NN | 0.333 ± 0.025 |
| LightGBM × Gamma GLM | 0.335 ± 0.027 |

### Embeddings (I3)

![Embeddings](reports/figures/embeddings.png)

- **VehBrand:** the embedding is meaningful. Distance between brands tracks the gap in their log claim frequency (Spearman 0.49 ± 0.11 over 5 seeds). B12, the high-frequency brand, sits apart, and B14 (lowest frequency) is the opposite outlier.
- **Region:** the embedding is mostly noise (Spearman 0.09 ± 0.06). Once Density and Area are in the model, region adds little.

### Interpretability

![Partial dependence](reports/figures/partial_dependence.png)

- **BonusMalus:** the GLM's log-linear term over-extrapolates. At BonusMalus = 150 it predicts 0.88 claims/year vs 0.49 for the FFNN.
- **DrivAge:** the GLM's step bins are visible. The FFNN is smooth but **under-predicts drivers aged 18–20** (≈0.08 vs ≈0.15 for the GLM and ≈0.19 for LightGBM). Partial dependence evaluates every policy at the same age, so a smooth tanh net with only ~1% of policies in that range flattens the spike.

### Q2 — Segmentation (I5): the VAE does not find better risk segments

Clusterers were fitted on train and are evaluated on test. VAE and raw-feature results are means over 3 seeds. Full sweep (k = 4..10) and segment profiles: [reports/segmentation_report.md](reports/segmentation_report.md).

| k = 6 | Max/min segment frequency | Deviance reduction vs one rate | Silhouette | Seed stability (ARI) |
|---|---|---|---|---|
| **GLM-predicted-frequency bands** | **4.79×** | **4.53%** | 0.46 | 1.00 (deterministic) |
| k-means on raw standardized features | 2.51× | 1.56% | 0.12 | 1.00 |
| VAE latent + k-means | 2.30× | 1.44% | 0.21 | 0.53 |
| VAE latent + GMM | 2.34× | 0.78% | 0.11 | 0.64 |

The answer to Q2 is **no**. The VAE is unsupervised: it compresses whatever explains the most variance in the rating factors (vehicle power, age, density), which is not what explains claims. Its clusters are also unstable across seeds (ARI ≈ 0.5). The supervised GLM bands spread risk twice as much at every k. The API still returns the VAE segment, which is useful for describing the portfolio but should not be used for pricing.

### Anomalies (I6)

The autoencoder's top 1% of test policies by reconstruction error are:
- older drivers (median 57 vs 44) with high BonusMalus (80 vs 50);
- mostly on rare brands (B14 ×12 over-represented, B5 ×4.7) and in small regions (Franche-Comté and Champagne-Ardenne ×11).

These policies have **twice the claim frequency** (0.204 vs 0.099). Agreement between the 3 autoencoder seeds on which policies are in that top 1% is low (Jaccard 0.25), so treat the flag as a soft signal.

## Failure analysis: where the models fail

Full tables: [reports/tables/failure_*.md](reports/tables).

1. **Short exposures (the largest failure).** Policies with ≤ 0.1 years of exposure have 3.1× more claims than predicted (A/E 3.14 for the FFNN, 3.31 for the GLM). Full-year policies have 21% fewer (A/E 0.79). Every model assumes claims scale proportionally with exposure, and this data breaks that: policies are likely cancelled or re-issued after a claim. All models share this failure, and modelling exposure non-linearly would be the next step.
2. **Multi-claim policies.** The 3.4K test policies with any claim account for 68% of the FFNN's deviance. For the 185 policies with 2+ claims the FFNN predicts 0.11 on average. The worst cases are policies like IDpol 3253234 (4 claims in 0.08 years, predicted 0.006) and IDpol 2247986, an apparent twin of it (3 claims, same region and brand, age 52 vs 53). These are unforecastable from rating factors.
3. **Young drivers.** The FFNN is calibrated on 18–20-year-olds overall (A/E 1.06), but its age curve is too smooth (see the partial dependence above). Ages 26–30 are over-predicted by about 9% (A/E 0.91).
4. **Severity.** No model ranks claim cost meaningfully (see above).
5. **Missing severities.** 27% of claims (9,658) have no amount in the severity table. On policies with complete severity data, predicted pure premium exceeds observed losses by ×1.30. That is the expected result if the missing amounts are random, but it cannot be verified.

## Deployment

The service is in `serve/`.
- **Endpoint:** FastAPI `POST /quote?model=nn|glm` returns:
  - expected frequency, expected claims, expected severity
  - pure premium, both per year and for the requested exposure
  - VAE risk segment with that segment's observed frequency
  - autoencoder anomaly flag and score
- **Validation:** a Pydantic schema enforces types, observed ranges and known categories. An unknown VehBrand or Region gets a 422 error that lists the allowed values. `GET /health` is also available.
- **Model choice:** the served frequency NN (FFNN with embeddings, 5-seed ensemble) was chosen by **validation** deviance, not test.

| Latency, single request, Apple M4 CPU, 1 thread, n = 1000 | p50 | p95 |
|---|---|---|
| HTTP end-to-end, NN (5 freq + 5 sev nets + AE + VAE) | 13.9 ms | 14.7 ms |
| HTTP end-to-end, GLM | 5.5 ms | 6.3 ms |
| In-process, NN | 12.7 ms | 13.2 ms |

- **Model size:** the whole serving bundle (all weights, GLM coefficients, preprocessing, segmenter) is **0.14 MB**. Latency is dominated by per-request pandas overhead, not by the networks.
- **Streamlit:** `make app` opens sliders for driver age, vehicle age and bonus-malus, with NN and GLM premiums side by side.
- **Docker:** `docker build -f serve/Dockerfile -t fremtpl-pricing .` *(Docker was not available on the dev machine, so the image has not been built or tested.)*

## Data decisions

Counts are from `data/processed/cleaning_report.json`.

- **Source:** the [mabilton/fremtpl2](https://huggingface.co/datasets/mabilton/fremtpl2) mirror. OpenML 41214 timed out. The tables are identical: 678,013 policies and 26,639 claim rows.
- **Consistency rule:** `ClaimNb` is kept exactly as published, so no frequency rows are dropped and the GLM stays comparable with Noll, Salzmann & Wüthrich (2018). Severity rows are used only for policies whose number of rows equals their `ClaimNb`.
  - 9,117 policies don't match; 9,116 of them have claims but no amounts (9,658 claims in total).
  - 195 orphan severity rows (6 policies missing from the frequency table) and 1 partial row are dropped, leaving 26,443 claims.
- **Caps:** as in the reference analysis: ClaimNb ≤ 4 (9 rows), Exposure ≤ 1 (1,224), VehAge ≤ 20 (8,322), DrivAge ≤ 90 (401), BonusMalus ≤ 150 (209). Density enters as log(Density).
- **Split:** 80/10/10 train/val/test, stratified on ClaimNb > 0, seed 2018. The test claim rate (0.1004) matches train (0.1007).
- **Exposure:** always enters as an offset or multiplier (an `init_score` for LightGBM, and the GLM skip connection for CANN), never as a feature.

## Reproducibility

- **Configs:** every run starts from a YAML file in `configs/` (15 experiment configs, all inheriting from `base.yaml`).
- **Seeds:** fixed seeds throughout. Neural nets and LightGBM use 5 seeds on the main split and 3 on the grouped split. The VAE and autoencoder use 3 seeds.
- **Tracking:** params, per-epoch validation metrics and final metrics are logged to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
- **NN setup:** Adam, learning rate 1e-3, batch 5,000 (1,000 for the 21K-claim severity net), up to 500 epochs, early stopping on validation deviance with patience 20.
- **Compute:** everything ran on a laptop CPU (Apple M4) in under an hour, so the RunPod budget was **$0**. The models are too small for a GPU to help. On RunPod the same `make all` command works unchanged.
- **Known macOS issue:** LightGBM and PyTorch ship separate `libomp` copies, and loading both in one process crashes PyTorch's multi-threaded code. Training processes therefore load only one of them, and `src/eval.py` loads LightGBM first and runs PyTorch single-threaded.

## Layout

```
configs/      one YAML per experiment (inherits base.yaml)
src/data/     load.py (download, consistency rule, caps, splits), features.py (GLM design, NN preprocessing), schema.py
src/models/   glm.py, gbm.py, nets.py (PyTorch modules), neural.py (training wrappers)
src/train.py  trains one config (train/val only)       src/eval.py   the only reader of the test split
src/segmentation.py  I5/I6                            src/export.py serving bundle (selected on validation)
serve/        app.py (FastAPI), schemas.py, predictor.py, streamlit_app.py, benchmark.py, Dockerfile
notebooks/    01_eda.ipynb: analysis only, train split only
reports/      figures/, tables/, segmentation_report.md, results.json
tests/        data rules, metrics vs sklearn, CANN-starts-at-GLM, exposure offset, save/load, API validation
```

## Limitations

- **Fairness / regulation:** the data has no protected attributes, but **DrivAge** (age) and **Region** (a geographic proxy for income and ethnicity) can act as proxies. Insurance pricing is regulated; for example, the EU's 2012 Test-Achats ruling bans gender-based premiums, and age and geography rating face scrutiny. A real deployment would need a proxy-discrimination review, and the young-driver behaviour shown above deserves one.
- **One portfolio, one year, one country.** There is no time-based split, so the results say nothing about drift.
- **Uneven tuning effort.** LightGBM received a small hand-chosen configuration and the NNs use the literature architecture. A larger search could reorder LightGBM and the NNs, but probably not the GLM.
- **Missing severities.** The assumption that the 27% of claims without amounts are missing at random is untestable here.
- **Pure premium only.** There are no expense, profit or reinsurance loadings, and large losses are handled with a single flat load.
