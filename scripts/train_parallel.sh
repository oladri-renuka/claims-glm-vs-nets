#!/usr/bin/env bash
# Train every experiment config. Deterministic baselines first (CANN needs the frozen GLM),
# then the seeded models JOBS at a time (each NN uses 2 threads).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
JOBS=${JOBS:-5}
run() { python3 -m src.train --config "configs/$1.yaml" > "logs/$1.log" 2>&1 && echo "DONE $1" || echo "FAIL $1"; }
export -f run
for c in freq_homogeneous freq_glm freq_homogeneous_grouped freq_glm_grouped sev_glm; do run "$c"; done
printf '%s\n' freq_cann freq_ffnn_emb freq_ffnn_onehot freq_lgbm sev_nn vae ae \
  freq_cann_grouped freq_ffnn_emb_grouped freq_lgbm_grouped | xargs -P "$JOBS" -I{} bash -c 'run {}'
