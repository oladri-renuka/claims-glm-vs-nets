#!/usr/bin/env bash
# Train every experiment config. Deterministic baselines first (CANN needs the frozen GLM,
# freq_lgbm_random reuses freq_lgbm's search result), then the seeded models JOBS at a time.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
JOBS=${JOBS:-5}
run() { python3 -m src.train --config "configs/$1.yaml" > "logs/$1.log" 2>&1 && echo "DONE $1" || echo "FAIL $1"; }
export -f run
for c in freq_homogeneous freq_glm freq_homogeneous_random freq_glm_random sev_glm; do run "$c"; done
# Long jobs first so they overlap with everything else.
printf '%s\n' ae vae freq_lgbm freq_cann freq_ffnn_emb freq_ffnn_onehot sev_nn \
  freq_cann_random freq_ffnn_emb_random | xargs -P "$JOBS" -I{} bash -c 'run {}'
run freq_lgbm_random
