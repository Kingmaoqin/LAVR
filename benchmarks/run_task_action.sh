#!/usr/bin/env bash
# Task-correctness action value on Nemotron-Diffusion-3B (GSM8K C/D, SVAMP). Needs transformers>=5.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/collect_task_labels.py --dataset gsm8k --n_screen 260 --n_prompts 180 --offset 440 --K 8 --n_cand 6 --tag taskC
python scripts/collect_task_labels.py --dataset gsm8k --n_screen 600 --n_prompts 570 --offset 700 --K 8 --n_cand 6 --mixed_frac 0.7 --tag taskD
python analysis/qualify_svamp.py
python scripts/collect_task_labels.py --dataset svamp --n_screen 300 --n_prompts 200 --offset 0 --K 8 --n_cand 6 --mixed_frac 0.7 --tag taskE_svamp --shard_model
python analysis/validate_task_collection.py --tag taskD --expected-docs 570 --no-overlap-tag taskC
python analysis/validate_task_collection.py --tag taskE_svamp --expected-docs 200

# cross-fitted concordance gains (rank-4 readout, 5 folds)
python analysis/taskC_crossfit_positive.py --tags taskC taskD --out analysis/results/taskCD_crossfit_confirmatory.json
python analysis/taskC_crossfit_positive.py --tag taskE_svamp  --out analysis/results/taskE_svamp_crossfit.json
# frozen transfer C -> D
python analysis/taskD_fit_frozen_primary.py
python analysis/taskD_apply_frozen_primary.py --tag taskD
# Path-LL vs task-correctness alignment, state readout on task populations
python analysis/task_utility_analysis.py --tags taskC taskE_svamp
python scripts/estimate_all.py --task --seeds 50 --out data/estimate_task.json
python scripts/task_state_readout.py --seeds 50
