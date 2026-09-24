#!/usr/bin/env bash
# Global state value on Nemotron-Diffusion-3B / GSM8K (800 prompts). Needs transformers>=5.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/phase_s_nemotron.py
python scripts/collect_v_readout.py --n_prompts 400 --offset 0   --K 8 --tag vreadA
python scripts/collect_v_readout.py --n_prompts 400 --offset 400 --K 8 --tag vreadB
python scripts/analyze_v_readout.py vreadA vreadB     # R2 / AUC vs output-statistics baseline
python scripts/audit_v_confidence.py                   # prompt-cluster bootstrap CIs, selective generation
