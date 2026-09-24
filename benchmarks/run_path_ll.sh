#!/usr/bin/env bash
# Path-LL local action value on MDLM / SEDD (OpenWebText windows).
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--n_prompts 200 --K 24 --n_cand 6 --n_cand_conf 3 --horizon 16"

python scripts/phase_s_qualify.py          # substrate checks (MDLM)
python scripts/phase_s_sedd.py             # substrate checks (SEDD)
python scripts/pipeline_selftest.py        # synthetic positive/negative control

# original study: three model-policy conditions x 400 documents
python scripts/collect_labels.py $COMMON --offset 0   --tag a3 --order ancestral
python scripts/collect_labels.py $COMMON --offset 200 --tag b3 --order ancestral
python scripts/collect_labels.py $COMMON --offset 0   --tag c3 --order confidence --order_temp 2.0
python scripts/collect_labels.py $COMMON --offset 200 --tag d3 --order confidence --order_temp 2.0
python scripts/collect_labels.py $COMMON --offset 0   --tag s1 --backbone sedd
python scripts/collect_labels.py $COMMON --offset 200 --tag s2 --backbone sedd

# per-seed V labels for noise ceilings
python scripts/backfill_v_seeds.py --tags a3 b3 c3 d3 s1 s2

# readouts, controls and uncertainty
python scripts/label_diagnostics.py --tags a3 b3
python scripts/exp1_decodability.py --tags a3 b3 --out results/exp1_anc
python scripts/exp1b_within_state.py A_pertok a3,b3
python scripts/exp2_matched.py --tags a3 b3
python scripts/audit_A_fairtest.py
python analysis/fairtest_corrected_ci.py
python scripts/estimate_all.py --seeds 50
python analysis/label_reliability_ceiling.py
python analysis/horizon_timestep_layer.py
python analysis/stratified_concordance.py
python analysis/r2_stats.py

# replication A: fresh documents, frozen protocol
python scripts/collect_labels.py $COMMON --offset 400 --tag freshA --order ancestral
python analysis/confirm_fresh.py --old_tags a3 b3 --fresh_tag freshA

# estimator / probe validity checks
python analysis/toy_exact.py
python analysis/synthetic_tests.py
