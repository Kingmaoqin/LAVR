"""Strict completion validator for Task-D/Task-E candidate-label datasets."""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expected-docs", type=int, required=True)
    ap.add_argument("--states-per-doc", type=int, default=4)
    ap.add_argument("--candidates-per-state", type=int, default=6)
    ap.add_argument("--no-overlap-tag")
    ap.add_argument("--write-meta", action="store_true")
    args = ap.parse_args()

    outdir = os.path.join(ROOT, "data", "labels_" + args.tag)
    shards = sorted(glob.glob(os.path.join(outdir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shards in {outdir}")
    d = D.load_labels([args.tag])
    doc = d["doc_id"].astype(np.int64)
    state = d["state_id"].astype(np.int64)
    docs, doc_counts = np.unique(doc, return_counts=True)
    states, state_counts = np.unique(state, return_counts=True)
    expected_states = args.expected_docs * args.states_per_doc
    expected_rows = expected_states * args.candidates_per_state

    errors = []
    if len(docs) != args.expected_docs:
        errors.append(f"docs {len(docs)} != {args.expected_docs}")
    expected_doc_rows = args.states_per_doc * args.candidates_per_state
    if not np.all(doc_counts == expected_doc_rows):
        bad = docs[doc_counts != expected_doc_rows]
        errors.append(f"{len(bad)} docs do not have {expected_doc_rows} rows")
    if len(states) != expected_states:
        errors.append(f"states {len(states)} != {expected_states}")
    if not np.all(state_counts == args.candidates_per_state):
        errors.append(f"{int(np.sum(state_counts != args.candidates_per_state))} "
                      "states have wrong candidate count")
    if len(doc) != expected_rows:
        errors.append(f"rows {len(doc)} != {expected_rows}")

    finite_keys = [k for k in (
        "A_task", "A_task_seeds", "A_task_sem", "Q_reward", "V_reward",
        "A_pertok", "A_full_seeds", "A_sem", "V_pertok", "logp_action",
        "C1", "C2", "C3", "H_i", "H_g") if k in d]
    nonfinite = {k: int(np.size(d[k]) - np.isfinite(d[k]).sum())
                 for k in finite_keys if not np.isfinite(d[k]).all()}
    if nonfinite:
        errors.append(f"nonfinite values: {nonfinite}")

    overlap = []
    if args.no_overlap_tag:
        other = D.load_labels([args.no_overlap_tag])
        overlap = sorted(set(map(int, docs)) & set(map(int, other["doc_id"])))
        if overlap:
            errors.append(f"overlap with {args.no_overlap_tag}: {overlap[:10]}")

    # Detect accidental duplicate documents across shard files directly; the
    # concatenated row counts alone would not identify this provenance defect.
    owners = {}
    duplicate_shard_docs = {}
    for path in shards:
        with np.load(path) as z:
            for value in np.unique(z["doc_id"]):
                value = int(value)
                if value in owners:
                    duplicate_shard_docs.setdefault(value, [owners[value]]).append(
                        os.path.basename(path))
                else:
                    owners[value] = os.path.basename(path)
    if duplicate_shard_docs:
        errors.append(f"documents repeated across shards: "
                      f"{list(duplicate_shard_docs)[:10]}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "tag": args.tag,
        "n_shards": len(shards),
        "n_docs": int(len(docs)),
        "n_states": int(len(states)),
        "n_rows": int(len(doc)),
        "states_per_doc": args.states_per_doc,
        "candidates_per_state": args.candidates_per_state,
        "finite_keys": finite_keys,
        "nonfinite": nonfinite,
        "no_overlap_tag": args.no_overlap_tag,
        "overlap": overlap,
        "duplicate_shard_docs": duplicate_shard_docs,
        "errors": errors,
    }
    qc_path = os.path.join(outdir, "validation.json")
    with open(qc_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)

    if args.write_meta:
        worker_meta = sorted(glob.glob(os.path.join(outdir, "meta_worker*.json")))
        single_meta = os.path.join(outdir, "meta.json")
        source_meta = [json.load(open(p)) for p in worker_meta]
        # A single-worker collector writes meta.json itself; preserve its full
        # run config while adding the independently validated canonical counts.
        if not source_meta and os.path.exists(single_meta):
            old_meta = json.load(open(single_meta))
            source_meta = (old_meta.get("source_worker_meta", [])
                           if old_meta.get("status") == "validated_complete"
                           else [old_meta])
        canonical = {
            "status": "validated_complete",
            "tag": args.tag,
            "n_docs": int(len(docs)),
            "n_states": int(len(states)),
            "n_examples": int(len(doc)),
            "validation": os.path.basename(qc_path),
            "source_worker_meta": source_meta,
        }
        with open(single_meta, "w") as f:
            json.dump(canonical, f, indent=2)
        print("wrote", single_meta)


if __name__ == "__main__":
    main()
