
"""Fallback TinyML-style edge profiler for smoke tests.

Final LLM-orchestration experiments should use ``src.profile_llms_real`` with
the compact edge LLM default ``qwen2.5:0.5b``.  This script is kept only as a
fast/debug fallback that writes the same edge_outputs.jsonl schema without
running a local LLM.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .common import ensure_dir, project_path, set_seed, write_jsonl

BASE = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]


def feature_frame(prompts: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in BASE:
        cc = f"current_{c}"
        if cc in prompts.columns:
            cols.append(cc)
    for c in BASE:
        tc = f"trend_{c}"
        if tc in prompts.columns:
            cols.append(tc)
    extra = ["window_len", "prompt_bytes", "criticality"]
    cols.extend([c for c in extra if c in prompts.columns])
    return prompts[cols].astype(float).fillna(0.0), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", choices=["tree", "logreg"], default="tree")
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()
    set_seed(args.seed)

    prompts = pd.read_csv(project_path("data/processed/prompts.csv"))
    split = pd.read_csv(project_path("data/processed/train_val_test_split.csv"))
    train_ids = set(split.loc[split.split == "train", "record_id"].tolist())
    train = prompts[prompts.record_id.isin(train_ids)].copy()
    X_all, cols = feature_frame(prompts)
    X_train = X_all.loc[train.index]
    y_train = train["ground_truth_tier"].astype(int)

    if args.model == "tree":
        base = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=3, random_state=args.seed)
    else:
        base = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=0.75, multi_class="auto", random_state=args.seed)),
        ])
    # Calibrate confidence when possible.  For tiny datasets/classes, fall back
    # to the base model.
    try:
        model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        model.fit(X_train, y_train)
    except Exception:
        model = base.fit(X_train, y_train)

    pred = model.predict(X_all).astype(int)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_all)
        classes = list(getattr(model, "classes_", sorted(prompts.ground_truth_tier.unique())))
        # Map probability rows to all class ids 0..5.
        full = np.zeros((len(prompts), 6), dtype=float)
        for j, c in enumerate(classes):
            if int(c) < 6:
                full[:, int(c)] = proba[:, j]
        conf = full.max(axis=1)
        sortedp = np.sort(full, axis=1)
        margin = sortedp[:, -1] - sortedp[:, -2]
        entropy = -np.sum(np.where(full > 0, full * np.log(full + 1e-12), 0.0), axis=1)
    else:
        conf = np.full(len(prompts), 0.60)
        margin = np.full(len(prompts), 0.20)
        entropy = np.full(len(prompts), 1.0)

    rng = np.random.default_rng(args.seed + 99)
    rows = []
    for k, (_, p) in enumerate(prompts.iterrows()):
        comp = p["complexity"]
        # Tiny edge latency grows slightly with window length but stays small.
        lat_base = {"simple": 0.035, "medium": 0.055, "complex": 0.080}[comp]
        latency_s = float(max(0.01, rng.lognormal(np.log(lat_base), 0.15)))
        text = f"DANGER_CLASS={int(pred[k])} CONFIDENCE={float(conf[k]):.2f}"
        response_bytes = len(text.encode())
        rows.append({
            "prompt_id": p["prompt_id"],
            "record_id": int(p["record_id"]),
            "complexity": comp,
            "mode_role": "edge",
            "model_name": f"tiny_{args.model}_depth{args.max_depth}",
            "prompt_bytes": int(p["prompt_bytes"]),
            "response_text": text,
            "predicted_tier": int(pred[k]),
            "ground_truth_tier": int(p["ground_truth_tier"]),
            "predicted_danger_class": int(pred[k]),
            "ground_truth_danger_class": int(p["ground_truth_tier"]),
            "correct": int(pred[k] == int(p["ground_truth_tier"])),
            "response_bytes": int(response_bytes),
            "input_tokens": max(1, int(p["prompt_bytes"]) // 4),
            "output_tokens": max(1, response_bytes // 4),
            "latency_s": latency_s,
            "ttft_s": min(latency_s, 0.01),
            "cost_usd": 0.0,
            "confidence": float(np.clip(conf[k], 0.01, 0.999)),
            "edge_margin": float(np.clip(margin[k], 0.0, 1.0)),
            "edge_entropy": float(entropy[k]),
        })
    outdir = ensure_dir(project_path("data/llm_profiles"))
    write_jsonl(rows, outdir / "edge_outputs.jsonl")
    print(f"Wrote constrained edge profile to {outdir/'edge_outputs.jsonl'} using {args.model}; features={cols}")


if __name__ == "__main__":
    main()
