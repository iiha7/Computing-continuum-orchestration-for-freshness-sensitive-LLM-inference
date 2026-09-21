"""Fit response-size predictors and calibrate edge decision-support reliability.

The edge LLM is not trusted to self-report confidence.  This script fits a
calibrated reliability model from observable edge-profile features and rewrites
``data/llm_profiles/edge_outputs.jsonl`` so that the legacy ``confidence`` field
means P(edge decision-support response is acceptable), not LLM self-confidence.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .common import ensure_dir, load_yaml, project_path, read_jsonl, save_json, write_jsonl

FILES = {"edge": "edge_outputs.jsonl", "cloud": "cloud_outputs.jsonl"}


def _binary_entropy(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, float(p)))
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def load_profiles() -> pd.DataFrame:
    rows: List[Dict] = []
    for mode, fn in FILES.items():
        p = project_path("data/llm_profiles", fn)
        if p.exists():
            rows.extend(read_jsonl(p))
    if not rows:
        raise FileNotFoundError("No LLM profiles found in data/llm_profiles. Run src.profile_llms_real first.")
    return pd.DataFrame(rows)


def response_size_predictor(quantile=None) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["complexity", "mode_role"]),
        ("num", "passthrough", ["prompt_bytes", "criticality", "deadline_s"]),
    ])
    reg = (RandomForestRegressor(n_estimators=200, random_state=1, min_samples_leaf=2)
           if quantile is None else
           GradientBoostingRegressor(loss="quantile", alpha=quantile, random_state=1))
    return Pipeline([("pre", pre), ("reg", reg)])


def reliability_features() -> List[str]:
    return [
        "complexity", "window_len", "prompt_bytes", "computed_danger_class",
        "response_valid", "latency_s", "output_tokens", "response_bytes",
        "risk_factor_f1", "escalation_correct", "urgency_correct", "verification_correct",
    ]


def fit_edge_reliability(edge_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Replace edge confidence with calibrated P(edge acceptable | observable features)."""
    split = pd.read_csv(project_path("data/processed/train_val_test_split.csv"))
    prompts = pd.read_csv(project_path("data/processed/prompts.csv"))[["prompt_id", "window_len", "iot_transmit", "computed_danger_class"]]
    df = edge_df.merge(split, on="record_id", how="left").merge(prompts, on="prompt_id", how="left", suffixes=("", "_prompt"))

    df["response_valid"] = pd.to_numeric(df.get("response_valid", 1), errors="coerce").fillna(0).astype(int)
    df["window_len"] = pd.to_numeric(df.get("window_len", 1), errors="coerce").fillna(1).astype(int)
    df["computed_danger_class"] = pd.to_numeric(df.get("computed_danger_class", df.get("computed_danger_class_prompt", 0)), errors="coerce").fillna(0).astype(int)
    for c in ["prompt_bytes", "latency_s", "output_tokens", "response_bytes", "response_quality", "risk_factor_f1"]:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)
    for c in ["correct", "escalation_correct", "urgency_correct", "verification_correct"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)
    df["iot_transmit"] = pd.to_numeric(df.get("iot_transmit", 1), errors="coerce").fillna(1).astype(int)

    # Reliability target: acceptable structured response, already stored as correct.
    train = df[(df["split"] == "train") & (df["iot_transmit"] == 1)].copy()
    if len(train) < 20 or train["correct"].nunique() < 2:
        train = df[(df["split"].isin(["train", "val"])) & (df["iot_transmit"] == 1)].copy()

    feat = reliability_features()
    if len(train) >= 10 and train["correct"].nunique() >= 2:
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["complexity"]),
            ("num", "passthrough", [x for x in feat if x != "complexity"]),
        ])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        model = Pipeline([("pre", pre), ("clf", clf)])
        model.fit(train[feat], train["correct"])
        prob = model.predict_proba(df[feat])[:, 1]
        joblib.dump({"model": model, "feature_cols": feat, "note": "P(edge response_quality >= threshold | observable edge profile features)"},
                    outdir / "edge_reliability_model.pkl")
        method = "logistic_reliability_model"
    else:
        prior = float(train["correct"].mean()) if len(train) else float(df["correct"].mean())
        prob = np.full(len(df), prior, dtype=float)
        save_json({"prior": prior, "feature_cols": feat, "note": "fallback prior; not enough data to fit reliability model"},
                  outdir / "edge_reliability_model.json")
        method = "fallback_prior"

    prob = np.asarray(prob, dtype=float)
    prob[df["response_valid"].to_numpy() == 0] = 0.0
    prob = np.clip(prob, 0.0, 1.0)

    df["calibrated_edge_confidence"] = prob
    df["confidence"] = prob
    df["edge_margin"] = prob
    df["edge_entropy"] = [_binary_entropy(p) for p in prob]
    df["confidence_source"] = "calibrated_edge_decision_support_reliability_not_self_reported"

    keep_original_cols = list(edge_df.columns)
    for c in ["response_valid", "self_reported_confidence", "edge_margin", "edge_entropy", "confidence", "calibrated_edge_confidence", "confidence_source"]:
        if c not in keep_original_cols:
            keep_original_cols.append(c)
    out_cols = [c for c in keep_original_cols if c in df.columns]
    calibrated = df[out_cols].where(pd.notna(df[out_cols]), None)

    edge_path = project_path("data/llm_profiles/edge_outputs.jsonl")
    backup_path = project_path("data/llm_profiles/edge_outputs_uncalibrated_backup.jsonl")
    if edge_path.exists() and not backup_path.exists():
        shutil.copyfile(edge_path, backup_path)
    write_jsonl(calibrated.to_dict(orient="records"), edge_path)

    save_json({
        "method": method,
        "num_training_rows": int(len(train)),
        "train_acceptable_rate_observed": float(train["correct"].mean()) if len(train) else None,
        "train_mean_response_quality": float(train["response_quality"].mean()) if len(train) else None,
        "mean_calibrated_edge_confidence": float(np.mean(prob)),
        "note": "The confidence field in edge_outputs.jsonl is calibrated reliability, not LLM self-confidence.",
    }, outdir / "edge_reliability_summary.json")
    return calibrated


def select_early_exit_threshold(edge_cal: pd.DataFrame, cloud_df: pd.DataFrame, outdir: Path) -> None:
    cfg = load_yaml(project_path("configs/experiment.yaml"))
    ee_cfg = cfg.get("baselines", {}).get("early_exit", {})
    safe_max = int(ee_cfg.get("safe_class_max", 1))
    lambda_cloud = float(ee_cfg.get("lambda_cloud_usage", 0.02))
    grid = ee_cfg.get("threshold_grid", [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)])

    split = pd.read_csv(project_path("data/processed/train_val_test_split.csv"))
    prompts = pd.read_csv(project_path("data/processed/prompts.csv"))[["prompt_id", "iot_transmit", "computed_danger_class"]]
    edge = edge_cal.merge(split, on="record_id", how="left").merge(prompts, on="prompt_id", how="left", suffixes=("", "_prompt"))
    cloud = cloud_df[["prompt_id", "correct", "response_quality"]].rename(columns={"correct": "cloud_correct", "response_quality": "cloud_quality"})
    df = edge.merge(cloud, on="prompt_id", how="left")
    val = df[(df["split"] == "val") & (df["iot_transmit"] == 1)].copy()
    if val.empty:
        val = df[(df["split"].isin(["train", "val"])) & (df["iot_transmit"] == 1)].copy()
    if val.empty:
        save_json({"confidence_threshold": float(ee_cfg.get("confidence_threshold", 0.70)), "note": "fallback; no validation rows"},
                  outdir / "early_exit_threshold.json")
        return
    val["cloud_quality"] = pd.to_numeric(val["cloud_quality"], errors="coerce").fillna(0.0)
    val["response_quality"] = pd.to_numeric(val["response_quality"], errors="coerce").fillna(0.0)
    val["computed_danger_class"] = pd.to_numeric(val.get("computed_danger_class", val.get("computed_danger_class_prompt", 0)), errors="coerce").fillna(0).astype(int)
    val["calibrated_edge_confidence"] = pd.to_numeric(val["calibrated_edge_confidence"], errors="coerce").fillna(0.0)

    rows = []
    best = None
    for th in grid:
        th = float(th)
        # Risk-conservative early-exit baseline: only accept edge when calibrated
        # reliability is high and the computed FWI class is within the safe range.
        accept = (val["calibrated_edge_confidence"] >= th) & (val["computed_danger_class"] <= safe_max)
        final_quality = np.where(accept, val["response_quality"], val["cloud_quality"])
        cloud_used = 1.0 - accept.astype(float)
        score = float(np.mean(final_quality) - lambda_cloud * np.mean(cloud_used))
        row = {"threshold": th, "score": score, "response_quality": float(np.mean(final_quality)),
               "cloud_usage": float(np.mean(cloud_used)), "edge_accept_rate": float(np.mean(accept))}
        rows.append(row)
        if best is None or (score > best["score"] + 1e-12) or (abs(score - best["score"]) <= 1e-12 and row["cloud_usage"] < best["cloud_usage"]):
            best = row
    save_json({"selected": best, "grid": rows, "safe_class_max": safe_max,
               "lambda_cloud_usage": lambda_cloud,
               "note": "Threshold selected on validation prompts using calibrated edge reliability and structured response quality; no self-confidence used."},
              outdir / "early_exit_threshold.json")


def main():
    outdir = ensure_dir(project_path("data/fitted"))
    profiles = load_profiles()
    prompts = pd.read_csv(project_path("data/processed/prompts.csv"))[["prompt_id", "criticality", "deadline_s"]]
    df = profiles.merge(prompts, on="prompt_id", how="left")

    features = ["complexity", "mode_role", "prompt_bytes", "criticality", "deadline_s"]
    mean_model = response_size_predictor(None).fit(df[features], df["response_bytes"])
    p90_model = response_size_predictor(0.90).fit(df[features], df["response_bytes"])
    joblib.dump({"mean": mean_model, "p90": p90_model, "feature_cols": features}, outdir / "response_size_predictor.pkl")

    latency, quality = {}, {}
    for (mode, comp), g in df.groupby(["mode_role", "complexity"]):
        key = f"{mode}/{comp}"
        latency[key] = {
            "mean_latency_s": float(g.latency_s.mean()),
            "p90_latency_s": float(g.latency_s.quantile(0.90)),
            "mean_response_bytes": float(g.response_bytes.mean()),
            "p90_response_bytes": float(g.response_bytes.quantile(0.90)),
        }
        quality[key] = {
            "acceptable_rate": float(g.correct.mean()),
            "mean_response_quality": float(pd.to_numeric(g.get("response_quality", 0.0), errors="coerce").fillna(0.0).mean()),
        }
    save_json(latency, outdir / "latency_tables.json")
    save_json(quality, outdir / "quality_tables.json")
    save_json(quality, outdir / "accuracy_tables.json")  # compatibility name
    save_json({
        "max_prompt_bytes": int(df.prompt_bytes.max()),
        "max_response_bytes": int(df.response_bytes.max()),
        "max_latency_s": float(df.latency_s.max()),
        "max_cost_usd": float(max(df.cost_usd.max(), 1e-9)),
    }, outdir / "normalization_stats.json")

    edge = profiles[profiles["mode_role"] == "edge"].copy()
    cloud = profiles[profiles["mode_role"] == "cloud"].copy()
    if not edge.empty:
        edge_cal = fit_edge_reliability(edge, outdir)
        if not cloud.empty:
            select_early_exit_threshold(edge_cal, cloud, outdir)
    print(f"Wrote fitted artifacts to {outdir}")
    print("Edge confidence, if present, is now calibrated decision-support reliability; LLM self-confidence is not used.")


if __name__ == "__main__":
    main()
