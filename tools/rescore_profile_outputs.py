"""Re-parse and re-score saved LLM profile JSONL files without re-running LLMs.

Use this when the parser/scoring rules change but the raw `response_text` values
are already available in data/llm_profiles/edge_outputs.jsonl or cloud_outputs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path

ALLOWED_ESCALATION = {"low", "moderate", "high", "critical"}
ALLOWED_ACTION = {
    "normal_monitoring",
    "increase_sampling",
    "send_warning",
    "alert_response_team",
    "urgent_escalation",
}
ALLOWED_URGENCY = {"normal", "time_sensitive", "urgent", "critical"}
ALLOWED_FACTORS = {
    "no_rain",
    "low_humidity",
    "high_temperature",
    "high_wind",
    "dry_fuel",
    "high_fwi",
    "increasing_fwi",
    "high_isi",
    "high_ffmc",
    "high_dmc",
    "high_dc",
    "high_bui",
    "decreasing_humidity",
    "increasing_wind",
}
ALIASES = {
    "low relative humidity": "low_humidity",
    "low_rh": "low_humidity",
    "high temperature": "high_temperature",
    "high temp": "high_temperature",
    "high wind": "high_wind",
    "no rain": "no_rain",
    "dry fuel": "dry_fuel",
    "dry fuels": "dry_fuel",
    "high fwi": "high_fwi",
    "increasing fwi": "increasing_fwi",
    "high isi": "high_isi",
    "time sensitive": "time_sensitive",
    "alert response team": "alert_response_team",
    "urgent escalation": "urgent_escalation",
    "increase sampling": "increase_sampling",
    "send warning": "send_warning",
    "normal monitoring": "normal_monitoring",
}


def normalize_label(x):
    if x is None:
        return None
    x = str(x).strip().lower().strip('"\'` ')
    x = x.replace("-", "_")
    x = re.sub(r"\s+", " ", x)
    if "|" in x:
        return None
    if x in ALIASES:
        return ALIASES[x]
    x2 = x.replace(" ", "_")
    if x2 in ALIASES:
        return ALIASES[x2]
    return x2


def extract_json(text):
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    match = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def normalize_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return None


def normalize_factor_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]
    if not isinstance(x, list):
        return []
    out = []
    for item in x:
        v = normalize_label(item)
        if v in ALLOWED_FACTORS and v not in out:
            out.append(v)
    return out


def f1_score(pred, ref):
    pred = set(pred or [])
    ref = set(ref or [])
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    return 2.0 * tp / (len(pred) + len(ref))


def rescore_row(row):
    obj = extract_json(row.get("response_text", ""))
    if not isinstance(obj, dict):
        row["response_valid"] = 0
        row["pred_escalation_risk"] = None
        row["pred_dominant_risk_factors"] = []
        row["pred_recommended_action"] = None
        row["pred_urgency"] = None
        row["pred_verification_need"] = None
        row["response_quality"] = 0.0
        row["correct"] = 0
        return row

    esc = normalize_label(obj.get("escalation_risk"))
    action = normalize_label(obj.get("recommended_action"))
    urgency = normalize_label(obj.get("urgency"))
    verification = normalize_bool(obj.get("verification_need"))
    factors = normalize_factor_list(obj.get("dominant_risk_factors"))

    row["pred_escalation_risk"] = esc if esc in ALLOWED_ESCALATION else None
    row["pred_dominant_risk_factors"] = factors
    row["pred_recommended_action"] = action if action in ALLOWED_ACTION else None
    row["pred_urgency"] = urgency if urgency in ALLOWED_URGENCY else None
    row["pred_verification_need"] = verification

    response_valid = int(
        row["pred_escalation_risk"] is not None
        and row["pred_recommended_action"] is not None
        and row["pred_urgency"] is not None
        and row["pred_verification_need"] is not None
    )
    row["response_valid"] = response_valid

    if response_valid:
        row["escalation_correct"] = int(row["pred_escalation_risk"] == row.get("ref_escalation_risk"))
        row["risk_factor_f1"] = f1_score(row["pred_dominant_risk_factors"], row.get("ref_dominant_risk_factors", []))
        row["action_correct"] = int(row["pred_recommended_action"] == row.get("ref_recommended_action"))
        row["urgency_correct"] = int(row["pred_urgency"] == row.get("ref_urgency"))
        row["verification_correct"] = int(row["pred_verification_need"] == row.get("ref_verification_need"))
        row["response_quality"] = (
            0.25 * row["escalation_correct"]
            + 0.25 * row["risk_factor_f1"]
            + 0.25 * row["action_correct"]
            + 0.15 * row["urgency_correct"]
            + 0.10 * row["verification_correct"]
        )
    else:
        row["escalation_correct"] = 0
        row["risk_factor_f1"] = 0.0
        row["action_correct"] = 0
        row["urgency_correct"] = 0
        row["verification_correct"] = 0
        row["response_quality"] = 0.0

    threshold = float(row.get("quality_threshold", 0.75))
    row["correct"] = int(row["response_quality"] >= threshold)
    row["self_reported_confidence"] = None
    if row.get("mode_role") == "edge":
        row["confidence"] = 0.0
        row["calibrated_edge_confidence"] = 0.0
        row["edge_margin"] = 0.0
        row["edge_entropy"] = math.log(2.0)
    return row


def summarize(rows):
    n = len(rows)
    if n == 0:
        return
    def mean(key):
        return sum(float(r.get(key, 0.0) or 0.0) for r in rows) / n
    print(f"Rows: {n}")
    for k in ["response_valid", "response_quality", "correct", "escalation_correct", "risk_factor_f1", "action_correct", "urgency_correct", "verification_correct"]:
        print(f"{k}: {mean(k):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()
    in_path = Path(args.input)
    if args.inplace:
        out_path = in_path
        backup = in_path.with_suffix(in_path.suffix + f".bak_{int(time.time())}")
        shutil.copy2(in_path, backup)
        print(f"Backup written to {backup}")
    else:
        if args.output is None:
            raise ValueError("Use --output or --inplace.")
        out_path = Path(args.output)
    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(rescore_row(json.loads(line)))
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote rescored file to {out_path}")
    summarize(rows)


if __name__ == "__main__":
    main()
