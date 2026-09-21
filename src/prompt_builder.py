"""Build shared wildfire decision-support prompts for edge/cloud LLMs.

The FWI danger class is computed by the IoT/preprocessing layer and is provided
as context.  The LLM task is no longer to reproduce a threshold lookup.  Instead,
both the compact edge LLM and the larger cloud LLM receive the SAME prompt and
produce a structured decision-support report.  The edge/cloud difference comes
from model capacity and latency, not from different prompts.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import pandas as pd

from .common import ensure_dir, project_path

FEATURES = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
CLASS_LABELS = {
    0: "Low", 1: "Moderate", 2: "High", 3: "Very High", 4: "Extreme", 5: "Very Extreme"
}
ESCALATION_CHOICES = ["low", "moderate", "high", "critical"]
ACTION_CHOICES = ["normal_monitoring", "increase_sampling", "send_warning", "alert_response_team", "urgent_escalation"]
URGENCY_CHOICES = ["normal", "time_sensitive", "urgent", "critical"]


def row_context(r: pd.Series) -> str:
    """Compact sensor context. FWI is included as a computed field/context."""
    return (
        f"Temperature={r['Temperature']} C, RH={r['RH']} %, Wind={r['Ws']} km/h, "
        f"Rain={r['Rain']} mm, FFMC={r['FFMC']}, DMC={r['DMC']}, DC={r['DC']}, "
        f"ISI={r['ISI']}, BUI={r['BUI']}, FWI={r['FWI']}"
    )


def window_context(win: pd.DataFrame) -> str:
    lines = []
    n = len(win)
    for k, (_, r) in enumerate(win.iterrows(), start=1):
        marker = "current" if k == n else f"t-{n-k}"
        lines.append(f"{marker}: {row_context(r)}")
    return "\n".join(lines)


def trend_features(win: pd.DataFrame) -> dict:
    if len(win) <= 1:
        return {f"trend_{c}": 0.0 for c in FEATURES}
    first, last = win.iloc[0], win.iloc[-1]
    return {f"trend_{c}": float(last[c]) - float(first[c]) for c in FEATURES}


def _risk_factors(current: pd.Series, trends: Dict[str, float]) -> List[str]:
    factors: List[str] = []
    if float(current["RH"]) <= 35:
        factors.append("low_humidity")
    if float(current["Temperature"]) >= 32:
        factors.append("high_temperature")
    if float(current["Ws"]) >= 15:
        factors.append("high_wind")
    if float(current["Rain"]) <= 0.1:
        factors.append("no_rain")
    if float(current["FWI"]) >= 21.3:
        factors.append("high_fwi")
    if float(trends.get("trend_FWI", 0.0)) >= 2.0:
        factors.append("increasing_fwi")
    if float(current["BUI"]) >= 40 or float(current["DMC"]) >= 30 or float(current["DC"]) >= 100:
        factors.append("dry_fuel")
    if float(current["ISI"]) >= 7:
        factors.append("high_isi")
    return factors


def reference_decision(win: pd.DataFrame) -> Dict:
    """Rule-derived decision-support reference for scoring LLM outputs.

    These rules are intentionally simple and deterministic.  They create a
    reproducible decision-support target from the current FWI class and recent
    trends, while leaving the LLM to produce the structured report.
    """
    current = win.iloc[-1]
    trends = trend_features(win)
    dc = int(current["danger_class"])
    fwi_delta = float(trends.get("trend_FWI", 0.0))
    rh_delta = float(trends.get("trend_RH", 0.0))
    wind_delta = float(trends.get("trend_Ws", 0.0))
    drying_trend = (rh_delta <= -5.0 and wind_delta >= 2.0)

    if dc >= 4 or (dc >= 3 and fwi_delta >= 5.0):
        escalation = "critical"
    elif dc >= 2 or fwi_delta >= 5.0 or (dc >= 1 and fwi_delta >= 2.0 and drying_trend):
        escalation = "high"
    elif dc >= 1 or fwi_delta >= 2.0 or drying_trend:
        escalation = "moderate"
    else:
        escalation = "low"

    factors = _risk_factors(current, trends)

    if escalation == "critical" or dc >= 4:
        action = "urgent_escalation"
        urgency = "critical"
    elif escalation == "high" or dc >= 2:
        action = "send_warning"
        urgency = "urgent"
    elif escalation == "moderate" or dc >= 1:
        action = "increase_sampling"
        urgency = "time_sensitive"
    else:
        action = "normal_monitoring"
        urgency = "normal"

    verification_need = bool(dc >= 2 or escalation in {"high", "critical"} or (len(win) > 1 and drying_trend))
    return {
        "ref_escalation_risk": escalation,
        "ref_dominant_risk_factors": factors,
        "ref_recommended_action": action,
        "ref_urgency": urgency,
        "ref_verification_need": verification_need,
    }


def output_schema_text() -> str:
    return (
        '{"escalation_risk":"low|moderate|high|critical",'
        '"dominant_risk_factors":["low_humidity|high_temperature|high_wind|no_rain|high_fwi|increasing_fwi|dry_fuel|high_isi"],'
        '"recommended_action":"normal_monitoring|increase_sampling|send_warning|alert_response_team|urgent_escalation",'
        '"urgency":"normal|time_sensitive|urgent|critical",'
        '"verification_need":true|false}'
    )


def example_block(records: pd.DataFrame, n_examples: int) -> str:
    """Build fixed few-shot examples from the training split only when available."""
    if n_examples <= 0:
        return ""
    split_path = project_path("data/processed/train_val_test_split.csv")
    if split_path.exists():
        split = pd.read_csv(split_path)
        train_ids = set(split.loc[split.split == "train", "record_id"].tolist())
        candidates = records[records.record_id.isin(train_ids)].copy()
    else:
        candidates = records.copy()
    if candidates.empty:
        candidates = records.copy()

    chosen = []
    # Prefer one Low, one Moderate, one High if possible.
    for cls in [0, 1, 2, 3, 4, 5]:
        sub = candidates[candidates.danger_class == cls]
        if not sub.empty:
            chosen.append(sub.iloc[len(sub)//2])
        if len(chosen) >= n_examples:
            break
    if len(chosen) < n_examples:
        for _, r in candidates.head(n_examples - len(chosen)).iterrows():
            chosen.append(r)

    lines = ["Examples (same schema; these demonstrate output format and decision-support labels):"]
    for j, r in enumerate(chosen[:n_examples], start=1):
        win = pd.DataFrame([r])
        ref = reference_decision(win)
        out = {
            "escalation_risk": ref["ref_escalation_risk"],
            "dominant_risk_factors": ref["ref_dominant_risk_factors"],
            "recommended_action": ref["ref_recommended_action"],
            "urgency": ref["ref_urgency"],
            "verification_need": ref["ref_verification_need"],
        }
        lines.append(
            f"Example {j} input: computed_FWI={float(r.FWI):.2f}, "
            f"computed_danger_class={int(r.danger_class)} ({CLASS_LABELS[int(r.danger_class)]}); "
            f"{row_context(r)}\nExample {j} output: {json.dumps(out, separators=(',', ':'))}"
        )
    return "\n".join(lines) + "\n"


def build_shared_prompt(win: pd.DataFrame, complexity: str, fewshot: str = "") -> str:
    current = win.iloc[-1]
    dc = int(current["danger_class"])
    current_label = CLASS_LABELS.get(dc, str(dc))
    if complexity == "simple":
        task = "Produce a concise wildfire decision-support report for the current reading."
    elif complexity == "medium":
        task = "Produce a wildfire decision-support report using the recent 3-reading trend."
    elif complexity == "complex":
        task = "Produce a wildfire decision-support report using the recent temporal window, trend, dryness, wind, rain, and escalation indicators."
    else:
        raise ValueError(complexity)

    return (
        "You are a wildfire decision-support assistant.\n"
        "The FWI module has already computed the current FWI and danger class. Do not re-derive the danger class as the main task.\n"
        "Use the computed class only as context, then decide escalation risk, risk factors, recommended action, urgency, and verification need.\n"
        "Allowed risk factors: low_humidity, high_temperature, high_wind, no_rain, high_fwi, increasing_fwi, dry_fuel, high_isi.\n"
        "Allowed actions: normal_monitoring, increase_sampling, send_warning, alert_response_team, urgent_escalation.\n"
        "Return strict JSON only with this schema and no extra text:\n"
        f"{output_schema_text()}\n"
        f"{fewshot}"
        f"Task: {task}\n"
        f"Current computed FWI={float(current['FWI']):.2f}; computed danger class={dc} ({current_label}).\n"
        f"Readings:\n{window_context(win)}\n"
        "JSON only:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-lengths", default="1,3,6", help="Window lengths for simple,medium,complex prompts.")
    ap.add_argument("--few-shot", type=int, default=3, help="Number of fixed training examples to include in the shared prompt. Use 0 to disable.")
    args = ap.parse_args()
    lens = [int(x.strip()) for x in args.window_lengths.split(",")]
    if len(lens) != 3:
        raise ValueError("--window-lengths must contain three integers for simple,medium,complex")
    win_map = {"simple": lens[0], "medium": lens[1], "complex": lens[2]}

    records = pd.read_csv(project_path("data/processed/fires_records.csv")).sort_values("record_id").reset_index(drop=True)
    fewshot = example_block(records, args.few_shot)
    rows = []
    for pos, r in records.iterrows():
        for complexity in ["simple", "medium", "complex"]:
            wlen = int(win_map[complexity])
            start = max(0, pos - wlen + 1)
            win = records.iloc[start:pos + 1].copy()
            text = build_shared_prompt(win, complexity, fewshot)
            trends = trend_features(win)
            ref = reference_decision(win)
            entry = {
                "prompt_id": f"{int(r.record_id)}_{complexity}_w{len(win)}",
                "record_id": int(r.record_id),
                "complexity": complexity,
                "task_type": complexity,
                "window_len": int(len(win)),
                "prompt_text": text,
                "prompt_bytes": len(text.encode()),
                # Compatibility: edge/cloud use the exact same prompt.
                "edge_prompt_text": text,
                "edge_prompt_bytes": len(text.encode()),
                "cloud_prompt_text": text,
                "cloud_prompt_bytes": len(text.encode()),
                "ground_truth_tier": int(r.ground_truth_tier),
                "danger_class": int(r.danger_class),
                "computed_danger_class": int(r.danger_class),
                "danger_label": str(r.danger_label),
                "criticality": float(r.criticality),
                "deadline_s": float(r.deadline_s),
                "iot_fwi": float(r.iot_fwi),
                "iot_transmit": int(r.iot_transmit),
                "iot_gate_decision": str(r.iot_gate_decision),
                "few_shot_examples": int(args.few_shot),
                "ref_escalation_risk": ref["ref_escalation_risk"],
                "ref_dominant_risk_factors": json.dumps(ref["ref_dominant_risk_factors"], separators=(",", ":")),
                "ref_recommended_action": ref["ref_recommended_action"],
                "ref_urgency": ref["ref_urgency"],
                "ref_verification_need": int(ref["ref_verification_need"]),
            }
            for c in FEATURES:
                entry[f"current_{c}"] = float(r[c])
            entry.update(trends)
            rows.append(entry)
    out = project_path("data/processed/prompts.csv")
    ensure_dir(out.parent)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} with {len(rows)} prompts")
    print(f"Shared prompt design: edge_prompt_text == cloud_prompt_text, few_shot={args.few_shot}")
    print("Prompt types/window lengths:")
    print(pd.DataFrame(rows).groupby(["complexity", "window_len"]).size().reset_index(name="n").to_string(index=False))
    print("Prompt byte summary:")
    print(pd.DataFrame(rows).groupby("complexity")[["prompt_bytes", "edge_prompt_bytes", "cloud_prompt_bytes"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
