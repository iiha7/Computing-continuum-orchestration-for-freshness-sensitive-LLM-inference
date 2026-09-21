"""Profile real edge/cloud LLMs and score decision-support responses.

Both edge and cloud receive the same prompt.  The prompt provides the computed
FWI/danger class as context, and the LLMs are scored on structured decision
support: escalation risk, dominant risk factors, recommended action, urgency,
and verification need.  LLM self-confidence is not used; edge reliability is
calibrated later by ``src.fit_models`` from observable profile features.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from .common import ensure_dir, project_path, write_jsonl

ESCALATION_CHOICES = {"low", "moderate", "high", "critical"}
ACTION_CHOICES = {"normal_monitoring", "increase_sampling", "send_warning", "alert_response_team", "urgent_escalation"}
URGENCY_CHOICES = {"normal", "time_sensitive", "urgent", "critical"}
RISK_FACTOR_CHOICES = {"low_humidity", "high_temperature", "high_wind", "no_rain", "high_fwi", "increasing_fwi", "dry_fuel", "high_isi"}


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    candidates = [s]
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        candidates.insert(0, m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _norm_key(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    aliases = {
        "watch": "increase_sampling",
        "monitor": "normal_monitoring",
        "normal": "normal_monitoring",
        "warning": "send_warning",
        "send_alert": "send_warning",
        "alert_team": "alert_response_team",
        "dispatch": "alert_response_team",
        "urgent": "urgent_escalation",
        "very_high": "high",
        "medium": "moderate",
        "time_sensitive": "time_sensitive",
        "timesensitive": "time_sensitive",
        "true": "true",
        "false": "false",
    }
    return aliases.get(s, s)


def _parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,;|]", str(value))
    out = []
    aliases = {
        "low_rh": "low_humidity",
        "humidity_low": "low_humidity",
        "hot": "high_temperature",
        "wind": "high_wind",
        "high_winds": "high_wind",
        "no_precipitation": "no_rain",
        "dry": "dry_fuel",
        "fuel_dryness": "dry_fuel",
        "fwi_high": "high_fwi",
        "fwi_increasing": "increasing_fwi",
        "rising_fwi": "increasing_fwi",
        "high_spread_index": "high_isi",
    }
    for item in raw:
        k = _norm_key(item)
        k = aliases.get(k, k)
        if k in RISK_FACTOR_CHOICES and k not in out:
            out.append(k)
    return out


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = _norm_key(value)
    if s in {"true", "yes", "1", "needed", "required"}:
        return True
    if s in {"false", "no", "0", "not_needed", "none"}:
        return False
    return None


def parse_decision_response(text: str) -> Dict[str, Any]:
    obj = _extract_json_object(text or "") or {}
    esc = _norm_key(obj.get("escalation_risk", ""))
    if esc not in ESCALATION_CHOICES:
        esc = None
    action = _norm_key(obj.get("recommended_action", obj.get("action", "")))
    if action not in ACTION_CHOICES:
        action = None
    urgency = _norm_key(obj.get("urgency", ""))
    if urgency not in URGENCY_CHOICES:
        urgency = None
    factors = _parse_list(obj.get("dominant_risk_factors", obj.get("risk_factors", [])))
    verify = _parse_bool(obj.get("verification_need", obj.get("requires_verification", None)))
    valid = int(obj != {} and esc is not None and action is not None and urgency is not None and verify is not None)
    return {
        "response_valid": valid,
        "pred_escalation_risk": esc,
        "pred_dominant_risk_factors": factors,
        "pred_recommended_action": action,
        "pred_urgency": urgency,
        "pred_verification_need": verify,
    }


def _risk_f1(pred: List[str], ref: List[str]) -> float:
    ps, rs = set(pred), set(ref)
    if not ps and not rs:
        return 1.0
    if not ps or not rs:
        return 0.0
    tp = len(ps & rs)
    precision = tp / max(len(ps), 1)
    recall = tp / max(len(rs), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _load_ref_factors(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        obj = json.loads(str(value))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return _parse_list(value)


def score_decision(parsed: Dict[str, Any], prompt_row: Dict[str, Any]) -> Dict[str, Any]:
    ref_factors = _load_ref_factors(prompt_row.get("ref_dominant_risk_factors", "[]"))
    ref_verify = bool(int(prompt_row.get("ref_verification_need", 0)))
    s_esc = int(parsed["pred_escalation_risk"] == prompt_row.get("ref_escalation_risk"))
    s_action = int(parsed["pred_recommended_action"] == prompt_row.get("ref_recommended_action"))
    s_urg = int(parsed["pred_urgency"] == prompt_row.get("ref_urgency"))
    s_verify = int(parsed["pred_verification_need"] == ref_verify)
    s_factors = _risk_f1(parsed["pred_dominant_risk_factors"], ref_factors)
    valid = int(parsed["response_valid"])
    quality = valid * (0.25 * s_esc + 0.25 * s_factors + 0.25 * s_action + 0.15 * s_urg + 0.10 * s_verify)
    return {
        "ref_escalation_risk": prompt_row.get("ref_escalation_risk"),
        "ref_dominant_risk_factors": ref_factors,
        "ref_recommended_action": prompt_row.get("ref_recommended_action"),
        "ref_urgency": prompt_row.get("ref_urgency"),
        "ref_verification_need": ref_verify,
        "escalation_correct": s_esc,
        "risk_factor_f1": float(s_factors),
        "action_correct": s_action,
        "urgency_correct": s_urg,
        "verification_correct": s_verify,
        "response_quality": float(quality),
    }


def estimate_tokens_from_bytes(n_bytes: int) -> int:
    return max(1, int(round(n_bytes / 4)))


def binary_entropy(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, float(p)))
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def call_ollama(prompt: str, model: str, base_url: str, max_tokens: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    latency_s = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    text = data.get("response", "")
    response_bytes = len(text.encode("utf-8"))
    input_tokens = int(data.get("prompt_eval_count") or estimate_tokens_from_bytes(len(prompt.encode("utf-8"))))
    output_tokens = int(data.get("eval_count") or estimate_tokens_from_bytes(response_bytes))
    return {"text": text, "latency_s": latency_s, "ttft_s": latency_s,
            "input_tokens": input_tokens, "output_tokens": output_tokens}


def call_openai_responses(prompt: str, model: str, max_tokens: int) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK first: pip install openai") from exc
    client = OpenAI()
    t0 = time.perf_counter()
    response = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens)
    latency_s = time.perf_counter() - t0
    text = getattr(response, "output_text", "") or ""
    response_bytes = len(text.encode("utf-8"))
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    if input_tokens is None:
        input_tokens = estimate_tokens_from_bytes(len(prompt.encode("utf-8")))
    if output_tokens is None:
        output_tokens = estimate_tokens_from_bytes(response_bytes)
    return {"text": text, "latency_s": latency_s, "ttft_s": latency_s,
            "input_tokens": int(input_tokens), "output_tokens": int(output_tokens)}


def call_provider(provider: str, prompt: str, model: str, max_tokens: int) -> Dict[str, Any]:
    provider = provider.lower().strip()
    if provider == "ollama":
        return call_ollama(prompt, model, os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), max_tokens)
    if provider == "openai":
        return call_openai_responses(prompt, model, max_tokens)
    raise ValueError(f"Unsupported provider: {provider}. Use 'ollama' or 'openai'.")


def _prompt_for_mode(prompt_row: Dict[str, Any], mode: str) -> Tuple[str, int, str]:
    # Fair evaluation: edge and cloud receive the same prompt_text.
    text = str(prompt_row.get("prompt_text") or "")
    if not text:
        text = str(prompt_row.get("edge_prompt_text") or prompt_row.get("cloud_prompt_text") or "")
    nbytes = int(prompt_row.get("prompt_bytes") or len(text.encode("utf-8")))
    return text, nbytes, "shared_prompt_text"


def profile_one(prompt_row: Dict[str, Any], mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    if mode == "edge":
        provider, model = args.edge_provider, args.edge_model
        max_tokens, cost_usd = args.edge_max_tokens, 0.0
    elif mode == "cloud":
        provider, model = args.cloud_provider, args.cloud_model
        max_tokens, cost_usd = args.cloud_max_tokens, None
    else:
        raise ValueError(mode)

    prompt_text, prompt_bytes, prompt_source = _prompt_for_mode(prompt_row, mode)
    call = call_provider(provider=provider, prompt=prompt_text, model=model, max_tokens=max_tokens)
    text = call["text"]
    parsed = parse_decision_response(text)
    score = score_decision(parsed, prompt_row)
    response_bytes = len(text.encode("utf-8"))
    input_tokens = int(call["input_tokens"])
    output_tokens = int(call["output_tokens"])
    if cost_usd is None:
        cost_usd = input_tokens * args.price_input_per_token + output_tokens * args.price_output_per_token

    quality = float(score["response_quality"])
    correct = int(quality >= args.acceptable_quality_threshold)
    default_reliability = 0.0 if mode == "edge" else 1.0
    computed_class = int(prompt_row.get("computed_danger_class", prompt_row.get("danger_class", prompt_row.get("ground_truth_tier", 0))))

    return {
        "prompt_id": prompt_row["prompt_id"],
        "record_id": int(prompt_row["record_id"]),
        "complexity": prompt_row["complexity"],
        "mode_role": mode,
        "model_name": model,
        "provider": provider,
        "prompt_bytes": prompt_bytes,
        "prompt_source": prompt_source,
        "response_text": text,
        "response_valid": int(parsed["response_valid"]),
        "self_reported_confidence": None,
        "computed_danger_class": computed_class,
        "predicted_tier": computed_class,  # compatibility: class is computed context, not LLM target
        "ground_truth_tier": computed_class,
        "predicted_danger_class": computed_class,
        "ground_truth_danger_class": computed_class,
        "pred_escalation_risk": parsed["pred_escalation_risk"],
        "pred_dominant_risk_factors": parsed["pred_dominant_risk_factors"],
        "pred_recommended_action": parsed["pred_recommended_action"],
        "pred_urgency": parsed["pred_urgency"],
        "pred_verification_need": parsed["pred_verification_need"],
        **score,
        "correct": correct,
        "quality_threshold": float(args.acceptable_quality_threshold),
        "response_bytes": int(response_bytes),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": float(call["latency_s"]),
        "ttft_s": float(call["ttft_s"]),
        "cost_usd": float(cost_usd),
        "confidence": float(default_reliability),
        "calibrated_edge_confidence": float(default_reliability) if mode == "edge" else None,
        "edge_margin": 0.0 if mode == "edge" else None,
        "edge_entropy": binary_entropy(0.5) if mode == "edge" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile real edge/cloud LLMs for the IoT-edge-cloud DRL simulator.")
    parser.add_argument("--prompts", default="data/processed/prompts.csv")
    parser.add_argument("--out-dir", default="data/llm_profiles")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--cloud-only", action="store_true")
    parser.add_argument("--edge-only", action="store_true")
    parser.add_argument("--edge-provider", default=os.environ.get("EDGE_PROVIDER", "ollama"), choices=["ollama", "openai"])
    parser.add_argument("--cloud-provider", default=os.environ.get("CLOUD_PROVIDER", "ollama"), choices=["ollama", "openai"])
    parser.add_argument("--edge-model", default=os.environ.get("EDGE_MODEL", "qwen2.5:0.5b"))
    parser.add_argument("--cloud-model", default=os.environ.get("CLOUD_MODEL", "qwen2.5:7b"))
    parser.add_argument("--edge-max-tokens", type=int, default=128)
    parser.add_argument("--cloud-max-tokens", type=int, default=128)
    parser.add_argument("--acceptable-quality-threshold", type=float, default=0.75)
    parser.add_argument("--price-input-per-token", type=float, default=0.00000015)
    parser.add_argument("--price-output-per-token", type=float, default=0.00000060)
    args = parser.parse_args()

    prompts = pd.read_csv(project_path(args.prompts))
    if args.limit is not None:
        prompts = prompts.head(args.limit).copy()

    outdir = ensure_dir(project_path(args.out_dir))
    if args.cloud_only and args.edge_only:
        raise ValueError("Use only one of --cloud-only or --edge-only")
    if args.cloud_only:
        modes_and_files: Tuple[Tuple[str, str], ...] = (("cloud", "cloud_outputs.jsonl"),)
    elif args.edge_only:
        modes_and_files = (("edge", "edge_outputs.jsonl"),)
    else:
        modes_and_files = (("edge", "edge_outputs.jsonl"), ("cloud", "cloud_outputs.jsonl"))

    for mode, filename in modes_and_files:
        rows = []
        print(f"\nProfiling mode={mode} ...")
        for _, row in tqdm(prompts.iterrows(), total=len(prompts)):
            rows.append(profile_one(row.to_dict(), mode, args))
            if args.sleep > 0:
                time.sleep(args.sleep)
        output_path = outdir / filename
        write_jsonl(rows, output_path)
        print(f"Wrote {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
