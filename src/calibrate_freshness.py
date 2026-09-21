"""Calibrate FWI-class freshness deadlines from profiled edge-only AoI.

Run after LLM profiling and before model fitting/training.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .common import ensure_dir, load_yaml, project_path, read_jsonl, save_json
from .freshness import calibrate_alpha_min, deadline_from_class, tau_from_deadline


def task_deadline_multiplier(cfg: dict, complexity: str) -> float:
    multipliers = cfg.get("freshness", {}).get("task_deadline_multipliers", {})
    return float(multipliers.get(str(complexity), 1.0))


def expected_access_delay_s(cfg: dict, access_trace_path=None) -> float:
    """Expected IoT access delay from normalized LoED trace.

    The final access model computes LoRa time-on-air during preprocessing and
    stores it as airtime_s. If an older trace contains access_delay_ms, that is
    used as a fallback. If no trace exists, a conservative debug fallback is used.
    """
    if access_trace_path is not None and access_trace_path.exists():
        try:
            df = pd.read_csv(access_trace_path)
            if "airtime_s" in df.columns and df["airtime_s"].notna().any():
                return float(df["airtime_s"].dropna().mean())
            if "access_delay_ms" in df.columns and df["access_delay_ms"].notna().any():
                return float(df["access_delay_ms"].dropna().mean() / 1000.0)
        except Exception:
            pass
    # Debug fallback only; run src.prepare_trace_datasets before paper runs.
    return 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling-period-s", type=float, default=None)
    ap.add_argument("--quantile", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    args = ap.parse_args()

    cfg = load_yaml(project_path("configs/experiment.yaml"))
    net = load_yaml(project_path("configs/network_traces.yaml"))
    fcfg = cfg["freshness"]

    sampling_period_s = float(args.sampling_period_s or fcfg["sampling_period_s"])
    quantile = float(args.quantile or fcfg["calibration_quantile"])
    margin = float(args.margin or fcfg["calibration_margin"])
    alpha_max = float(fcfg["alpha_max"])
    min_clip, max_clip = [float(v) for v in fcfg["alpha_min_clip"]]
    q = float(fcfg["utility_at_deadline"])

    edge_rows = read_jsonl(project_path("data/llm_profiles/edge_outputs.jsonl"))
    if not edge_rows:
        raise FileNotFoundError("Run profile_llms_simulated.py or profile_llms_real.py before freshness calibration.")

    access_path = project_path(net["network_model"]["access_trace_path"])
    t_access = expected_access_delay_s(cfg, access_path)
    edge_aoi = np.array([t_access + float(r["latency_s"]) for r in edge_rows], dtype=float)

    alpha_min = calibrate_alpha_min(edge_aoi, sampling_period_s, quantile, margin, min_clip, max_clip)

    records_path = project_path("data/processed/fires_records.csv")
    prompts_path = project_path("data/processed/prompts.csv")
    records = pd.read_csv(records_path); prompts = pd.read_csv(prompts_path)

    def dline(c):
        return deadline_from_class(int(c), sampling_period_s, alpha_max, alpha_min)

    records["deadline_s"] = records["danger_class"].apply(dline)
    prompts["deadline_s_base"] = prompts["danger_class"].apply(dline)
    prompts["deadline_s"] = prompts.apply(
        lambda r: float(r["deadline_s_base"]) * task_deadline_multiplier(cfg, r.get("complexity", "simple")),
        axis=1,
    )
    records.to_csv(records_path, index=False); prompts.to_csv(prompts_path, index=False)

    deadlines = {str(c): float(dline(c)) for c in range(6)}
    task_deadline_multipliers = dict(fcfg.get("task_deadline_multipliers", {}))
    taus = {str(c): float(tau_from_deadline(deadlines[str(c)], q)) for c in range(6)}
    out = {
        "sampling_period_s": sampling_period_s,
        "alpha_max": alpha_max,
        "alpha_min_default": float(alpha_min),
        "calibration_quantile": quantile,
        "calibration_margin": margin,
        "edge_aoi_mean_s": float(edge_aoi.mean()),
        "edge_aoi_quantile_s": float(np.quantile(edge_aoi, quantile)),
        "expected_access_delay_s": float(t_access),
        "utility_at_deadline": q,
        "deadline_by_class_s": deadlines,
        "task_deadline_multipliers": task_deadline_multipliers,
        "tau_by_class_s": taus,
        "sensitivity_multipliers": fcfg["sensitivity_multipliers"],
    }
    outdir = ensure_dir(project_path("data/fitted"))
    save_json(out, outdir / "freshness_calibration.json")
    print("Freshness calibration complete.")
    print(f"alpha_min={alpha_min:.4f}, expected_access_delay={t_access:.4f}s")
    print("Deadlines by class:", deadlines)
    print("Task deadline multipliers:", task_deadline_multipliers)


if __name__ == "__main__":
    main()
