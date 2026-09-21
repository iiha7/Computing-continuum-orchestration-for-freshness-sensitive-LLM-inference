"""Calibrate reward/state normalization constants from training data.

The constants are empirical quantiles computed after wildfire preprocessing,
trace preprocessing, LLM profiling, freshness calibration, and model fitting.
Only the training split is used.
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

from .common import ensure_dir, load_yaml, project_path, read_jsonl, save_json

PROFILE_FILES = {
    "edge": "edge_outputs.jsonl",
    "cloud": "cloud_outputs.jsonl",
}


def _cloud_latency_scale(cfg: dict) -> float:
    return float(cfg.get("simulation", {}).get("cloud_llm", {}).get("latency_scale", 1.0))


def _load_profiles() -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for mode, filename in PROFILE_FILES.items():
        rows = read_jsonl(project_path("data/llm_profiles", filename))
        out[mode] = {str(r["prompt_id"]): r for r in rows}
    return out


def _q(values: Iterable[float], quantile: float, floor: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(floor)
    return float(max(np.quantile(arr, quantile), floor))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantile", type=float, default=None)
    ap.add_argument("--max-samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_yaml(project_path("configs/experiment.yaml"))
    net = load_yaml(project_path("configs/network_traces.yaml"))["network_model"]
    norm_cfg = cfg.get("normalization", {})
    q = float(args.quantile if args.quantile is not None else norm_cfg.get("quantile", 0.95))

    prompts = pd.read_csv(project_path(cfg["paths"]["prompts"]))
    split = pd.read_csv(project_path(cfg["paths"]["split"]))
    train_ids = set(split.loc[split.split == "train", "record_id"].tolist())
    prompts = prompts[prompts.record_id.isin(train_ids)].reset_index(drop=True)
    if prompts.empty:
        raise ValueError("No training prompts found. Run data_prepare and prompt_builder first.")

    access = pd.read_csv(project_path(net["access_trace_path"]))
    cloud = pd.read_csv(project_path(net["cloud_rtt_trace_path"]))
    profiles = _load_profiles()
    cloud_latency_scale = _cloud_latency_scale(cfg)
    e_cfg = cfg["energy"]
    iot_cfg = cfg.get("simulation", {}).get("iot_filter", {})

    rng = np.random.default_rng(args.seed)
    n = min(int(args.max_samples), max(len(prompts), 1) * max(min(len(access), len(cloud)), 1))
    p_idx = rng.integers(0, len(prompts), size=n)
    a_idx = rng.integers(0, len(access), size=n)
    c_idx = rng.integers(0, len(cloud), size=n)

    prompt_bytes: List[float] = []
    response_bytes: List[float] = []
    bits_values: List[float] = []
    cost_values: List[float] = []
    aoi_values: List[float] = []
    energy_values: List[float] = []
    rtt_values: List[float] = []
    access_delay_values: List[float] = []

    for pi, ai, ci in zip(p_idx, a_idx, c_idx):
        p = prompts.iloc[int(pi)]
        pid = str(p["prompt_id"])
        access_delay_s = float(access.iloc[int(ai)].get("access_delay_s", access.iloc[int(ai)].get("airtime_s", 0.05)))
        row_cloud = cloud.iloc[int(ci)]
        cloud_path_delay_s = float(row_cloud.get("one_way_delay_ms", 0.5 * row_cloud.get("rtt_ms", 100.0))) / 1000.0
        rtt_values.append(cloud_path_delay_s * 1000.0)
        access_delay_values.append(access_delay_s)

        # IoT filter: no edge/cloud transmission for locally low-risk packets.
        if int(p.get("iot_transmit", 1)) == 0:
            aoi_values.append(float(iot_cfg.get("local_latency_s", 0.005)))
            energy_values.append(float(iot_cfg.get("local_energy_j", 0.002)))
            bits_values.append(0.0)
            cost_values.append(0.0)
            prompt_bytes.append(float(p.get("prompt_bytes", 1)))
            response_bytes.append(1.0)
            continue

        ep = profiles["edge"][pid]
        cp = profiles["cloud"][pid]

        # Edge-only
        e_lat = float(ep["latency_s"])
        e_energy = float(e_cfg["p_edge_infer_w"]) * e_lat
        aoi_values.append(access_delay_s + e_lat)
        energy_values.append(e_energy)
        bits_values.append(0.0)
        cost_values.append(0.0)
        prompt_bytes.append(float(ep.get("prompt_bytes", p.get("prompt_bytes", 0))))
        response_bytes.append(float(ep.get("response_bytes", 0)))

        # Cloud-only. Cloud delay is one-way and bytes do not add transmission time.
        pb = float(cp.get("prompt_bytes", 0))
        rb = float(cp.get("response_bytes", 0))
        aoi_values.append(access_delay_s + cloud_path_delay_s + cloud_latency_scale * float(cp["latency_s"]))
        energy_values.append(0.0)
        bits_values.append(8.0 * (pb + rb))
        cost_values.append(float(cp.get("cost_usd", 0.0)))
        prompt_bytes.append(pb)
        response_bytes.append(rb)

    calibrated = {
        "mode": "empirical_train_quantile",
        "quantile": q,
        "n_sampled_contexts": int(n),
        "max_bits_per_request": _q(bits_values, q, floor=1.0),
        "max_energy_j": _q(energy_values, q, floor=1e-6),
        "max_cost": _q(cost_values, q, floor=1e-12),
        "max_aoi_s": _q(aoi_values, q, floor=1e-6),
        "max_prompt_bytes": _q(prompt_bytes, q, floor=1.0),
        "max_response_bytes": _q(response_bytes, q, floor=1.0),
        "max_rtt_ms": _q(rtt_values, q, floor=1e-6),
        "max_access_delay_s": _q(access_delay_values, q, floor=1e-6),
        "cloud_latency_scale": float(cloud_latency_scale),
        "notes": "Empirical 95th-percentile normalizers from training split, LLM profiles, LoED, and cloud-delay trace. Cloud processing latency uses the configured simulation.cloud_llm.latency_scale.",
    }
    outdir = ensure_dir(project_path("data/fitted"))
    out_path = outdir / "normalization.json"
    save_json(calibrated, out_path)
    print(f"Wrote calibrated normalization to {out_path}")
    for k, v in calibrated.items():
        if k != "notes":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
