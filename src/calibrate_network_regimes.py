"""Calibrate empirical latency-burden regimes from the cloud-edge one-way delay trace.

Because the selected cloud-edge latency dataset provides latency but not throughput,
network regimes are defined by a one-way latency burden:

    eta_lat = T_cloud_up / D

where T_cloud_up is the one-way edge-to-cloud delay used by the cloud
Age-of-Processing path and D is the active freshness deadline.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from .common import ensure_dir, load_yaml, project_path, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=200000,
                    help="Max request-trace pairs for quantile calibration.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = load_yaml(project_path("configs/experiment.yaml"))
    net = load_yaml(project_path("configs/network_traces.yaml"))["network_model"]
    prompts = pd.read_csv(project_path(cfg["paths"]["prompts"]))
    split = pd.read_csv(project_path(cfg["paths"]["split"]))
    train_ids = set(split.loc[split.split == "train", "record_id"].tolist())
    prompts = prompts[prompts.record_id.isin(train_ids)].reset_index(drop=True)
    cloud = pd.read_csv(project_path(net["cloud_rtt_trace_path"]))

    n_pairs = min(args.max_pairs, len(prompts) * len(cloud))
    p_idx = rng.integers(0, len(prompts), size=n_pairs)
    c_idx = rng.integers(0, len(cloud), size=n_pairs)
    pp = prompts.iloc[p_idx].reset_index(drop=True)
    cc = cloud.iloc[c_idx].reset_index(drop=True)

    delay_ms = cc["one_way_delay_ms"].astype(float).to_numpy() if "one_way_delay_ms" in cc.columns else 0.5 * cc["rtt_ms"].astype(float).to_numpy()
    eta = (delay_ms / 1000.0) / pp["deadline_s"].astype(float).to_numpy()
    eta = eta[np.isfinite(eta)]
    qs = [float(np.quantile(eta, q)) for q in net["regime_definition"]["quantiles"]]

    out = {
        "definition": "eta_lat=(cloud_one_way_delay_seconds)/D",
        "eta_latency_quantile_thresholds": qs,
        "quantiles": net["regime_definition"]["quantiles"],
        "n_pairs": int(len(eta)),
        "eta_latency_mean": float(np.mean(eta)),
        "eta_latency_p95": float(np.quantile(eta, 0.95)),
    }
    outdir = ensure_dir(project_path("data/fitted"))
    save_json(out, outdir / "network_regimes.json")
    print("Network latency-regime calibration complete.")
    print("eta_lat thresholds:", qs)


if __name__ == "__main__":
    main()
