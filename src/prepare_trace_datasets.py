
"""Preprocess raw LoED and cloud-edge RTT CSV/XLSX files.

Outputs:
  data/network_traces/access_trace.csv
  data/network_traces/cloud_rtt_trace.csv

The AoP communication model uses LoED CRC outcomes to estimate an empirical
access reliability, then computes access delay as LoRa time-on-air plus
expected retransmission overhead.  Cloud latency is stored as both raw RTT and
one-way delay; cloud AoP uses the one-way upload delay only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .common import ensure_dir, project_path, set_seed
from .network_traces import normalize_loed_access_trace, normalize_cloud_edge_rtt_trace


def read_table(path: str | Path) -> pd.DataFrame:
    p = project_path(str(path)) if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)


def demo_access(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    snr = rng.normal(0, 6, n)
    sf = rng.choice([7, 8, 9, 10, 11, 12], n, p=[0.35, 0.25, 0.18, 0.12, 0.07, 0.03])
    # success probability rises with SNR and falls with higher SF in this demo.
    p = 1 / (1 + np.exp(-(snr + 7 - 0.8 * (sf - 7)) / 3.0))
    crc = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="s").astype(str),
        "device_address": rng.choice(["devA", "devB", "devC"], n),
        "physical_payload": "",
        "gateway": rng.choice(["gw_main", "gw_aux"], n, p=[0.75, 0.25]),
        "crc_status": crc,
        "frequency": rng.choice([867100000, 867300000, 867900000, 868300000], n),
        "spreading_factor": sf,
        "bandwidth": 125,
        "code_rate": "4/5",
        "rssi": rng.normal(-106, 9, n),
        "snr": snr,
        "size": rng.integers(20, 80, n),
        "mtype": 10,
        "fcnt": np.arange(n),
        "fport": 5,
    })


def demo_cloud(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    milan = rng.lognormal(np.log(55), 0.45, n)
    ams = rng.lognormal(np.log(45), 0.35, n)
    frankfurt = rng.lognormal(np.log(50), 0.35, n)
    min_vals = np.minimum.reduce([milan, ams, frankfurt])
    min_labels = np.where(min_vals == milan, "Milan-MIL01.csv", np.where(min_vals == ams, "Amsterdam-AMS03.csv", "Frankfurt-FRA01.csv"))
    return pd.DataFrame({
        "prb": np.arange(50000, 50000 + n),
        "minMedian": min_vals,
        "minLabel": min_labels,
        "value": milan,
        "label": "Milan-MIL01.csv",
        "value.1": ams,
        "label.1": "Amsterdam-AMS03.csv",
        "value.2": frankfurt,
        "label.2": "Frankfurt-FRA01.csv",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--access-raw", default=None, help="Raw LoED access CSV/XLSX.")
    ap.add_argument("--cloud-rtt-raw", default=None, help="Raw cloud-edge RTT CSV/XLSX.")
    ap.add_argument("--cloud-label", default="Milan-MIL01.csv", help="Fixed cloud region label to extract.")
    ap.add_argument("--cloud-mode", choices=["fixed", "min"], default="fixed")
    ap.add_argument("--gateway-strategy", choices=["largest", "all"], default="largest")
    ap.add_argument("--gateway-id", default=None, help="Optional explicit LoED gateway id.")
    ap.add_argument("--payload-bytes-override", type=int, default=None)
    ap.add_argument("--snr-bin-width", type=float, default=2.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--retransmission-backoff-s", type=float, default=1.0)
    ap.add_argument("--demo", action="store_true", help="Generate demo traces for software validation only.")
    ap.add_argument("--n-demo", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    set_seed(args.seed)

    outdir = ensure_dir(project_path("data/network_traces"))
    if args.demo:
        raw_access = demo_access(args.n_demo, args.seed)
        raw_cloud = demo_cloud(args.n_demo, args.seed)
    else:
        if args.access_raw is None or args.cloud_rtt_raw is None:
            raise ValueError("Provide --access-raw and --cloud-rtt-raw, or use --demo for smoke tests.")
        raw_access = read_table(args.access_raw)
        raw_cloud = read_table(args.cloud_rtt_raw)

    access = normalize_loed_access_trace(
        raw_access,
        gateway_strategy=args.gateway_strategy,
        gateway_id=args.gateway_id,
        payload_bytes_override=args.payload_bytes_override,
        snr_bin_width=args.snr_bin_width,
        max_attempts=args.max_attempts,
        retransmission_backoff_s=args.retransmission_backoff_s,
    )
    cloud = normalize_cloud_edge_rtt_trace(raw_cloud, cloud_label=args.cloud_label, mode=args.cloud_mode)

    access.to_csv(outdir / "access_trace.csv", index=False)
    cloud.to_csv(outdir / "cloud_rtt_trace.csv", index=False)

    print(f"Wrote {outdir/'access_trace.csv'} ({len(access)} rows)")
    print(access[["sample_id", "gateway", "crc_status", "spreading_factor", "bandwidth", "snr", "p_success", "expected_attempts", "access_delay_s"]].head().to_string(index=False))
    print(f"\nWrote {outdir/'cloud_rtt_trace.csv'} ({len(cloud)} rows)")
    print(cloud.head().to_string(index=False))
    if args.demo:
        print("\nWARNING: demo traces are for software validation only, not paper results.")


if __name__ == "__main__":
    main()
