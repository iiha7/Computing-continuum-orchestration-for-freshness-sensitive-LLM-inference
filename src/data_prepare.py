"""Prepare wildfire sensing records for the IoT--Edge--Cloud simulator.

Creates:
- data/processed/fires_records.csv
- data/processed/train_val_test_split.csv

The decision target is a six-class Fire Weather Index (FWI) danger class:
    0 Low          : FWI < 11.2
    1 Moderate     : 11.2 <= FWI < 21.3
    2 High         : 21.3 <= FWI < 38.0
    3 Very High    : 38.0 <= FWI < 50.0
    4 Extreme      : 50.0 <= FWI < 70.0
    5 Very Extreme : FWI >= 70.0

If the raw CSV does not contain an FWI column but contains ISI and BUI, FWI is
computed from the standard final Canadian FWI combination of ISI and BUI. This
matches the intended IoT-layer behavior: the sensing node/gateway can compute a
lightweight fire-weather risk index before deciding whether to transmit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ensure_dir, project_path, set_seed

BASE_REQ = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]
REQ = BASE_REQ + ["FWI"]
DANGER_LABELS = ["Low", "Moderate", "High", "Very High", "Extreme", "Very Extreme"]
IOT_FWI_THRESHOLD = 11.2


def compute_fwi_from_isi_bui(isi: float, bui: float) -> float:
    """Compute the final FWI value from ISI and BUI.

    This is the last step of the Canadian FWI system. It assumes that ISI and
    BUI have already been computed from meteorological inputs. It is used only
    when the raw dataset does not provide FWI directly.
    """
    isi = max(0.0, float(isi))
    bui = max(0.0, float(bui))
    if bui <= 80.0:
        fd = 0.626 * (bui ** 0.809) + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))
    b = 0.1 * isi * fd
    if b <= 1.0:
        return float(b)
    return float(np.exp(2.72 * ((0.434 * np.log(b)) ** 0.647)))


def fwi_to_danger_class(fwi: float) -> int:
    """Map Fire Weather Index to six fire-danger classes."""
    fwi = float(fwi)
    if fwi < 11.2:
        return 0
    if fwi < 21.3:
        return 1
    if fwi < 38.0:
        return 2
    if fwi < 50.0:
        return 3
    if fwi < 70.0:
        return 4
    return 5


def danger_label(class_id: int) -> str:
    return DANGER_LABELS[int(class_id)]


def criticality_from_class(class_id: int) -> float:
    return [0.20, 0.35, 0.55, 0.75, 0.90, 1.00][int(class_id)]


def deadline_from_class(class_id: int) -> float:
    """Placeholder deadline overwritten by src.calibrate_freshness."""
    return 10.0


def generate_synthetic_fires(n: int, seed: int) -> pd.DataFrame:
    """Generate synthetic wildfire records for smoke tests only."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(30, 8, n).clip(10, 52)
    rh = rng.normal(52, 22, n).clip(5, 98)
    wind = rng.normal(17, 7, n).clip(1, 50)
    rain = rng.exponential(0.7, n).clip(0, 10)
    dryness = (temp - 16) * 1.1 + (75 - rh) * 0.45 + wind * 0.35 - rain * 5.0
    ffmc = (62 + dryness + rng.normal(0, 6, n)).clip(15, 99)
    dmc = (4 + np.maximum(dryness, 0) * 1.6 + rng.normal(0, 10, n)).clip(0, 180)
    dc = (25 + np.maximum(dryness, 0) * 5.5 + rng.normal(0, 35, n)).clip(0, 700)
    isi = (0.055 * ffmc + 0.32 * wind - 0.6 * rain + rng.normal(0, 2.2, n)).clip(0, 35)
    bui = (0.65 * dmc + 0.06 * dc + rng.normal(0, 8, n)).clip(0, 220)
    fwi = np.array([compute_fwi_from_isi_bui(i, b) for i, b in zip(isi, bui)]).clip(0, 85)
    return pd.DataFrame({"Temperature": temp.round(1), "RH": rh.round(1), "Ws": wind.round(1),
                         "Rain": rain.round(2), "FFMC": ffmc.round(1), "DMC": dmc.round(1),
                         "DC": dc.round(1), "ISI": isi.round(1), "BUI": bui.round(1), "FWI": fwi.round(1)})


def load_real(raw_path: Path, allow_download: bool) -> pd.DataFrame | None:
    if raw_path.exists():
        return pd.read_csv(raw_path)
    if allow_download:
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=547)
            return pd.concat([ds.data.features, ds.data.targets], axis=1)
        except Exception as exc:
            print(f"Download failed, falling back if allowed: {exc}")
    return None


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, compute FWI when missing, and keep numeric fields."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        for r in REQ:
            if c.strip().lower() == r.lower():
                rename[c] = r
    df = df.rename(columns=rename)
    missing_base = [c for c in BASE_REQ if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing columns {missing_base}; found {df.columns.tolist()}")

    out = df[[c for c in BASE_REQ if c in df.columns]].copy()
    for c in BASE_REQ:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "FWI" in df.columns:
        out["FWI"] = pd.to_numeric(df["FWI"], errors="coerce")
        fwi_source = "provided"
    else:
        out["FWI"] = [compute_fwi_from_isi_bui(i, b) for i, b in zip(out["ISI"], out["BUI"])]
        fwi_source = "computed_from_ISI_BUI"
    out["fwi_source"] = fwi_source
    return out.dropna(subset=REQ).reset_index(drop=True)


def safe_two_stage_split(df: pd.DataFrame, label_col: str, seed: int) -> Tuple[pd.Index, pd.Index, pd.Index]:
    counts = df[label_col].value_counts()
    stratify_all: Optional[pd.Series] = df[label_col] if counts.min() >= 2 and counts.size > 1 else None
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=seed, stratify=stratify_all)
    train_counts = df.loc[train_idx, label_col].value_counts()
    stratify_train: Optional[pd.Series] = df.loc[train_idx, label_col] if train_counts.min() >= 2 and train_counts.size > 1 else None
    train_idx, val_idx = train_test_split(train_idx, test_size=0.20, random_state=seed, stratify=stratify_train)
    return train_idx, val_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", default="data/raw/algerian_forest_fires.csv")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--fallback-synthetic", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--iot-fwi-threshold", type=float, default=IOT_FWI_THRESHOLD,
                    help="FWI threshold below which the IoT layer suppresses edge transmission.")
    args = ap.parse_args(); set_seed(args.seed)

    df = load_real(project_path(args.raw_csv), args.allow_download); source = "real"
    if df is None:
        if not args.fallback_synthetic:
            raise FileNotFoundError("Provide raw CSV, use --allow-download, or --fallback-synthetic")
        df = generate_synthetic_fires(args.n_synthetic, args.seed); source = "synthetic_fallback"

    df = canonicalize(df)
    df.insert(0, "record_id", np.arange(len(df)))
    df["ground_truth_tier"] = df["FWI"].apply(fwi_to_danger_class).astype(int)
    df["danger_class"] = df["ground_truth_tier"]
    df["danger_label"] = df["danger_class"].apply(danger_label)
    df["criticality"] = df["danger_class"].apply(criticality_from_class)
    df["deadline_s"] = df["danger_class"].apply(deadline_from_class)
    df["iot_fwi"] = df["FWI"].astype(float)
    df["iot_transmit"] = (df["iot_fwi"] >= float(args.iot_fwi_threshold)).astype(int)
    df["iot_gate_decision"] = np.where(df["iot_transmit"] == 1, "transmit_to_edge", "local_low_no_transmit")
    df["source"] = source

    outdir = ensure_dir(project_path("data/processed"))
    train_idx, val_idx, test_idx = safe_two_stage_split(df, "ground_truth_tier", args.seed)
    split = pd.DataFrame({"record_id": df.record_id, "split": "train"})
    split.loc[split.record_id.isin(df.loc[val_idx, "record_id"]), "split"] = "val"
    split.loc[split.record_id.isin(df.loc[test_idx, "record_id"]), "split"] = "test"

    df.to_csv(outdir / "fires_records.csv", index=False)
    split.to_csv(outdir / "train_val_test_split.csv", index=False)
    print(f"Wrote {outdir/'fires_records.csv'} ({len(df)} records, source={source})")
    print("FWI danger-class distribution:")
    print(df.groupby(["danger_class", "danger_label"]).size().reset_index(name="n").to_string(index=False))
    print("IoT gate distribution:")
    print(df["iot_gate_decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
