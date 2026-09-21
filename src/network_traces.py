
"""Trace utilities for the AoP-driven IoT--edge--cloud communication model.

The simulator uses two independent public traces:

1) IoT -> Edge access trace from LoED (LoRaWAN at the Edge Dataset).  Unlike
   earlier versions that retained only successful CRC rows, the AoP version
   keeps both successful and failed CRC observations when the radio metadata is
   valid.  The trace is used to estimate an empirical packet-success
   probability conditioned on LoRaWAN configuration and SNR bin.  IoT access
   delay is modeled as LoRa time-on-air plus expected retransmission overhead.

2) Edge -> Cloud latency trace from the cloud-edge latency dataset.  The raw
   dataset reports RTT.  The normalized trace stores both raw RTT and one-way
   delay.  Cloud AoP uses a one-way upload leg plus cloud processing; no
   downlink delay is modeled in this version.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import base64
import math
import numpy as np
import pandas as pd

from .common import load_json, project_path


@dataclass
class AccessSample:
    """One IoT-to-edge LoRaWAN access observation."""
    sample_id: int
    timestamp: str
    gateway: str
    device_address: str
    rssi_dbm: float
    snr_db: float
    spreading_factor: int
    bandwidth_khz: float
    frequency_hz: float
    payload_size_bytes: Optional[float]
    airtime_s: float
    p_success: float
    expected_attempts: float
    access_delay_s: float
    crc_status: int
    snr_bin: str


@dataclass
class CloudRTTSample:
    """One edge-to-cloud latency observation."""
    sample_id: int
    probe_id: int
    cloud_region: str
    rtt_ms: float
    one_way_delay_ms: float
    min_cloud_region: str
    min_rtt_ms: float
    regime_id: int
    regime_name: str


class EmpiricalTraceSampler:
    """Sample rows sequentially or iid from a CSV trace.

    For unrelated traces (fire records, LoED, cloud latency), we do not impose
    one-to-one row alignment.  Each trace is replayed with an independent random
    starting offset.
    """

    def __init__(self, csv_path: Path, sampling_mode: str = "sequential", seed: int = 1):
        self.path = Path(csv_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing trace file: {self.path}")
        self.df = pd.read_csv(self.path)
        if len(self.df) == 0:
            raise ValueError(f"Trace file is empty: {self.path}")
        self.sampling_mode = sampling_mode
        self.rng = np.random.default_rng(seed)
        self.idx = int(self.rng.integers(0, len(self.df))) if sampling_mode == "sequential" else 0

    def next_row(self) -> pd.Series:
        if self.sampling_mode == "empirical_iid":
            return self.df.iloc[int(self.rng.integers(0, len(self.df)))]
        row = self.df.iloc[self.idx % len(self.df)]
        self.idx += 1
        return row


def _to_float(row, name: str, default: float) -> float:
    return float(row[name]) if name in row and pd.notna(row[name]) else float(default)


def _to_str(row, name: str, default: str = "") -> str:
    return str(row[name]) if name in row and pd.notna(row[name]) else default


class AccessTrace(EmpiricalTraceSampler):
    """Sampler for normalized LoED access traces."""

    def sample(self) -> AccessSample:
        r = self.next_row()
        return AccessSample(
            sample_id=int(_to_float(r, "sample_id", self.idx)),
            timestamp=_to_str(r, "time", _to_str(r, "timestamp", "")),
            gateway=_to_str(r, "gateway", ""),
            device_address=_to_str(r, "device_address", ""),
            rssi_dbm=_to_float(r, "rssi", _to_float(r, "rssi_dbm", -120.0)),
            snr_db=_to_float(r, "snr", _to_float(r, "snr_db", 0.0)),
            spreading_factor=int(_to_float(r, "spreading_factor", 7)),
            bandwidth_khz=_to_float(r, "bandwidth", _to_float(r, "bandwidth_khz", 125.0)),
            frequency_hz=_to_float(r, "frequency", 0.0),
            payload_size_bytes=_to_float(r, "size", np.nan) if "size" in r else None,
            airtime_s=_to_float(r, "airtime_s", 0.05),
            p_success=float(np.clip(_to_float(r, "p_success", 1.0), 1e-3, 1.0)),
            expected_attempts=_to_float(r, "expected_attempts", 1.0),
            access_delay_s=_to_float(r, "access_delay_s", _to_float(r, "airtime_s", 0.05)),
            crc_status=int(_to_float(r, "crc_status", 1)),
            snr_bin=_to_str(r, "snr_bin", ""),
        )


class CloudRTTTrace(EmpiricalTraceSampler):
    """Sampler for normalized edge-to-cloud latency traces."""

    def __init__(self, csv_path: Path, sampling_mode: str = "sequential", seed: int = 1,
                 stress: Optional[dict] = None):
        super().__init__(csv_path, sampling_mode, seed)
        self.stress = stress or {"delay_multiplier": 1.0}
        self.regime_thresholds = None
        regime_path = project_path("data/fitted/network_regimes.json")
        if regime_path.exists():
            self.regime_thresholds = load_json(regime_path).get("eta_latency_quantile_thresholds", None)

    def regime_for_eta(self, eta: float) -> tuple[int, str]:
        if not self.regime_thresholds:
            return 0, "uncalibrated"
        q25, q50, q75, q95 = [float(x) for x in self.regime_thresholds]
        if eta <= q25:
            return 0, "Q1_low_latency_burden"
        if eta <= q50:
            return 1, "Q2_medium_latency_burden"
        if eta <= q75:
            return 2, "Q3_high_latency_burden"
        if eta <= q95:
            return 3, "Q4_severe_latency_burden"
        return 4, "Q5_tail_latency_burden"

    def sample(self, eta_hint: Optional[float] = None) -> CloudRTTSample:
        r = self.next_row()
        dmul = float(self.stress.get("delay_multiplier", 1.0))
        raw_rtt_ms = _to_float(r, "rtt_ms", _to_float(r, "raw_rtt_ms", 100.0))
        one_way_ms = _to_float(r, "one_way_delay_ms", raw_rtt_ms / 2.0)
        raw_rtt_ms = max(1e-3, raw_rtt_ms * dmul)
        one_way_ms = max(1e-3, one_way_ms * dmul)
        rid, rname = self.regime_for_eta(float(eta_hint)) if eta_hint is not None else (0, "unassigned")
        return CloudRTTSample(
            sample_id=int(_to_float(r, "sample_id", self.idx)),
            probe_id=int(_to_float(r, "probe_id", _to_float(r, "prb", -1))),
            cloud_region=_to_str(r, "cloud_region", ""),
            rtt_ms=float(raw_rtt_ms),
            one_way_delay_ms=float(one_way_ms),
            min_cloud_region=_to_str(r, "min_cloud_region", ""),
            min_rtt_ms=_to_float(r, "min_rtt_ms", np.nan),
            regime_id=int(rid),
            regime_name=rname,
        )


# ---------------------------------------------------------------------------
# LoRaWAN airtime, reliability, and preprocessing
# ---------------------------------------------------------------------------

def _parse_bandwidth_khz(value) -> float:
    bw = float(value)
    return bw / 1000.0 if bw > 1000 else bw


def _payload_size_from_physical_payload(s: str) -> float:
    try:
        if not isinstance(s, str) or not s.strip():
            return np.nan
        return float(len(base64.b64decode(s, validate=False)))
    except Exception:
        return np.nan


def _parse_code_rate(value) -> int:
    """Return LoRa CR denominator offset: 1 for 4/5, 2 for 4/6, ..."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 1
    s = str(value).strip()
    if "/" in s:
        try:
            den = int(s.split("/")[-1])
            return int(np.clip(den - 4, 1, 4))
        except Exception:
            return 1
    try:
        v = int(float(s))
        return int(np.clip(v, 1, 4))
    except Exception:
        return 1


def lora_time_on_air_s(payload_bytes: int, sf: int, bandwidth_khz: float,
                       coding_rate: int = 1, preamble_symbols: int = 8,
                       explicit_header: bool = True, crc: bool = True) -> float:
    """Compute LoRa packet time-on-air in seconds.

    This follows the common Semtech LoRa time-on-air expression. `coding_rate`
    is 1 for 4/5, 2 for 4/6, 3 for 4/7, and 4 for 4/8.
    """
    sf = int(sf)
    bw_hz = float(bandwidth_khz) * 1000.0
    pl = max(1, int(payload_bytes))
    de = 1 if (sf >= 11 and bw_hz <= 125000) else 0
    ih = 0 if explicit_header else 1
    crc_i = 1 if crc else 0
    cr = int(np.clip(coding_rate, 1, 4))
    t_sym = (2 ** sf) / bw_hz
    payload_arg = (8 * pl - 4 * sf + 28 + 16 * crc_i - 20 * ih) / (4 * (sf - 2 * de))
    payload_symbols = 8 + max(math.ceil(payload_arg) * (cr + 4), 0)
    return float((preamble_symbols + 4.25 + payload_symbols) * t_sym)


def expected_attempts_truncated(p_success: float, max_attempts: int) -> float:
    """Expected transmissions for a truncated geometric retransmission process.

    The final term includes all cases where the packet has not succeeded before
    the maximum attempt; this is standard for expected service-time modeling.
    """
    p = float(np.clip(p_success, 1e-6, 1.0))
    nmax = max(1, int(max_attempts))
    if nmax == 1:
        return 1.0
    exp_n = 0.0
    for n in range(1, nmax):
        exp_n += n * ((1.0 - p) ** (n - 1)) * p
    exp_n += nmax * ((1.0 - p) ** (nmax - 1))
    return float(exp_n)


def estimate_success_probability(valid: pd.DataFrame, snr_bin_width: float = 2.0,
                                 smoothing_alpha: float = 1.0,
                                 smoothing_beta: float = 1.0,
                                 min_group_count: int = 5) -> pd.DataFrame:
    """Estimate P(CRC=1 | gateway, SF, BW, SNR bin) with beta smoothing.

    This implementation uses group transforms so the row count is preserved.
    """
    v = valid.copy()
    lo = math.floor(float(v["snr"].min()) / snr_bin_width) * snr_bin_width
    hi = math.ceil(float(v["snr"].max()) / snr_bin_width) * snr_bin_width
    if hi <= lo:
        hi = lo + snr_bin_width
    bins = np.arange(lo, hi + snr_bin_width, snr_bin_width)
    v["snr_bin"] = pd.cut(v["snr"], bins=bins, include_lowest=True).astype(str)

    global_p = (v["crc_status_num"].sum() + smoothing_alpha) / (len(v) + smoothing_alpha + smoothing_beta)

    keys = ["gateway", "spreading_factor", "bandwidth", "snr_bin"]
    g = v.groupby(keys)["crc_status_num"]
    g_sum = g.transform("sum")
    g_count = g.transform("count")
    p_group = (g_sum + smoothing_alpha) / (g_count + smoothing_alpha + smoothing_beta)
    p_group = p_group.where(g_count >= min_group_count, np.nan)

    keys2 = ["spreading_factor", "bandwidth"]
    g2 = v.groupby(keys2)["crc_status_num"]
    g2_sum = g2.transform("sum")
    g2_count = g2.transform("count")
    p_sf_bw = (g2_sum + smoothing_alpha) / (g2_count + smoothing_alpha + smoothing_beta)

    v["p_success"] = p_group.fillna(p_sf_bw).fillna(float(global_p)).clip(1e-3, 0.999)
    v["p_success_group_count"] = g_count.astype(int)
    return v

def normalize_loed_access_trace(df: pd.DataFrame, gateway_strategy: str = "largest",
                                gateway_id: Optional[str] = None,
                                payload_bytes_override: Optional[int] = None,
                                snr_bin_width: float = 2.0,
                                max_attempts: int = 3,
                                retransmission_backoff_s: float = 1.0,
                                smoothing_alpha: float = 1.0,
                                smoothing_beta: float = 1.0) -> pd.DataFrame:
    """Normalize raw LoED table into an access trace.

    The normalized trace includes radio features, LoRa airtime, empirical packet
    success probability, expected attempts, and expected access delay.  It keeps
    failed CRC rows because they are needed to estimate link reliability.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    required = ["crc_status", "rssi", "snr", "spreading_factor", "bandwidth", "gateway"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"LoED access trace is missing required columns: {missing}")

    out["crc_status_num"] = pd.to_numeric(out["crc_status"], errors="coerce")
    # Normalize common CRC encodings: True/False may become nan above.
    if out["crc_status_num"].isna().any():
        s = out["crc_status"].astype(str).str.lower().str.strip()
        out.loc[s.isin(["1", "true", "ok", "valid", "crc_ok", "success"]), "crc_status_num"] = 1
        out.loc[s.isin(["0", "false", "bad", "invalid", "crc_bad", "fail", "failed"]), "crc_status_num"] = 0
    out["rssi"] = pd.to_numeric(out["rssi"], errors="coerce")
    out["snr"] = pd.to_numeric(out["snr"], errors="coerce")
    out["spreading_factor"] = pd.to_numeric(out["spreading_factor"], errors="coerce")
    out["bandwidth"] = pd.to_numeric(out["bandwidth"], errors="coerce").apply(lambda x: _parse_bandwidth_khz(x) if pd.notna(x) else np.nan)
    out["frequency"] = pd.to_numeric(out["frequency"], errors="coerce") if "frequency" in out.columns else np.nan

    valid = out[
        out["crc_status_num"].isin([0, 1])
        & out["rssi"].notna()
        & out["snr"].notna()
        & out["spreading_factor"].between(7, 12)
        & out["bandwidth"].notna()
    ].copy()
    if valid.empty:
        raise ValueError("No valid LoED rows after CRC/RSSI/SNR/SF/BW filtering.")

    if gateway_id is not None:
        valid = valid[valid["gateway"].astype(str) == str(gateway_id)].copy()
        if valid.empty:
            raise ValueError(f"No valid LoED rows for gateway_id={gateway_id}")
    elif gateway_strategy == "largest":
        selected = valid["gateway"].astype(str).value_counts().idxmax()
        valid = valid[valid["gateway"].astype(str) == selected].copy()
    elif gateway_strategy == "all":
        pass
    else:
        raise ValueError("gateway_strategy must be 'largest' or 'all'.")

    if payload_bytes_override is not None:
        payload_size = np.full(len(valid), float(payload_bytes_override))
    elif "size" in valid.columns:
        size_num = pd.to_numeric(valid["size"], errors="coerce")
        decoded = valid.get("physical_payload", pd.Series([np.nan] * len(valid), index=valid.index)).apply(_payload_size_from_physical_payload)
        payload_size = size_num.where(size_num > 0, decoded).fillna(51.0).astype(float)
    else:
        payload_size = valid.get("physical_payload", pd.Series([np.nan] * len(valid), index=valid.index)).apply(_payload_size_from_physical_payload).fillna(51.0).astype(float)

    if "code_rate" in valid.columns:
        cr = valid["code_rate"].apply(_parse_code_rate).astype(int)
    else:
        cr = pd.Series([1] * len(valid), index=valid.index)

    valid = estimate_success_probability(
        valid,
        snr_bin_width=snr_bin_width,
        smoothing_alpha=smoothing_alpha,
        smoothing_beta=smoothing_beta,
    )

    airtime = np.array([
        lora_time_on_air_s(int(pl), int(sf), float(bw), int(c))
        for pl, sf, bw, c in zip(payload_size, valid["spreading_factor"], valid["bandwidth"], cr)
    ], dtype=float)
    exp_attempts = np.array([expected_attempts_truncated(p, max_attempts) for p in valid["p_success"]], dtype=float)
    access_delay = exp_attempts * airtime + np.maximum(exp_attempts - 1.0, 0.0) * float(retransmission_backoff_s)

    norm = pd.DataFrame({
        "sample_id": np.arange(len(valid)),
        "time": valid["time"].astype(str) if "time" in valid.columns else "",
        "gateway": valid["gateway"].astype(str),
        "device_address": valid["device_address"].astype(str) if "device_address" in valid.columns else "",
        "crc_status": valid["crc_status_num"].astype(int),
        "frequency": valid["frequency"].astype(float),
        "spreading_factor": valid["spreading_factor"].astype(int),
        "bandwidth": valid["bandwidth"].astype(float),
        "rssi": valid["rssi"].astype(float),
        "snr": valid["snr"].astype(float),
        "snr_bin": valid["snr_bin"].astype(str),
        "size": payload_size.astype(float),
        "airtime_s": airtime,
        "p_success": valid["p_success"].astype(float),
        "expected_attempts": exp_attempts,
        "access_delay_s": access_delay,
        "max_attempts": int(max_attempts),
        "retransmission_backoff_s": float(retransmission_backoff_s),
    }).reset_index(drop=True)
    norm["sample_id"] = np.arange(len(norm))
    return norm


# ---------------------------------------------------------------------------
# Cloud-edge RTT preprocessing
# ---------------------------------------------------------------------------

def normalize_cloud_edge_rtt_trace(df: pd.DataFrame, cloud_label: str = "Milan-MIL01.csv",
                                   mode: str = "fixed") -> pd.DataFrame:
    """Normalize cloud-edge latency table into cloud_rtt_trace.csv.

    The raw dataset reports RTT.  The output stores raw RTT in ``rtt_ms`` and
    stores half of it in ``one_way_delay_ms``.  Cloud AoP uses this one-way
    upload delay only.
    """
    raw = df.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if "prb" not in raw.columns:
        raise ValueError("Cloud RTT table must contain a 'prb' probe-id column.")

    records = []
    for _, row in raw.iterrows():
        if pd.isna(row.get("prb")):
            continue
        probe_id = int(row["prb"])
        min_rtt = float(row["minMedian"]) if "minMedian" in raw.columns and pd.notna(row.get("minMedian")) else np.nan
        min_label = str(row["minLabel"]) if "minLabel" in raw.columns and pd.notna(row.get("minLabel")) else ""

        if mode == "min":
            if not np.isfinite(min_rtt):
                continue
            rtt = min_rtt
            region = min_label
        elif mode == "fixed":
            rtt = None
            region = cloud_label
            label_cols = [c for c in raw.columns if c == "label" or c.startswith("label.")]
            for lcol in label_cols:
                if str(row.get(lcol, "")).strip() == cloud_label:
                    suffix = "" if lcol == "label" else lcol.replace("label", "")
                    vcol = "value" + suffix
                    if vcol in raw.columns and pd.notna(row.get(vcol)):
                        rtt = float(row[vcol])
                        break
            if rtt is None:
                continue
        else:
            raise ValueError("mode must be 'fixed' or 'min'.")

        if not np.isfinite(rtt) or rtt <= 0:
            continue
        records.append({
            "sample_id": len(records),
            "probe_id": probe_id,
            "cloud_region": region,
            "rtt_ms": float(rtt),
            "one_way_delay_ms": float(rtt) / 2.0,
            "min_cloud_region": min_label,
            "min_rtt_ms": float(min_rtt) if np.isfinite(min_rtt) else np.nan,
        })

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError(f"No cloud RTT rows extracted for mode={mode}, cloud_label={cloud_label}")
    return out
