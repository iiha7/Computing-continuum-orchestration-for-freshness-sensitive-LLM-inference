"""Digital-twin environment for IoT-gated edge/cloud LLM orchestration.

Actions after the IoT pre-transmission gate:
    0 = edge-only
    1 = cloud-only

Before either edge/cloud action is executed, the IoT layer computes/observes the
Fire Weather Index (FWI). If FWI < 11.2, the reading is classified locally as
Low danger and is not transmitted to the edge. This models a lightweight IoT
pre-filter that reduces unnecessary access traffic for very low-risk readings.

Communication model:
    * IoT -> Edge: LoED LoRaWAN trace. Access delay is LoRa time-on-air plus
      expected retransmission overhead estimated from empirical CRC success
      probability. RSSI/SNR/SF/BW are state features.
    * Edge -> Cloud: cloud-edge latency trace. Cloud AoP uses the uplink
      one-way delay plus cloud processing. No cloud downlink delay is modeled
      in the current version. The normalized trace stores both raw RTT and
      one-way delay.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .common import load_json, load_yaml, project_path, read_jsonl
from .freshness import deadline_from_class, freshness_utility
from .network_traces import AccessTrace, CloudRTTTrace

ACTION_NAMES = {0: "edge_only", 1: "cloud_only"}
COMPLEXITY_ID = {"simple": 0, "medium": 1, "complex": 2}
MODE_TO_PROFILE = {0: "edge", 1: "cloud"}


class IoTEdgeCloudEnv:
    """Gym-like environment with reset() and step(action)."""

    def __init__(self, split="train", network_profile="nominal", deadline_profile="calibrated",
             seed=1, max_steps: Optional[int] = None, enforce_cloud_budget: Optional[bool] = None):
        self.cfg = load_yaml(project_path("configs/experiment.yaml"))
        self.net_cfg = load_yaml(project_path("configs/network_traces.yaml"))["network_model"]
        self.norm = self._load_normalization_constants()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.split = split
        self.network_profile_name = network_profile
        self.deadline_profile = deadline_profile
        self.max_steps = max_steps or int(self.cfg["simulation"]["max_steps_per_episode"])

        budget_cfg = self.cfg.get("orchestration", {}).get("cloud_budget", {})
        self.enforce_cloud_budget = bool(
            budget_cfg.get("enabled", False) if enforce_cloud_budget is None else enforce_cloud_budget
        )
        self.cloud_budget_adaptive = bool(budget_cfg.get("adaptive", False))
        self.cloud_budget_fraction = float(budget_cfg.get("cloud_budget_fraction", 1.0))
        self.cloud_budget_used = 0
        self.cloud_budget_limit = max(1, int(np.floor(self.cloud_budget_fraction * self.max_steps)))
        self.current_cloud_budget_fraction = self.cloud_budget_fraction
        self.current_budget_regime = "unassigned"

        self.records = pd.read_csv(project_path(self.cfg["paths"]["records"]))
        self.prompts = pd.read_csv(project_path(self.cfg["paths"]["prompts"]))
        split_df = pd.read_csv(project_path(self.cfg["paths"]["split"]))
        ids = split_df.loc[split_df.split == split, "record_id"].tolist()
        self.records = self.records[self.records.record_id.isin(ids)].reset_index(drop=True)
        self.prompts = self.prompts[self.prompts.record_id.isin(ids)].reset_index(drop=True)

        self.edge_profiles = self._load_profile("edge_outputs.jsonl")
        self.cloud_profiles = self._load_profile("cloud_outputs.jsonl")
        self.size_predictor = joblib.load(project_path("data/fitted/response_size_predictor.pkl"))

        stress = self.net_cfg["stress_tests"][network_profile]
        sampling_mode = self.net_cfg.get("sampling_mode", "sequential")
        self.access_trace = AccessTrace(project_path(self.net_cfg["access_trace_path"]), sampling_mode, seed=seed)
        self.cloud_rtt_trace = CloudRTTTrace(project_path(self.net_cfg["cloud_rtt_trace_path"]), sampling_mode, seed=seed + 13, stress=stress)

        self.fresh_calib = self._load_freshness_calibration()
        self.t = 0
        self.previous_action = 0
        self.battery = float(self.cfg["simulation"]["initial_battery"])
        
        self.current_prompt = None
        self.current_access = None
        self.current_channel = None

    def _load_normalization_constants(self) -> Dict:
        norm = dict(self.cfg.get("normalization", {}))
        path = norm.get("fitted_path", "data/fitted/normalization.json")
        p = project_path(path)
        if p.exists():
            fitted = load_json(p)
            for k, v in fitted.items():
                if k.startswith("max_"):
                    norm[k] = float(v)
            norm["source"] = str(p)
        else:
            norm["source"] = "configs/experiment.yaml fallback"
        self.cfg["normalization"] = norm
        return norm

    def _load_freshness_calibration(self) -> Dict:
        path = project_path("data/fitted/freshness_calibration.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "sampling_period_s": float(self.cfg["freshness"]["sampling_period_s"]),
            "alpha_max": float(self.cfg["freshness"]["alpha_max"]),
            "alpha_min_default": 0.15,
            "utility_at_deadline": float(self.cfg["freshness"]["utility_at_deadline"]),
        }

    def _load_profile(self, filename: str) -> Dict[str, Dict]:
        return {r["prompt_id"]: r for r in read_jsonl(project_path("data/llm_profiles", filename))}

    def _cloud_budget_fraction_for_episode(self) -> float:
        """Choose an episode cloud budget from current edge-cloud latency burden."""
        budget_cfg = self.cfg.get("orchestration", {}).get("cloud_budget", {})

        if not self.enforce_cloud_budget:
            self.current_budget_regime = "disabled"
            return 1.0

        if not bool(budget_cfg.get("adaptive", False)):
            self.current_budget_regime = "fixed"
            return float(budget_cfg.get("cloud_budget_fraction", 1.0))

        regime_name = getattr(self.current_channel, "regime_name", "unassigned")
        self.current_budget_regime = str(regime_name)

        regime_map = budget_cfg.get("fractions_by_latency_regime", {})
        if regime_name in regime_map:
            return float(regime_map[regime_name])

        profile_map = budget_cfg.get("fractions_by_network_profile", {})
        if self.network_profile_name in profile_map:
            self.current_budget_regime = str(self.network_profile_name)
            return float(profile_map[self.network_profile_name])

        return float(budget_cfg.get("cloud_budget_fraction", 1.0))


    def _reset_cloud_budget_for_episode(self) -> None:
        self.cloud_budget_used = 0
        self.current_cloud_budget_fraction = float(np.clip(self._cloud_budget_fraction_for_episode(), 0.0, 1.0))
        self.cloud_budget_limit = max(
            0,
            int(np.floor(self.current_cloud_budget_fraction * self.max_steps))
        )


    def _cloud_budget_remaining_fraction(self) -> float:
        if not self.enforce_cloud_budget:
            return 1.0
        if self.cloud_budget_limit <= 0:
            return 0.0
        return max(0.0, (self.cloud_budget_limit - self.cloud_budget_used) / self.cloud_budget_limit)


    def _cloud_budget_used_fraction(self) -> float:
        if not self.enforce_cloud_budget:
            return 0.0
        if self.cloud_budget_limit <= 0:
            return 1.0
        return min(1.0, self.cloud_budget_used / self.cloud_budget_limit)


    def _cloud_allowed_by_budget(self) -> bool:
        if not self.enforce_cloud_budget:
            return True
        return self.cloud_budget_used < self.cloud_budget_limit
    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.t = 0
        self.previous_action = 0
        self.battery = float(self.cfg["simulation"]["initial_battery"])
        self.current_prompt = self._sample_prompt()
        self.current_access = self.access_trace.sample()
        self.current_channel = self._sample_cloud_delay_for_prompt(self.current_prompt)
        self._reset_cloud_budget_for_episode()
        return self._get_state(), {}

    def step(self, action: int):
        info = self.simulate_action(action)

        if self.enforce_cloud_budget and int(info.get("cloud_used", 0)) == 1:
            self.cloud_budget_used += 1

        reward = self._compute_reward(info, action)
        info["reward"] = reward
        self.previous_action = int(action)
        self.battery = max(0.0, self.battery - info["energy_j"] * 0.0005)
        self.t += 1
        terminated = False
        truncated = self.t >= self.max_steps
        self.current_prompt = self._sample_prompt()
        self.current_access = self.access_trace.sample()
        self.current_channel = self._sample_cloud_delay_for_prompt(self.current_prompt)
        return self._get_state(), float(reward), terminated, truncated, info

    def _deadline_multiplier(self) -> float:
        mapping = {
            "calibrated": 1.0,
            "criticality_dependent": 1.0,
            "normal": 1.0,
            "urgency_0.5x": 0.5,
            "urgency_1.0x": 1.0,
            "urgency_1.5x": 1.5,
            "urgency_2.0x": 2.0,
            "strict": 0.5,
            "relaxed": 2.0,
        }
        if self.deadline_profile not in mapping:
            raise ValueError(f"Unknown deadline profile: {self.deadline_profile}")
        return float(mapping[self.deadline_profile])

    def _deadline_for_class(self, class_id: int) -> float:
        fcfg = self.cfg["freshness"]
        alpha_min = float(self.fresh_calib["alpha_min_default"]) * self._deadline_multiplier()
        lo, hi = [float(v) for v in fcfg["alpha_min_clip"]]
        alpha_min = float(np.clip(alpha_min, lo, hi))
        return deadline_from_class(
            class_id=int(class_id),
            sampling_period_s=float(self.fresh_calib["sampling_period_s"]),
            alpha_max=float(self.fresh_calib["alpha_max"]),
            alpha_min=alpha_min,
        )

    def _task_deadline_multiplier(self, complexity: str) -> float:
        multipliers = self.cfg.get("freshness", {}).get("task_deadline_multipliers", {})
        return float(multipliers.get(str(complexity), 1.0))

    def _deadline_for_prompt(self, prompt: Dict) -> float:
        # Base deadline is criticality/FWI-class dependent. A task multiplier then
        # gives richer medium/complex decision-support reports more time than a
        # simple alert. This does not require LLM re-profiling.
        base = self._deadline_for_class(int(prompt["danger_class"]))
        return float(base * self._task_deadline_multiplier(str(prompt.get("complexity", "simple"))))

    def _task_quality_multiplier(self, complexity: str) -> float:
        multipliers = self.cfg.get("reward", {}).get("task_quality_multipliers", {})
        return float(multipliers.get(str(complexity), 1.0))

    def _cloud_latency_scale(self) -> float:
        # Cloud profiles may be produced on the local workstation. Scaling lets the
        # simulator represent GPU/server-side cloud inference latency without
        # re-running expensive LLM calls. Edge latency remains measured/unscaled.
        return float(self.cfg.get("simulation", {}).get("cloud_llm", {}).get("latency_scale", 1.0))

    def _sample_prompt(self) -> Dict:
        probs = self.cfg["simulation"]["complexity_probs"]
        comps = list(probs.keys())
        p = np.array([probs[c] for c in comps], dtype=float)
        p /= p.sum()
        comp = self.rng.choice(comps, p=p)
        subset = self.prompts[self.prompts.complexity == comp]
        d = subset.iloc[int(self.rng.integers(0, len(subset)))].to_dict()
        d["deadline_s"] = self._deadline_for_prompt(d)
        return d

    def _sample_cloud_delay_for_prompt(self, prompt: Dict):
        ch = self.cloud_rtt_trace.sample(eta_hint=None)
        eta = (ch.one_way_delay_ms / 1000.0) / max(float(prompt["deadline_s"]), 1e-12)
        rid, rname = self.cloud_rtt_trace.regime_for_eta(eta)
        ch.regime_id = rid
        ch.regime_name = rname
        return ch

    def _iot_filter_enabled(self) -> bool:
        return bool(self.cfg.get("simulation", {}).get("iot_filter", {}).get("enabled", True))

    def _iot_should_transmit(self, prompt: Dict) -> bool:
        if not self._iot_filter_enabled():
            return True
        if "iot_transmit" in prompt and not pd.isna(prompt["iot_transmit"]):
            return int(prompt["iot_transmit"]) == 1
        threshold = float(self.cfg["simulation"]["iot_filter"].get("fwi_threshold", 11.2))
        return float(prompt.get("iot_fwi", 999.0)) >= threshold

    def _iot_access_delay(self) -> float:
        return float(getattr(self.current_access, "access_delay_s", self.current_access.airtime_s))

    def _profile_for_action(self, action: int) -> Dict:
        pid = self.current_prompt["prompt_id"]
        if action == 0:
            return self.edge_profiles[pid]
        if action == 1:
            return self.cloud_profiles[pid]
        raise ValueError(action)

    def _prompt_bytes_for_mode(self, prompt: Dict, mode_role: str) -> int:
        """Return the shared prompt size seen by both edge and cloud."""
        return int(prompt.get("prompt_bytes", prompt.get("edge_prompt_bytes", prompt.get("cloud_prompt_bytes", 0))))

    def _predict_response_size_for_prompt(self, prompt: Dict, mode_role: str, quantile="p90") -> float:
        x = pd.DataFrame([{
            "complexity": prompt["complexity"],
            "mode_role": mode_role,
            "prompt_bytes": self._prompt_bytes_for_mode(prompt, mode_role),
            "criticality": prompt["criticality"],
            "deadline_s": prompt["deadline_s"],
        }])
        return float(max(1.0, self.size_predictor[quantile].predict(x)[0]))

    def _predict_response_size(self, mode_role: str, quantile="p90") -> float:
        return self._predict_response_size_for_prompt(self.current_prompt, mode_role, quantile)

    @staticmethod
    def _comm_bit_energy(bits: float, coeff_j_per_kbit: float) -> float:
        return float(coeff_j_per_kbit) * float(bits) / 1000.0

    def _iot_no_transmit_info(self, requested_action: int) -> Dict:
        p = self.current_prompt
        fcfg = self.cfg["simulation"].get("iot_filter", {})
        pred = int(fcfg.get("local_predicted_class", 0))
        gt = int(p["ground_truth_tier"])
        aoi = float(fcfg.get("local_latency_s", 0.005))
        energy = float(fcfg.get("local_energy_j", 0.002))
        deadline = float(p["deadline_s"])
        # Low-risk IoT-local gate is deterministic and considered a correct/complete local decision.
        quality = 1.0 if pred == gt else 0.0
        correct = int(quality >= float(self.cfg.get("evaluation", {}).get("quality_threshold", 0.75)))
        stale = int(aoi > deadline)
        fresh_u = freshness_utility(aoi, deadline, float(self.cfg["freshness"]["utility_at_deadline"]))
        return self._info_dict(
            p=p, requested_action=requested_action, effective_mode="iot_no_transmit",
            pred=pred, correct=correct, response_quality=quality, confidence=1.0, aoi=aoi, deadline=deadline,
            stale=stale, fresh_u=fresh_u, bits=0.0, uplink_bits=0.0, downlink_bits=0.0,
            energy=energy, edge_energy=energy, comm_energy=0.0, cost=0.0,
            cloud_used=0, iot_transmitted=0, profile={},
        )
    
    def _cloud_budget_remaining_fraction(self) -> float:
        if not self.enforce_cloud_budget:
            return 1.0
        return max(0.0, (self.cloud_budget_limit - self.cloud_budget_used) / max(1, self.cloud_budget_limit))


    def _cloud_allowed_by_budget(self) -> bool:
        if not self.enforce_cloud_budget:
            return True
        return self.cloud_budget_used < self.cloud_budget_limit

    def simulate_action(self, action: int, use_predicted_response=False, size_quantile="p90") -> Dict:
        requested_action = int(action)
        action = int(action)

        if action not in ACTION_NAMES:
            raise ValueError(action)

        p = self.current_prompt
        if not self._iot_should_transmit(p):
            return self._iot_no_transmit_info(requested_action)

        # Budget-aware orchestration: if cloud budget is exhausted, a requested
        # cloud action is executed at the edge instead.
        if action == 1 and not self._cloud_allowed_by_budget():
            action = 0

        ch = self.current_channel
        t_iot = self._iot_access_delay()
        e_cfg = self.cfg["energy"]
        prof = self._profile_for_action(action)
        mode = MODE_TO_PROFILE[action]
        resp = self._predict_response_size(mode, size_quantile) if use_predicted_response else prof["response_bytes"]
        raw_lat = float(prof["latency_s"])
        lat = raw_lat * (self._cloud_latency_scale() if action == 1 else 1.0)
        quality = float(prof.get("response_quality", float(prof.get("correct", 0))))
        correct = int(prof.get("correct", int(quality >= float(self.cfg.get("evaluation", {}).get("quality_threshold", 0.75)))))
        pred = int(prof.get("predicted_tier", p.get("ground_truth_tier", 0)))
        conf = float(prof.get("confidence", prof.get("calibrated_edge_confidence", 0.0)) or 0.0)
        cost = float(prof["cost_usd"])
        edge_energy = comm_energy = 0.0
        uplink_bits = downlink_bits = 0.0
        cloud_used = 0

        if action == 0:
            # Age-of-processing: generation -> IoT access -> edge processing done.
            aoi = t_iot + lat
            edge_energy = float(e_cfg["p_edge_infer_w"]) * lat
        else:
            cloud_used = 1
            # Age-of-processing: generation -> IoT access -> edge-cloud upload ->
            # cloud processing. No cloud downlink delay is modeled in this version.
            aoi = t_iot + ch.one_way_delay_ms / 1000.0 + lat
            uplink_bits = float(prof["prompt_bytes"]) * 8.0
            downlink_bits = float(resp) * 8.0

        bits = uplink_bits + downlink_bits
        comm_energy = self._comm_bit_energy(uplink_bits, e_cfg.get("e_tx_j_per_kbit", 0.0)) + \
            self._comm_bit_energy(downlink_bits, e_cfg.get("e_rx_j_per_kbit", 0.0))
        energy = edge_energy + comm_energy
        deadline = float(p["deadline_s"])
        stale = int(aoi > deadline)
        fresh_u = freshness_utility(aoi, deadline, float(self.cfg["freshness"]["utility_at_deadline"]))
        return self._info_dict(
            p=p, requested_action=requested_action, effective_mode=ACTION_NAMES[action],
            pred=pred, correct=correct, response_quality=quality, confidence=conf, aoi=aoi, deadline=deadline,
            stale=stale, fresh_u=fresh_u, bits=bits, uplink_bits=uplink_bits, downlink_bits=downlink_bits,
            energy=energy, edge_energy=edge_energy, comm_energy=comm_energy, cost=cost,
            cloud_used=cloud_used, iot_transmitted=1, profile=prof,
        )

    def _info_dict(self, p: Dict, requested_action: int, effective_mode: str, pred: int, correct: int,
                   response_quality: float, confidence: float, aoi: float, deadline: float, stale: int, fresh_u: float,
                   bits: float, uplink_bits: float, downlink_bits: float, energy: float,
                   edge_energy: float, comm_energy: float, cost: float, cloud_used: int,
                   iot_transmitted: int, profile: Dict | None = None) -> Dict:
        ch = self.current_channel
        profile = profile or {}
        task_value_multiplier = self._task_quality_multiplier(str(p.get("complexity", "simple")))
        fresh_value = float(response_quality) * fresh_u
        value_weighted_fresh_quality = task_value_multiplier * fresh_value
        return {
            "record_id": int(p["record_id"]),
            "prompt_id": p["prompt_id"],
            "complexity": p["complexity"],
            "action": int(requested_action),
            "requested_mode": ACTION_NAMES.get(int(requested_action), "unknown"),
            "mode": effective_mode,
            "ground_truth_tier": int(p["ground_truth_tier"]),
            "predicted_tier": int(pred),
            "ground_truth_danger_class": int(p["ground_truth_tier"]),
            "predicted_danger_class": int(pred),
            "correct": int(correct),
            "response_quality": float(response_quality),
            "stale": int(stale),
            "fresh_correct": int(correct == 1 and stale == 0),
            "fresh_quality": float(fresh_value),
            "freshness_utility": float(fresh_u),
            "fresh_value": float(fresh_value),
            "task_value_multiplier": float(task_value_multiplier),
            "value_weighted_fresh_quality": float(value_weighted_fresh_quality),
            "escalation_correct": int(profile.get("escalation_correct", correct)),
            "risk_factor_f1": float(profile.get("risk_factor_f1", response_quality)),
            "action_correct": int(profile.get("action_correct", correct)),
            "urgency_correct": int(profile.get("urgency_correct", correct)),
            "verification_correct": int(profile.get("verification_correct", correct)),
            "aoi_s": float(aoi),
            "aop_s": float(aoi),
            "profile_latency_s": float(profile.get("latency_s", 0.0) or 0.0),
            "effective_processing_latency_s": float((profile.get("latency_s", 0.0) or 0.0) * (self._cloud_latency_scale() if effective_mode == "cloud_only" else 1.0)),
            "cloud_latency_scale": float(self._cloud_latency_scale() if effective_mode == "cloud_only" else 1.0),
            "access_delay_s": float(self._iot_access_delay()) if iot_transmitted else 0.0,
            "deadline_s": float(deadline),
            "bits": float(bits),
            "uplink_bits": float(uplink_bits),
            "downlink_bits": float(downlink_bits),
            "energy_j": float(energy),
            "edge_compute_energy_j": float(edge_energy),
            "comm_energy_j": float(comm_energy),
            "cost_usd": float(cost),
            "cloud_used": int(cloud_used),
            "cloud_budget_enabled": int(self.enforce_cloud_budget),
            "cloud_budget_adaptive": int(self.cloud_budget_adaptive),
            "cloud_budget_fraction": float(self.current_cloud_budget_fraction),
            "cloud_budget_limit": int(self.cloud_budget_limit),
            "cloud_budget_used_so_far": int(self.cloud_budget_used),
            "cloud_budget_remaining_fraction": float(self._cloud_budget_remaining_fraction()),
            "cloud_budget_used_fraction": float(self._cloud_budget_used_fraction()),
            "cloud_budget_regime": str(self.current_budget_regime),
            "iot_transmitted": int(iot_transmitted),
            "iot_gate_active": int(self._iot_filter_enabled()),
            "iot_fwi": float(p.get("iot_fwi", np.nan)),
            "channel_profile": self.network_profile_name,
            "network_condition": self.network_profile_name,
            "channel_state": ch.regime_name,
            "network_regime": ch.regime_name,
            "network_regime_id": int(ch.regime_id),
            "rtt_ms": ch.rtt_ms,
            "cloud_one_way_delay_ms": ch.one_way_delay_ms,
            "cloud_path_delay_s": float(ch.one_way_delay_ms / 1000.0),
            "cloud_region": ch.cloud_region,
            "probe_id": ch.probe_id,
            "min_cloud_region": ch.min_cloud_region,
            "min_rtt_ms": ch.min_rtt_ms,
            "access_gateway": self.current_access.gateway,
            "access_device_address": self.current_access.device_address,
            "access_rssi_dbm": float(self.current_access.rssi_dbm),
            "access_snr_db": float(self.current_access.snr_db),
            "access_spreading_factor": int(self.current_access.spreading_factor),
            "access_bandwidth_khz": float(self.current_access.bandwidth_khz),
            "access_airtime_s": float(self.current_access.airtime_s),
            "access_p_success": float(getattr(self.current_access, "p_success", 1.0)),
            "access_expected_attempts": float(getattr(self.current_access, "expected_attempts", 1.0)),
            "access_crc_status": int(getattr(self.current_access, "crc_status", 1)),
            "criticality": float(p["criticality"]),
            "confidence": float(confidence),
            "previous_action": int(self.previous_action),
        }

    def peek_action(self, action: int, use_predicted_response=True, size_quantile="p90") -> Dict:
        state = self.rng.bit_generator.state
        out = self.simulate_action(action, use_predicted_response, size_quantile)
        self.rng.bit_generator.state = state
        return out

    def _compute_reward(self, info: Dict, action: int) -> float:
        rw = self.cfg["reward"]
        norm = self.norm
        naoi = min(info["aoi_s"] / max(info["deadline_s"], 1e-6), 5.0)
        nbits = min(info["bits"] / max(norm["max_bits_per_request"], 1e-12), 5.0)
        nenergy = min(info["energy_j"] / max(norm["max_energy_j"], 1e-12), 5.0)
        ncost = min(info["cost_usd"] / max(norm["max_cost"], 1e-12), 5.0)
        reward = (rw["w_correct"] * info.get("value_weighted_fresh_quality", info["fresh_value"])
                  - rw["w_stale"] * info["stale"]
                  - rw["w_aoi"] * naoi
                  - rw["w_bits"] * nbits
                  - rw["w_energy"] * nenergy
                  - rw["w_cost"] * ncost)
        if info["criticality"] >= 0.99 and info["stale"]:
            reward -= rw["emergency_stale_extra_penalty"]
        return float(reward)

    def _get_state(self) -> np.ndarray:
        p = self.current_prompt
        ch = self.current_channel
        norm = self.norm
        r = self.records[self.records.record_id == int(p["record_id"])].iloc[0]
        ep = self.edge_profiles[p["prompt_id"]]
        pred_mean = self._predict_response_size("cloud", "mean")
        pred_p90 = self._predict_response_size("cloud", "p90")
        sensor = np.array([
            r.Temperature / 50,
            r.RH / 100,
            r.Ws / 50,
            r.Rain / 10,
            r.FFMC / 100,
            r.DMC / 120,
            r.ISI / 30,
            r.FWI / 70,
        ], dtype=np.float32)
        other = np.array([
            COMPLEXITY_ID[p["complexity"]] / 2,
            self._prompt_bytes_for_mode(p, "cloud") / norm["max_prompt_bytes"],
            pred_mean / norm["max_response_bytes"],
            pred_p90 / norm["max_response_bytes"],
            p["criticality"],
            p["deadline_s"] / float(self.fresh_calib["sampling_period_s"]),
            ep["confidence"],
            float(ep.get("edge_margin", ep.get("confidence", 0.5))),
            min(float(ep.get("edge_entropy", 1.0)) / 2.0, 1.0),
            float(p.get("computed_danger_class", p.get("danger_class", ep.get("computed_danger_class", 0)))) / 5.0,
            ch.regime_id / 4.0,
            ch.one_way_delay_ms / norm["max_rtt_ms"],
            (max(-140.0, min(-40.0, self.current_access.rssi_dbm)) + 140.0) / 100.0,
            np.clip((self.current_access.snr_db + 25.0) / 50.0, 0.0, 1.0),
            (self.current_access.spreading_factor - 7) / 5.0,
            min(self.current_access.access_delay_s / norm["max_access_delay_s"], 5.0),
            float(getattr(self.current_access, "p_success", 1.0)),
            min(float(getattr(self.current_access, "expected_attempts", 1.0)) / 5.0, 1.0),
            float(p.get("iot_fwi", r.FWI)) / 70.0,
            float(p.get("iot_transmit", 1)),
            self.battery,
            self.previous_action,
            self.current_cloud_budget_fraction,
            self._cloud_budget_remaining_fraction(),
            self._cloud_budget_used_fraction(),
            
        ], dtype=np.float32)
        return np.clip(np.concatenate([sensor, other]), 0.0, 5.0).astype(np.float32)

    @property
    def observation_dim(self) -> int:
        return len(self._get_state())

    @property
    def action_dim(self) -> int:
        return 2

    def feasible_actions_p90(self) -> List[int]:
        # If IoT gate suppresses the packet, edge/cloud action is ignored.
        if not self._iot_should_transmit(self.current_prompt):
            return [0, 1]

        feasible = [
            a for a in range(2)
            if self.peek_action(a, True, "p90")["aoi_s"] <= float(self.current_prompt["deadline_s"])
        ]

        if 0 not in feasible:
            feasible.append(0)

        # If cloud budget is exhausted, remove cloud from the feasible action set.
        if not self._cloud_allowed_by_budget() and 1 in feasible:
            feasible.remove(1)

        return sorted(set(feasible))
