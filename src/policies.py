"""Baseline policies for IoT-gated edge/cloud orchestration."""
from __future__ import annotations
import numpy as np
from .common import load_json, project_path

class BasePolicy:
    name = "BasePolicy"
    def act(self, env) -> int:
        raise NotImplementedError

class EdgeOnly(BasePolicy):
    """Always use the edge model after the IoT gate."""
    name = "EdgeOnly"
    def act(self, env) -> int:
        return 0

class CloudOnly(BasePolicy):
    """Always use the cloud model after the IoT gate."""
    name = "CloudOnly"
    def act(self, env) -> int:
        return 1

class RandomPolicy(BasePolicy):
    name = "Random"
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
    def act(self, env) -> int:
        return int(self.rng.integers(0, 2))

class EarlyExitOffloading(BasePolicy):
    """Calibrated early-exit/offloading baseline.

    The compact edge LLM is run for every transmitted request. Raw LLM
    self-confidence is not used.  ``src.fit_models`` calibrates ``confidence``
    as P(edge decision-support response is acceptable | observable edge-profile
    features).  The baseline accepts edge only when calibrated reliability is
    high and the computed FWI danger class is within the configured safe range;
    otherwise it offloads to cloud.
    """
    name = "EarlyExitOffloading"
    def __init__(self, threshold=None, safe_class=None):
        self.threshold = threshold
        self.safe_class = safe_class
        self._auto_threshold = None

    def _threshold(self, env) -> float:
        cfg = env.cfg.get("baselines", {}).get("early_exit", {})
        if self.threshold is not None:
            return float(self.threshold)
        configured = cfg.get("confidence_threshold", "auto")
        if configured != "auto":
            return float(configured)
        if self._auto_threshold is None:
            path = project_path("data/fitted/early_exit_threshold.json")
            if path.exists():
                obj = load_json(path)
                self._auto_threshold = float(obj.get("selected", {}).get("threshold", obj.get("confidence_threshold", 0.70)))
            else:
                self._auto_threshold = 0.70
        return float(self._auto_threshold)

    def act(self, env) -> int:
        cfg = env.cfg.get("baselines", {}).get("early_exit", {})
        theta = self._threshold(env)
        y_safe = int(self.safe_class if self.safe_class is not None else cfg.get("safe_class_max", 1))
        edge_profile = env.edge_profiles[env.current_prompt["prompt_id"]]
        confidence = float(edge_profile.get("calibrated_edge_confidence", edge_profile.get("confidence", 0.0)) or 0.0)
        response_valid = int(edge_profile.get("response_valid", 1))
        computed_class = int(env.current_prompt.get("computed_danger_class", env.current_prompt.get("danger_class", 5)))
        if response_valid == 1 and confidence >= theta and computed_class <= y_safe:
            return 0
        return 1

POLICY_REGISTRY = {
    "EdgeOnly": EdgeOnly,
    "CloudOnly": CloudOnly,
    "EarlyExitOffloading": EarlyExitOffloading,
    "Random": RandomPolicy,
}
# Main paper baselines. EarlyExitOffloading remains available by explicitly
# passing --policies EarlyExitOffloading, but is not evaluated by default.
BASELINE_NAMES = ["EdgeOnly", "CloudOnly"]
