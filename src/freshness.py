"""Freshness and deadline utilities for FWI danger classes.

The FWI thresholds define semantic severity only. They do not prescribe network
AoI deadlines. This module derives deadlines from the sensing period and from a
calibration of the fastest feasible processing path.
"""
from __future__ import annotations
import math
import numpy as np

N_DANGER_CLASSES = 6


def alpha_from_class(class_id: int, alpha_max: float, alpha_min: float, n_classes: int = N_DANGER_CLASSES) -> float:
    """Allowed fraction of the sensing interval for an FWI danger class.

    The interpolation is monotonic and is controlled by only two endpoints:
    alpha_max for class 0 and alpha_min for class n_classes-1. This avoids
    assigning arbitrary deadlines to each class.
    """
    c = int(np.clip(class_id, 0, n_classes - 1))
    alpha_max = float(alpha_max); alpha_min = float(alpha_min)
    return float(alpha_max * (alpha_min / alpha_max) ** (c / (n_classes - 1)))


def deadline_from_class(class_id: int, sampling_period_s: float, alpha_max: float, alpha_min: float) -> float:
    """Class-dependent freshness deadline D(c)=alpha(c)T_s."""
    return float(alpha_from_class(class_id, alpha_max, alpha_min) * float(sampling_period_s))


def tau_from_deadline(deadline_s: float, utility_at_deadline: float = 0.5) -> float:
    """Exponential freshness time constant derived from U(D)=q.

    If U(a)=exp(-a/tau) and U(D)=q, then tau=-D/log(q). Setting q=0.5 means
    that a result retains 50% of its value at the calibrated deadline.
    """
    q = float(utility_at_deadline)
    if not 0.0 < q < 1.0:
        raise ValueError("utility_at_deadline must be in (0, 1).")
    return float(-float(deadline_s) / math.log(q))


def freshness_utility(aoi_s: float, deadline_s: float, utility_at_deadline: float = 0.5) -> float:
    """Continuous value-of-information decay in (0, 1]."""
    tau = tau_from_deadline(deadline_s, utility_at_deadline)
    return float(math.exp(-float(aoi_s) / max(tau, 1e-12)))


def calibrate_alpha_min(edge_aoi_samples, sampling_period_s: float, quantile: float = 0.95,
                        margin: float = 1.2, min_clip: float = 0.01, max_clip: float = 1.0) -> float:
    """Calibrate the highest-urgency deadline from edge-only AoI.

    D_min = margin * Q_quantile(A_edge), alpha_min = D_min / T_s.
    """
    x = np.asarray(edge_aoi_samples, dtype=float)
    if x.size == 0:
        raise ValueError("edge_aoi_samples is empty.")
    d_min = float(margin * np.quantile(x, quantile))
    return float(np.clip(d_min / float(sampling_period_s), min_clip, max_clip))
