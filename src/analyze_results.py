"""Generate result tables and figures from evaluation logs."""
from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import ensure_dir, project_path
from .env_iot_edge_cloud import ACTION_NAMES

POLICY_ORDER = ["EdgeOnly", "CloudOnly", "EarlyExitOffloading", "DRL-Proposed"]

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["policy", "channel_profile", "deadline_profile"]).agg(
        acceptable_rate=("correct", "mean"),
        response_quality=("response_quality", "mean"),
        fresh_accuracy=("fresh_correct", "mean"),
        freshness_weighted_accuracy=("fresh_value", "mean"),
        deadline_miss_rate=("stale", "mean"),
        mean_aop_s=("aop_s", "mean"),
        p90_aop_s=("aop_s", lambda x: x.quantile(0.90)),
        mean_aoi_s=("aoi_s", "mean"),
        p90_aoi_s=("aoi_s", lambda x: x.quantile(0.90)),
        mean_bits=("bits", "mean"),
        mean_uplink_bits=("uplink_bits", "mean"),
        mean_downlink_bits=("downlink_bits", "mean"),
        mean_energy_j=("energy_j", "mean"),
        mean_cost_usd=("cost_usd", "mean"),
        cloud_usage_rate=("cloud_used", "mean"),
        iot_transmission_rate=("iot_transmitted", "mean"),
        escalation_accuracy=("escalation_correct", "mean"),
        risk_factor_f1=("risk_factor_f1", "mean"),
        action_accuracy=("action_correct", "mean"),
        urgency_accuracy=("urgency_correct", "mean"),
        verification_accuracy=("verification_correct", "mean"),
        n=("correct", "size"),
    ).reset_index()

def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, filename, deadline_profile="calibrated"):
    sub = summary[summary.deadline_profile == deadline_profile]
    nets = ["nominal", "delay_stress", "severe_delay_stress"]
    policies = [p for p in POLICY_ORDER if p in set(sub.policy)]
    x = np.arange(len(nets)); width = 0.80 / max(1, len(policies))
    plt.figure(figsize=(9, 4.5))
    for i, p in enumerate(policies):
        vals = [float(sub[(sub.policy == p) & (sub.channel_profile == n)][metric].iloc[0])
                if len(sub[(sub.policy == p) & (sub.channel_profile == n)]) else np.nan for n in nets]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=p)
    plt.xticks(x, nets, rotation=20); plt.ylabel(ylabel); plt.xlabel("Trace perturbation")
    plt.title(f"{ylabel} ({deadline_profile})")
    plt.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.savefig(filename); plt.close()

def plot_action_heatmap(df: pd.DataFrame, filename):
    sub = df[(df.policy == "DRL-Proposed") & (df.deadline_profile == "calibrated")]
    if sub.empty:
        return
    tab = pd.crosstab(sub.channel_state, sub["mode"], normalize="index")
    cols = ["iot_no_transmit"] + [ACTION_NAMES[i] for i in range(2)]
    for c in cols:
        if c not in tab.columns:
            tab[c] = 0.0
    tab = tab[cols]
    plt.figure(figsize=(7.5, 4)); plt.imshow(tab.values, aspect="auto")
    plt.colorbar(label="Action fraction")
    plt.yticks(np.arange(len(tab.index)), tab.index)
    plt.xticks(np.arange(len(cols)), cols, rotation=25, ha="right")
    plt.title("DRL effective action distribution by empirical latency burden")
    plt.tight_layout(); plt.savefig(filename); plt.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="data/results/evaluation_logs/all_results.csv")
    args = ap.parse_args()
    df = pd.read_csv(project_path(args.results))
    table_dir = ensure_dir(project_path("data/results/tables")); fig_dir = ensure_dir(project_path("data/results/figures"))
    summary = summarize(df); summary.to_csv(table_dir / "main_results.csv", index=False)
    ad = df.groupby(["policy", "channel_profile", "deadline_profile", "mode"]).size().reset_index(name="count")
    ad["fraction"] = ad["count"] / ad.groupby(["policy", "channel_profile", "deadline_profile"])["count"].transform("sum")
    ad.to_csv(table_dir / "action_distribution.csv", index=False)
    plot_metric(summary, "deadline_miss_rate", "Deadline miss rate", fig_dir / "fig_deadline_miss.pdf")
    plot_metric(summary, "response_quality", "Mean response quality", fig_dir / "fig_response_quality.pdf")
    plot_metric(summary, "freshness_weighted_accuracy", "Freshness-weighted response quality", fig_dir / "fig_freshness_weighted_accuracy.pdf")
    plot_metric(summary, "mean_bits", "Mean bits per request", fig_dir / "fig_comm_load.pdf")
    plot_metric(summary, "mean_energy_j", "Mean energy per request (J)", fig_dir / "fig_energy.pdf")
    plot_metric(summary, "iot_transmission_rate", "IoT transmission rate", fig_dir / "fig_iot_transmission_rate.pdf")
    plot_action_heatmap(df, fig_dir / "fig_action_heatmap.pdf")
    head = summary[(summary.deadline_profile == "calibrated") & (summary.channel_profile.isin(["nominal", "delay_stress", "severe_delay_stress"]))]
    print("\nHeadline metrics:")
    print(head[["policy", "channel_profile", "response_quality", "freshness_weighted_accuracy", "deadline_miss_rate", "mean_aop_s", "mean_bits", "cloud_usage_rate", "iot_transmission_rate"]].sort_values(["channel_profile", "policy"]).to_string(index=False))
    print(f"Wrote tables to {table_dir} and figures to {fig_dir}")

if __name__ == "__main__":
    main()
