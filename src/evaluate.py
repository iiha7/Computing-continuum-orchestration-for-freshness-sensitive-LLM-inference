"""Evaluate the baselines and the trained DRL policy."""
from __future__ import annotations
import argparse, csv
from .common import ensure_dir, project_path
from .dqn_agent import DQNAgent
from .env_iot_edge_cloud import IoTEdgeCloudEnv
from .policies import BASELINE_NAMES, POLICY_REGISTRY

NETWORK_PROFILES = ["nominal", "delay_stress", "severe_delay_stress"]
DEADLINE_PROFILES = ["urgency_0.5x", "calibrated", "urgency_1.5x", "urgency_2.0x"]

def evaluate_policy(policy_name: str, seed: int, episodes: int, steps_per_episode: int, network_profile: str, deadline_profile: str, device="cpu"):
    enforce_budget = policy_name == "DRL-Proposed"

    env = IoTEdgeCloudEnv(
        split="test",
        network_profile=network_profile,
        deadline_profile=deadline_profile,
        seed=seed,
        max_steps=steps_per_episode,
        enforce_cloud_budget=enforce_budget,
    )
    if policy_name == "DRL-Proposed":
        model_path = project_path("data/results/models", f"dqn_seed_{seed}.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}; train first.")
        agent = DQNAgent.load(str(model_path), device=device); policy = None
    else:
        cls = POLICY_REGISTRY[policy_name]
        policy = cls(seed=seed) if policy_name == "Random" else cls(); agent = None
    rows = []
    for ep in range(episodes):
        state, _ = env.reset()
        for t in range(steps_per_episode):
            action = agent.act(state, epsilon=0.0, legal_actions=env.feasible_actions_p90()) if policy_name == "DRL-Proposed" else policy.act(env)
            ns, reward, terminated, truncated, info = env.step(action)
            info.update({"policy": policy_name, "seed": seed, "episode": ep, "t": t, "deadline_profile": deadline_profile})
            rows.append(info); state = ns
            if terminated or truncated:
                break
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1,2,3,4,5])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--steps-per-episode", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    policies = BASELINE_NAMES + ["DRL-Proposed"] if "all" in args.policies else args.policies
    out_dir = ensure_dir(project_path("data/results/evaluation_logs"))
    out_path = out_dir / "all_results.csv"
    all_rows = []
    for pol in policies:
        for seed in args.seeds:
            for net in NETWORK_PROFILES:
                for dl in DEADLINE_PROFILES:
                    print(f"Evaluating policy={pol} seed={seed} network={net} deadline={dl}", flush=True)
                    all_rows.extend(evaluate_policy(pol, seed, args.episodes, args.steps_per_episode, net, dl, args.device))
    fields = sorted(all_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_path} ({len(all_rows)} rows)")

if __name__ == "__main__":
    main()
