"""Train the proposed Dueling Double DQN mode-switching agent."""
from __future__ import annotations
import argparse, csv
from collections import deque
import numpy as np
import torch
from tqdm import trange
from .common import ensure_dir, project_path, set_seed
from .dqn_agent import DQNAgent, DQNConfig
from .env_iot_edge_cloud import IoTEdgeCloudEnv
from .replay_buffer import PrioritizedReplayBuffer

def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linear exploration schedule."""
    return start + min(1.0, step/max(1,decay_steps))*(end-start)

def train_one_seed(seed: int, steps: int, risk_mask: bool, device: str):
    set_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    env = IoTEdgeCloudEnv(
    split='train',
    network_profile='nominal',
    deadline_profile='calibrated',
    seed=seed,
    enforce_cloud_budget=True,
    )
    state, _ = env.reset(); cfg = DQNConfig(); agent = DQNAgent(env.observation_dim, env.action_dim, cfg, device)
    replay = PrioritizedReplayBuffer(100000, env.observation_dim, seed=seed)
    model_dir = ensure_dir(project_path('data/results/models')); log_dir = ensure_dir(project_path('data/results/training_logs'))
    log_path = log_dir / f'dqn_seed_{seed}.csv'
    recent_reward, recent_stale, recent_fresh, recent_bits, recent_cloud, recent_loss = [deque(maxlen=1000) for _ in range(6)]
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['step','epsilon','mean_reward','mean_loss','fresh_accuracy','deadline_miss_rate','mean_bits','cloud_usage_rate'])
        writer.writeheader()
        for step in trange(steps, desc=f'training seed={seed}'):
            eps = linear_epsilon(step, 1.0, 0.05, int(0.35*steps))
            legal = env.feasible_actions_p90() if risk_mask else None
            action = agent.act(state, epsilon=eps, legal_actions=legal)
            next_state, reward, terminated, truncated, info = env.step(action); done = terminated or truncated
            replay.add(state, action, reward, next_state, done); state = next_state
            if done: state, _ = env.reset()
            recent_reward.append(reward); recent_stale.append(info['stale']); recent_fresh.append(info['fresh_correct']); recent_bits.append(info['bits']); recent_cloud.append(info['cloud_used'])
            if replay.size >= cfg.batch_size and step % 64 == 0:
                # Train every 64 environment steps for speed. Increase frequency for final runs if desired.
                batch = replay.sample(cfg.batch_size); loss, td = agent.update(batch); replay.update_priorities(batch[5], td); recent_loss.append(loss)
            if (step+1) % 1000 == 0:
                writer.writerow({'step':step+1, 'epsilon':eps, 'mean_reward':np.mean(recent_reward), 'mean_loss':np.mean(recent_loss) if recent_loss else 0,
                                 'fresh_accuracy':np.mean(recent_fresh), 'deadline_miss_rate':np.mean(recent_stale), 'mean_bits':np.mean(recent_bits), 'cloud_usage_rate':np.mean(recent_cloud)})
                f.flush()
    model_path = model_dir / f'dqn_seed_{seed}.pt'; agent.save(str(model_path)); print(f'Saved {model_path} and {log_path}')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', nargs='+', type=int, default=[1,2,3,4,5]); ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--risk-mask', action='store_true'); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    for seed in args.seeds: train_one_seed(seed, args.steps, args.risk_mask, args.device)

if __name__ == '__main__': main()
