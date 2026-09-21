"""Prioritized replay buffer for DQN training."""
from __future__ import annotations
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, alpha=0.6, beta=0.4, seed=1):
        self.capacity = int(capacity); self.alpha = alpha; self.beta = beta; self.rng = np.random.default_rng(seed)
        self.pos = 0; self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64); self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32); self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)
    def add(self, s, a, r, ns, done):
        maxp = self.priorities[:self.size].max() if self.size else 1.0
        i = self.pos; self.states[i]=s; self.actions[i]=a; self.rewards[i]=r; self.next_states[i]=ns; self.dones[i]=float(done); self.priorities[i]=maxp
        self.pos = (self.pos+1)%self.capacity; self.size = min(self.size+1, self.capacity)
    def sample(self, batch_size: int):
        pr = self.priorities[:self.size] ** self.alpha; probs = pr/pr.sum()
        idx = self.rng.choice(self.size, size=batch_size, replace=self.size<batch_size, p=probs)
        w = (self.size*probs[idx]) ** (-self.beta); w = w/w.max()
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx], idx, w.astype(np.float32)
    def update_priorities(self, idxs, td_errors, eps=1e-5):
        self.priorities[idxs] = np.abs(td_errors).astype(np.float32) + eps
