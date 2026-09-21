"""Dueling Double DQN agent."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    """Dueling architecture: Q(s,a)=V(s)+A(s,a)-mean(A)."""
    def __init__(self, state_dim: int, action_dim: int, hidden=(256,256)):
        super().__init__(); layers=[]; last=state_dim
        for h in hidden:
            layers += [nn.Linear(last,h), nn.ReLU()]; last=h
        self.feature = nn.Sequential(*layers)
        self.value = nn.Sequential(nn.Linear(last, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], 1))
        self.advantage = nn.Sequential(nn.Linear(last, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], action_dim))
    def forward(self, x):
        z = self.feature(x); v = self.value(z); a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)

@dataclass
class DQNConfig:
    lr: float = 1e-4; gamma: float = 0.99; batch_size: int = 32; target_update_interval: int = 250; hidden: tuple = (32,32)

class DQNAgent:
    """Dueling Double DQN with optional legal-action mask."""
    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig, device='cpu'):
        self.state_dim=state_dim; self.action_dim=action_dim; self.cfg=cfg; self.device=torch.device(device)
        self.q = DuelingQNetwork(state_dim, action_dim, cfg.hidden).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim, cfg.hidden).to(self.device); self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr); self.train_steps=0
    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon=0.0, legal_actions: Optional[List[int]]=None) -> int:
        if legal_actions is None or len(legal_actions)==0: legal_actions = list(range(self.action_dim))
        if np.random.random() < epsilon: return int(np.random.choice(legal_actions))
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0); qv = self.q(s).cpu().numpy()[0]
        mask = np.full(self.action_dim, -1e9, dtype=np.float32); mask[legal_actions]=0.0
        return int(np.argmax(qv + mask))
    def update(self, batch):
        s,a,r,ns,d,idx,w = batch
        s=torch.tensor(s,dtype=torch.float32,device=self.device); ns=torch.tensor(ns,dtype=torch.float32,device=self.device)
        a=torch.tensor(a,dtype=torch.long,device=self.device).unsqueeze(1); r=torch.tensor(r,dtype=torch.float32,device=self.device).unsqueeze(1)
        d=torch.tensor(d,dtype=torch.float32,device=self.device).unsqueeze(1); w=torch.tensor(w,dtype=torch.float32,device=self.device).unsqueeze(1)
        qsa = self.q(s).gather(1,a)
        with torch.no_grad():
            na = self.q(ns).argmax(dim=1, keepdim=True); nq = self.target(ns).gather(1, na); target = r + self.cfg.gamma*(1.0-d)*nq
        td = target - qsa; loss = (w * F.smooth_l1_loss(qsa, target, reduction='none')).mean()
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step(); self.train_steps += 1
        if self.train_steps % self.cfg.target_update_interval == 0: self.target.load_state_dict(self.q.state_dict())
        return float(loss.item()), td.detach().cpu().numpy().squeeze()
    def save(self, path: str):
        torch.save({'state_dim': self.state_dim, 'action_dim': self.action_dim, 'cfg': self.cfg.__dict__, 'q_state_dict': self.q.state_dict()}, path)
    @staticmethod
    def load(path: str, device='cpu'):
        ckpt = torch.load(path, map_location=device); cfg = DQNConfig(**ckpt['cfg'])
        agent = DQNAgent(ckpt['state_dim'], ckpt['action_dim'], cfg, device); agent.q.load_state_dict(ckpt['q_state_dict']); agent.target.load_state_dict(agent.q.state_dict())
        return agent
