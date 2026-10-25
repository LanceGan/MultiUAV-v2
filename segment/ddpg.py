"""Single-agent DDPG for the single-segment trajectory (comparison baseline).

DDPG = TD3 without the twin-Q minimum, target-policy smoothing, and delayed
actor updates.  Reuses the deterministic ``Actor`` from ``segment/td3.py`` and a
single Q critic.  Same ``select_actions``/``train``/``save``/``load`` interface.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment.td3 import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


class Critic(nn.Module):
    """Single Q critic on concatenated state + action."""

    def __init__(self, state_dim, action_dim, net_width):
        super().__init__()
        in_dim = state_dim + action_dim
        self.l1 = nn.Linear(in_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        q = self.l4(q)
        return q


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, min_action,
                 net_width=128, a_lr=1e-4, c_lr=1e-3, gamma=0.99,
                 batch_size=256, tau=0.005):
        self.max_action = _to_tensor(max_action)
        self.min_action = _to_tensor(min_action)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.actor = Actor(state_dim, action_dim, net_width,
                           self.max_action, self.min_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

    def select_action(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1)).to(device)
            return self.actor(s).cpu().numpy().flatten()

    def train(self, replay):
        s, a, r, s_prime, done = replay.sample(self.batch_size)

        with torch.no_grad():
            next_a = self.actor_target(s_prime)
            target_q = self.critic_target(s_prime, next_a)
            target_q = r + (1.0 - done) * self.gamma * target_q

        cur_q = self.critic(s, a)
        critic_loss = F.mse_loss(cur_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"critic_loss": float(critic_loss.detach().cpu().item()),
                "actor_loss": float(actor_loss.detach().cpu().item())}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
