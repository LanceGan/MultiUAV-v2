"""Soft Actor-Critic (SAC) for the single-segment trajectory (comparison).

Stochastic Gaussian policy with tanh squashing, twin Q critic (reused from
``segment/td3.py``), fixed entropy temperature ``alpha``.  ``select_action``
returns the mean (deterministic eval); ``sample_action`` samples from the policy
(training).  Same ``save``/``load`` interface.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from segment.td3 import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


class GaussianActor(nn.Module):
    """Gaussian policy with tanh squashing -> [min_action, max_action]."""

    def __init__(self, state_dim, action_dim, net_width, max_action, min_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mean_layer = nn.Linear(net_width, action_dim)
        self.log_std_layer = nn.Linear(net_width, action_dim)
        self.register_buffer("action_scale", (max_action - min_action) / 2.0)
        self.register_buffer("action_bias", (max_action + min_action) / 2.0)

    def forward(self, state):
        h = F.relu(self.l1(state))
        h = F.relu(self.l2(h))
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def mean_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, min_action,
                 net_width=128, a_lr=3e-4, c_lr=3e-4, gamma=0.99,
                 batch_size=256, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha

        self.actor = GaussianActor(state_dim, action_dim, net_width,
                                   _to_tensor(max_action),
                                   _to_tensor(min_action)).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

    def select_action(self, state):
        """Deterministic mean action (eval)."""
        with torch.no_grad():
            s = torch.FloatTensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1)).to(device)
            return self.actor.mean_action(s).cpu().numpy().flatten()

    def sample_action(self, state):
        """Sampled action from the Gaussian policy (training)."""
        with torch.no_grad():
            s = torch.FloatTensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1)).to(device)
            a, _ = self.actor.sample(s)
            return a.cpu().numpy().flatten()

    def train(self, replay):
        s, a, r, s_prime, done = replay.sample(self.batch_size)

        with torch.no_grad():
            next_a, next_log_prob = self.actor.sample(s_prime)
            q1_t, q2_t = self.critic_target(s_prime, next_a)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            target_q = r + (1.0 - done) * self.gamma * q_t

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        new_a, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
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
