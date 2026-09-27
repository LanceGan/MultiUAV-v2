"""Parameter-shared independent TD3 (Task 7).

A single shared actor plus a single twin critic (one ``SharedTD3`` instance) is
used by *all* UAVs. Every UAV's local observation is fed through the same actor
to produce its action, and the shared networks train on the pooled replay buffer.

TD3 triple: target policy smoothing (``policy_noise`` / ``noise_clip`` scaled by
the action span ``max_action - min_action``), twin-Q minimum for the target, and
a delayed actor / target update (every ``policy_update_delay`` critic updates).

The actor maps tanh outputs to ``[min_action, max_action]`` via
``action_scale = (max - min)/2`` and ``action_bias = (max + min)/2``, so
``v in [0, v_max]`` and ``theta in [0, 2*pi]``.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action, min_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, action_dim)

        self.register_buffer("action_scale", (max_action - min_action) / 2.0)
        self.register_buffer("action_bias", (max_action + min_action) / 2.0)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = torch.tanh(self.l4(a))
        return a * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Twin critic (Q1, Q2) on concatenated state + action."""

    def __init__(self, state_dim, action_dim, net_width):
        super().__init__()
        input_dim = state_dim + action_dim
        # Q1
        self.l1 = nn.Linear(input_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, 1)
        # Q2
        self.l5 = nn.Linear(input_dim, net_width)
        self.l6 = nn.Linear(net_width, net_width)
        self.l7 = nn.Linear(net_width, net_width)
        self.l8 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1


class SharedTD3(object):
    def __init__(self, state_dim, action_dim, max_action, min_action=None,
                 gamma=0.99, net_width=128, a_lr=1e-4, c_lr=1e-3,
                 batch_size=256, tau=0.005, policy_noise_scale=0.2,
                 noise_clip_scale=0.5, policy_update_delay=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_update_delay = policy_update_delay

        max_action = _to_tensor(max_action)
        min_action = (torch.zeros_like(max_action) if min_action is None
                      else _to_tensor(min_action))
        self.max_action = max_action
        self.min_action = min_action

        # Per-dimension target-policy smoothing scale. The angle is smoothed in
        # RADIANS (0.5 rad ~ 29 deg), not by its full 2*pi action span, which
        # otherwise injects an enormous angular perturbation into the target Q.
        smooth_scale = torch.tensor([float(max_action[0]), 0.5],
                                    dtype=torch.float32, device=device)
        self.policy_noise = policy_noise_scale * smooth_scale
        self.noise_clip = noise_clip_scale * smooth_scale

        self.actor = Actor(state_dim, action_dim, net_width,
                           max_action, min_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self._update_counter = 0

    def select_action(self, state):
        """Deterministic action for one UAV's local observation."""
        with torch.no_grad():
            s = torch.FloatTensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1)).to(device)
            a = self.actor(s)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        
        s, a, r, s_prime, done = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(a) * self.policy_noise,
                                -self.noise_clip, self.noise_clip)
            next_action = self.actor_target(s_prime) + noise
            next_action = torch.max(torch.min(next_action, self.max_action),
                                    self.min_action)

            target_q1, target_q2 = self.critic_target(s_prime, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = r + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = (F.mse_loss(current_q1, target_q)
                       + F.mse_loss(current_q2, target_q))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self._update_counter += 1
        actor_loss = None
        if self._update_counter % self.policy_update_delay == 0:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": (float(actor_loss.detach().cpu().item())
                           if actor_loss is not None else None),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
