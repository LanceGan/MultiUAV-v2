"""Single-agent TD3 for the single-segment trajectory (new design).

Reuses the standard TD3 training logic from the single-UAV project
(``SingleUAV/TD3.py``): twin-Q minimum, target-policy smoothing, delayed actor /
target updates.  The action is mapped to ``[min_action, max_action]`` via
scale/bias (so ``v in [0, v_max]`` and ``theta in [0, 2*pi]``), and the target
noise is scaled per-dimension (the angle is smoothed in radians, not by its full
``2*pi`` span).  No TensorBoard / dead code.
"""

import copy

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
    """state -> [v, theta] in [min_action, max_action]."""

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
    """Twin Q critic (Q1, Q2) on concatenated state + action."""

    def __init__(self, state_dim, action_dim, net_width):
        super().__init__()
        in_dim = state_dim + action_dim
        self.l1 = nn.Linear(in_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, 1)
        self.l5 = nn.Linear(in_dim, net_width)
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


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, min_action,
                 net_width=128, a_lr=1e-4, c_lr=1e-3, gamma=0.99,
                 batch_size=256, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2):
        self.max_action = _to_tensor(max_action)
        self.min_action = _to_tensor(min_action)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_delay = policy_delay

        self.actor = Actor(state_dim, action_dim, net_width,
                           self.max_action, self.min_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        # Per-dimension target-policy smoothing scale: v spans v_max, the angle
        # is smoothed in RADIANS (not 2*pi), matching the old trajectory code.
        smooth_scale = torch.tensor([float(self.max_action[0]), 0.5],
                                    dtype=torch.float32, device=device)
        self.policy_noise = policy_noise * smooth_scale
        self.noise_clip = noise_clip * smooth_scale

        self._update_counter = 0

    def select_action(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1)).to(device)
            return self.actor(s).cpu().numpy().flatten()

    def train(self, replay):
        s, a, r, s_prime, done = replay.sample(self.batch_size)

        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(a) * self.policy_noise,
                                -self.noise_clip, self.noise_clip)
            next_a = torch.clamp(self.actor_target(s_prime) + noise,
                                 self.min_action, self.max_action)
            target_q1, target_q2 = self.critic_target(s_prime, next_a)
            target_q = torch.min(target_q1, target_q2)
            target_q = r + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = (F.mse_loss(current_q1, target_q)
                       + F.mse_loss(current_q2, target_q))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        self._update_counter += 1
        actor_loss = None
        if self._update_counter % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            for p, tp in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(),
                             self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": (float(actor_loss.detach().cpu().item())
                           if actor_loss is not None else None),
        }

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
