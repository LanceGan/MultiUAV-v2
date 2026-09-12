"""Multi-Agent PPO (MAPPO) for the layer-3 trajectory task.

A centralized-critic / decentralized-actor (CTDE) on-policy algorithm used as an
*alternative* to the parameter-shared independent TD3 (``trajectory/td3.py``).

Architecture
------------
* **n_agents independent actors** (one per UAV). Each actor is a Gaussian policy
  that maps a single UAV's *local* observation (``local_state_dim``) to a
  ``mean`` + per-dim ``log_std``.  The mean is produced by a tanh head mapped to
  ``[min_action, max_action]`` (``action_scale`` / ``action_bias``, same mapping
  as TD3), the ``log_std`` is a learnable per-dimension parameter.
* **1 centralized critic** (value function) that maps the *global* state (the
  concatenation of all UAVs' observations, ``n_agents * local_state_dim``) to a
  scalar ``V``.  It is trained on the *team* reward (sum of per-agent rewards),
  matching the "sum of all agent rewards" convention.

Training (on-policy)
--------------------
1. **Collect** ~``rollout_steps`` transitions across episodes, storing
   ``(global_state, obs_list, actions, log_probs, team_reward, value, done)``.
2. **GAE** computes advantages and returns from the centralized value function
   (``done = info["terminal"]``; truncation bootstraps with ``done = 0``).
3. **Multi-epoch update**: for each minibatch, the critic is fit with MSE to the
   returns, and each actor is updated with the clipped PPO ratio loss + an
   entropy bonus.  A single (shared) advantage stream is used for every agent
   (cooperative team with a shared centralized value).

The algorithm class exposes the unified interface used by the other layer-3
algorithms (``select_actions`` / ``train`` / ``save`` / ``load``); the
``train_mappo`` function is the episode-loop that ties it to
``MultiUAVInspectionEnv``.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -3.0
LOG_STD_MAX = 1.0


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


class MAPPOActor(nn.Module):
    """Gaussian actor for one UAV: tanh mean head + learnable per-dim log_std."""

    def __init__(self, state_dim, action_dim, net_width, max_action, min_action,
                 log_std_init):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.register_buffer("action_scale", (max_action - min_action) / 2.0)
        self.register_buffer("action_bias", (max_action + min_action) / 2.0)

        log_std_init = torch.as_tensor(log_std_init, dtype=torch.float32)
        if log_std_init.dim() == 0:
            log_std_init = log_std_init.repeat(action_dim)
        self.log_std = nn.Parameter(log_std_init)

    def forward(self, state):
        h = torch.tanh(self.l1(state))
        h = torch.tanh(self.l2(h))
        mean = torch.tanh(self.l3(h))
        mean = mean * self.action_scale + self.action_bias
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def distribution(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)


class MAPPOCritic(nn.Module):
    """Centralized value function: global state -> scalar V."""

    def __init__(self, global_state_dim, net_width):
        super().__init__()
        self.l1 = nn.Linear(global_state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

    def forward(self, global_state):
        h = torch.tanh(self.l1(global_state))
        h = torch.tanh(self.l2(h))
        return self.l3(h)


class _RolloutBuffer(object):
    """Fixed-capacity on-policy buffer of collected transitions."""

    def __init__(self, n_agents, local_state_dim, action_dim, capacity):
        self.n_agents = int(n_agents)
        self.local_state_dim = int(local_state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.global_dim = self.n_agents * self.local_state_dim
        self._alloc()

    def _alloc(self):
        c = self.capacity
        self.global_states = np.zeros((c, self.global_dim), np.float32)
        self.next_global_states = np.zeros((c, self.global_dim), np.float32)
        self.obs = np.zeros((c, self.n_agents, self.local_state_dim), np.float32)
        self.actions = np.zeros((c, self.n_agents, self.action_dim), np.float32)
        self.log_probs = np.zeros((c, self.n_agents), np.float32)
        self.rewards = np.zeros((c,), np.float32)
        self.values = np.zeros((c,), np.float32)
        self.dones = np.zeros((c,), np.float32)
        self.idx = 0

    def store(self, obs_list, actions, log_probs, reward, value, done, next_obs_list):
        i = self.idx
        self.global_states[i] = np.concatenate(
            [np.asarray(o, np.float32).reshape(-1) for o in obs_list])
        self.next_global_states[i] = np.concatenate(
            [np.asarray(o, np.float32).reshape(-1) for o in next_obs_list])
        for j in range(self.n_agents):
            self.obs[i, j] = np.asarray(obs_list[j], np.float32)
            self.actions[i, j] = np.asarray(actions[j], np.float32)
            self.log_probs[i, j] = float(log_probs[j])
        self.rewards[i] = float(reward)
        self.values[i] = float(value)
        self.dones[i] = 1.0 if done else 0.0
        self.idx += 1

    def is_full(self):
        return self.idx >= self.capacity

    def __len__(self):
        return self.idx


class MAPPO(object):
    """Multi-Agent PPO: independent Gaussian actors + one centralized critic."""

    def __init__(self, n_agents, local_state_dim, action_dim, max_action,
                 min_action=None, net_width=256, critic_width=512, a_lr=1e-4,
                 c_lr=1e-3, gamma=0.99, batch_size=256, clip_ratio=0.2,
                 epochs=10, gae_lambda=0.95, entropy_coef=0.01,
                 max_grad_norm=0.5, rollout_steps=2048, std_init_frac=0.15):
        self.n_agents = int(n_agents)
        self.local_state_dim = int(local_state_dim)
        self.action_dim = int(action_dim)
        self.global_state_dim = self.n_agents * self.local_state_dim
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.clip_ratio = float(clip_ratio)
        self.epochs = int(epochs)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.rollout_steps = int(rollout_steps)

        max_action = _to_tensor(max_action)
        min_action = (torch.zeros_like(max_action) if min_action is None
                      else _to_tensor(min_action))
        self.max_action = max_action
        self.min_action = min_action

        # Per-dim initial std: a fraction of each action dimension's span
        # (v span = v_max, theta span = 2*pi), so both dims start well-scaled.
        action_range = max_action - min_action
        std_init = torch.clamp(action_range * std_init_frac, min=1e-3, max=2.0)
        log_std_init = torch.log(std_init)

        self.actors = nn.ModuleList([
            MAPPOActor(local_state_dim, action_dim, net_width,
                       max_action, min_action, log_std_init).to(device)
            for _ in range(self.n_agents)
        ])
        self.critic = MAPPOCritic(self.global_state_dim, critic_width).to(device)

        self.actor_optimizers = [
            torch.optim.Adam(a.parameters(), lr=a_lr) for a in self.actors]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=c_lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _act_one(self, obs, i, add_noise, noise_scale):
        s = _to_tensor(obs).reshape(1, -1)
        mean, std = self.actors[i](s)
        if add_noise:
            scale = _to_tensor(noise_scale)
            if scale.dim() == 0 or scale.numel() == self.action_dim:
                std = std * scale
            dist = Normal(mean, std)
            a = dist.sample()
        else:
            a = mean
        a = torch.clamp(a, self.min_action, self.max_action)
        return a.flatten().detach().cpu().numpy().astype(np.float32)

    def select_actions(self, obs_list, add_noise=False, noise_scale=0.1):
        """Return one action per agent.

        ``add_noise=False`` -> deterministic mean (eval).  ``add_noise=True`` ->
        sample from the Gaussian policy; ``noise_scale`` (scalar or per-dim
        array) multiplicatively scales the policy std (use 1.0 for the policy's
        own exploration level).
        """
        return [self._act_one(obs_list[i], i, add_noise, noise_scale)
                for i in range(self.n_agents)]

    @torch.no_grad()
    def act(self, obs_list, add_noise=True):
        """Collect-time helper: actions + log_probs + centralized value."""
        gs = np.concatenate(
            [np.asarray(o, np.float32).reshape(1, -1) for o in obs_list], axis=1)
        value = self.critic(_to_tensor(gs)).item()

        actions = []
        log_probs = []
        for i in range(self.n_agents):
            s = _to_tensor(obs_list[i]).reshape(1, -1)
            mean, std = self.actors[i](s)
            if add_noise:
                dist = Normal(mean, std)
                a = dist.sample()
                lp = float(dist.log_prob(a).sum(-1).item())
            else:
                a = mean
                lp = 0.0
            a = torch.clamp(a, self.min_action, self.max_action)
            actions.append(a.flatten().cpu().numpy().astype(np.float32))
            log_probs.append(lp)
        return actions, log_probs, value

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def train(self, buffer):
        """One PPO update round (multi-epoch) on a collected rollout buffer."""
        T = len(buffer)
        if T == 0:
            return {"critic_loss": None, "actor_loss": None, "entropy": None}

        rewards = buffer.rewards[:T]
        values = buffer.values[:T]
        dones = buffer.dones[:T]

        # Bootstrap value of the state AFTER the last collected transition.
        with torch.no_grad():
            last_global = _to_tensor(buffer.next_global_states[T - 1]).reshape(1, -1)
            last_value = self.critic(last_global).item()
        if dones[T - 1] > 0.5:
            last_value = 0.0  # the last step was terminal -> no bootstrap

        # GAE (team reward + centralized V -> single advantage stream).
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        next_value = last_value
        for t in range(T - 1, -1, -1):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values

        # Standardize advantages for stable actor updates.
        adv_mean = float(advantages.mean())
        adv_std = float(advantages.std()) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs = _to_tensor(buffer.obs[:T])             # (T, n, local_dim)
        actions = _to_tensor(buffer.actions[:T])     # (T, n, action_dim)
        old_log_probs = _to_tensor(buffer.log_probs[:T])  # (T, n)
        adv_t = _to_tensor(advantages)               # (T,)
        ret_t = _to_tensor(returns)                  # (T,)
        gs = _to_tensor(buffer.global_states[:T])    # (T, global_dim)

        stats = {"critic_loss": 0.0, "actor_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        for _ in range(self.epochs):
            perm = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                idx = perm[start:start + self.batch_size]

                # --- centralized critic: MSE(V, GAE returns) ---
                v_pred = self.critic(gs[idx])
                critic_loss = F.mse_loss(v_pred, ret_t[idx].unsqueeze(1))
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # --- independent actors: clipped PPO ratio + entropy ---
                actor_losses = []
                entropies = []
                for i in range(self.n_agents):
                    mean, std = self.actors[i](obs[idx, i])
                    dist = Normal(mean, std)
                    new_log_prob = dist.log_prob(actions[idx, i]).sum(-1)
                    log_ratio = new_log_prob - old_log_probs[idx, i]
                    # Clamp the log-ratio so exp() cannot overflow to inf.
                    ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
                    a = adv_t[idx]
                    surr1 = ratio * a
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * a
                    entropy = dist.entropy().sum(-1).mean()
                    actor_loss = (-torch.min(surr1, surr2).mean()
                                  - self.entropy_coef * entropy)
                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_optimizers[i].step()
                    actor_losses.append(float(actor_loss.item()))
                    entropies.append(float(entropy.item()))

                stats["critic_loss"] += float(critic_loss.item())
                stats["actor_loss"] += float(np.mean(actor_losses))
                stats["entropy"] += float(np.mean(entropies))
                n_updates += 1

        for k in stats:
            stats[k] = stats[k] / max(n_updates, 1)
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        for a, sd in zip(self.actors, ckpt["actors"]):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_mappo(cfg, uav_traverse, total_episode=2000, model_subdir="MAPPO",
                model_dir=None, log_dir=None, log_every=10, rollout_steps=2048):
    """Train MAPPO on the injected ``uav_traverse`` and return a result dict.

    Returns ``{"ep_rewards": [...], "best_reward": float, "model": <MAPPO>}``.
    The per-episode *team* reward (sum of all agent rewards) is appended to
    ``ep_rewards`` and serialised to ``results/logs/UAV_<N>_<Algo>_rewards.npz``.
    """
    from trajectory.env import MultiUAVInspectionEnv

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = model_dir or os.path.join(ROOT, "results", "models")
    log_dir = log_dir or os.path.join(ROOT, "results", "logs")
    uav_num = int(cfg["uav_num"])
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MultiUAVInspectionEnv(cfg, safe_distance=0.1)
    env.set_uav_traverse(uav_traverse)
    local_state_dim = env.local_obs_dim
    action_dim = env.action_dim
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)

    model = MAPPO(n_agents=uav_num, local_state_dim=local_state_dim,
                  action_dim=action_dim, max_action=max_action,
                  min_action=min_action, rollout_steps=rollout_steps)
    buffer = _RolloutBuffer(uav_num, local_state_dim, action_dim, rollout_steps)

    save_dir = os.path.join(model_dir, "UAV_%d" % uav_num, model_subdir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "UAV_%d_%s.log" % (uav_num, model_subdir))

    best_reward = -float("inf")
    ep_rewards = []

    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=== MAPPO training (centralized critic + independent actors) ===")
    log("uav_num=%d total_episode=%d seed=%d local_state_dim=%d action_dim=%d T=%d rollout_steps=%d"
        % (uav_num, total_episode, seed, local_state_dim, action_dim, env.T, rollout_steps))

    for ep in range(int(total_episode)):
        obs_list = env.reset()
        ep_reward = 0.0
        steps = 0
        while True:
            actions, log_probs, value = model.act(obs_list, add_noise=True)
            next_obs, rewards, dones, info = env.step(actions)

            team_reward = float(np.sum(rewards))
            done = bool(info["terminal"])
            buffer.store(obs_list, actions, log_probs, team_reward, value,
                         done, next_obs)

            if buffer.is_full():
                model.train(buffer)
                buffer = _RolloutBuffer(uav_num, local_state_dim, action_dim,
                                        rollout_steps)

            ep_reward += team_reward
            steps += 1
            obs_list = next_obs
            if info["terminal"] or info["truncated"]:
                break

        ep_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
            model.save(os.path.join(save_dir, "best.pth"))

        if (ep + 1) % log_every == 0 or ep == 0:
            log("ep %4d | reward %10.2f | avg100 %10.2f | steps %4d"
                % (ep + 1, ep_reward, float(np.mean(ep_rewards[-100:])), steps))

    # Flush the leftover partial buffer so all collected data is used.
    if len(buffer) > 0:
        model.train(buffer)

    model.save(os.path.join(save_dir, "last.pth"))
    np.savez(os.path.join(log_dir, "UAV_%d_%s_rewards.npz" % (uav_num, model_subdir)),
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64))

    arr = np.asarray(ep_rewards, dtype=np.float64)
    log("=== done === best_reward=%.2f avg_last100=%.2f nan_count=%d"
        % (best_reward, float(np.mean(arr[-100:])), int(np.isnan(arr).sum())))
    log_file.close()
    return {"ep_rewards": ep_rewards, "best_reward": best_reward, "model": model}
