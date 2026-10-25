"""Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for layer 3.

An algorithm-contrast baseline against the existing parameter-shared TD3.  It
keeps the *same* per-agent local observation -> ``[v, theta]`` action interface
as ``trajectory/td3.py`` but uses the classic MADDPG design:

* **n_agents independent actors** — one per UAV, each mapping its own local
  observation (``local_state_dim``) to its own ``[v, theta]`` action.
* **1 centralised critic** — inputs the *global* state (all local observations
  concatenated -> ``n_agents * local_state_dim``) plus the *global* action (all
  actions concatenated -> ``n_agents * action_dim``) and outputs a single Q.

DDPG (not TD3) training — the requested contrast vs the TD3 baseline:

* single Q critic (no twin-Q ``min``),
* no target-policy smoothing,
* no delayed actor update (every critic update also updates every actor).

Training (per ``train`` call, one batch):

1. Critic Bellman update: ``y = r + (1-done) * gamma * Q'(s', a'_target)``
   where ``a'_target`` concatenates each agent's *target* actor output on its
   own local observation; minimise ``MSE(Q(s, a), y)`` (single Q TD error).
2. Each agent ``i``'s actor maximises ``Q(s, a_mix)`` where ``a_mix`` uses the
   *current* actor output for agent ``i`` and the *target* actor outputs for
   every other agent (so the gradient only flows into actor ``i``).
3. Soft-update all targets (critic + every actor) with ``tau`` every step.

``done`` in the replay buffer is ``info["terminal"]`` (team completion only), so
a horizon truncation bootstraps as ``done = 0`` — identical convention to TD3.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trajectory.env import MultiUAVInspectionEnv
from trajectory.replay import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


class Actor(nn.Module):
    """Per-agent actor: local observation -> ``[v, theta]`` in ``[min, max]``.

    Maps tanh outputs to ``[min_action, max_action]`` via
    ``action_scale = (max - min)/2`` and ``action_bias = (max + min)/2``, so
    ``v in [0, v_max]`` and ``theta in [0, 2*pi]`` (same as the TD3 actor).
    """

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
    """Centralised single-Q critic on concatenated global state + global action."""

    def __init__(self, state_dim, action_dim, net_width):
        super().__init__()
        input_dim = state_dim + action_dim
        self.l1 = nn.Linear(input_dim, net_width)
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


class MADDPG(object):
    """MADDPG: independent per-agent actors + one centralised critic.

    Args match the unified algorithm interface (see project task spec).
    """

    def __init__(self, n_agents, local_state_dim, action_dim, max_action,
                 min_action=None, net_width=256, critic_width=512,
                 a_lr=1e-4, c_lr=1e-3, gamma=0.99, batch_size=256, tau=0.005):
        self.n_agents = int(n_agents)
        self.local_state_dim = int(local_state_dim)
        self.action_dim = int(action_dim)
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.tau = tau

        max_action_t = _to_tensor(max_action)
        min_action_t = (torch.zeros_like(max_action_t) if min_action is None
                        else _to_tensor(min_action))
        self.max_action = max_action_t
        self.min_action = min_action_t
        # numpy copies for exploration-noise clipping in ``select_actions``.
        self.max_action_np = np.asarray(max_action, dtype=np.float32)
        self.min_action_np = (np.asarray(min_action, dtype=np.float32)
                              if min_action is not None
                              else np.zeros_like(self.max_action_np))

        self.global_state_dim = self.n_agents * self.local_state_dim
        self.global_action_dim = self.n_agents * self.action_dim

        # n_agents independent actors (+ targets).
        self.actors = nn.ModuleList(
            [Actor(local_state_dim, action_dim, net_width,
                   self.max_action, self.min_action).to(device)
             for _ in range(self.n_agents)])
        self.actor_targets = nn.ModuleList(
            [copy.deepcopy(a) for a in self.actors])
        self.actor_optimizers = [
            torch.optim.Adam(a.parameters(), lr=a_lr) for a in self.actors]

        # 1 centralised critic (+ target).
        self.critic = Critic(self.global_state_dim, self.global_action_dim,
                             critic_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=c_lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_actions(self, obs_list, add_noise=False, noise_scale=0.1):
        """Return one action per agent.

        ``noise_scale`` may be a scalar (scaled by the per-dim action span) or a
        per-dim array ``[v_scale, theta_scale]``; Gaussian noise
        ``N(0, 1) * noise_scale`` is added when ``add_noise`` is set, then the
        action is clipped back to ``[min_action, max_action]``.
        """
        actions = []
        for i in range(self.n_agents):
            with torch.no_grad():
                s = torch.FloatTensor(
                    np.asarray(obs_list[i], dtype=np.float32).reshape(1, -1)
                ).to(device)
                a = self.actors[i](s).cpu().numpy().flatten()
            if add_noise:
                ns = np.asarray(noise_scale, dtype=np.float32)
                if ns.ndim == 0:
                    ns = float(ns) * (self.max_action_np - self.min_action_np)
                a = a + np.random.normal(0.0, 1.0, size=self.action_dim) * ns
                a = np.clip(a, self.min_action_np, self.max_action_np)
            actions.append(a.astype(np.float32))
        return actions

    def select_action(self, obs):
        """Deterministic action for one UAV (compat with the TD3-style calls)."""
        return self.select_actions([obs], add_noise=False)[0]

    # ------------------------------------------------------------------
    # Internal action reconstruction from a global state batch
    # ------------------------------------------------------------------
    def _slice_state(self, s_global, i):
        """Slice agent ``i``'s local observation out of a global state batch."""
        return s_global[:, i * self.local_state_dim:
                        (i + 1) * self.local_state_dim]

    def _target_actions(self, s_global):
        """Concatenate every agent's *target* actor output on the batch."""
        acts = [self.actor_targets[i](self._slice_state(s_global, i))
                for i in range(self.n_agents)]
        return torch.cat(acts, dim=1)

    def _actor_mix_action(self, s_global, i):
        """Global action for actor ``i``'s gradient: current actor for ``i``,
        detached *target* actors for every other agent."""
        acts = []
        for j in range(self.n_agents):
            s_j = self._slice_state(s_global, j)
            if j == i:
                acts.append(self.actors[j](s_j))
            else:
                acts.append(self.actor_targets[j](s_j).detach())
        return torch.cat(acts, dim=1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, replay_buffer):
        """One DDPG step: critic Bellman update + every actor's policy update.

        ``replay_buffer`` stores *global* transitions ``(S, A, r, S', done)``
        with ``S`` the concatenated local observations and ``A`` the concatenated
        actions (``r`` is the team reward, i.e. the sum of per-agent rewards).
        """
        s, a, r, s_prime, done = replay_buffer.sample(self.batch_size)

        # 1) Critic: single-Q Bellman target (no twin-Q, no smoothing).
        with torch.no_grad():
            next_actions = self._target_actions(s_prime)
            target_q = self.critic_target(s_prime, next_actions)
            target_q = r + (1.0 - done) * self.gamma * target_q

        current_q = self.critic(s, a)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # 2) Each actor maximises the centralised Q (no delayed update).
        actor_losses = []
        for i in range(self.n_agents):
            full_action = self._actor_mix_action(s, i)
            actor_loss = -self.critic(s, full_action).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()
            actor_losses.append(float(actor_loss.detach().cpu().item()))

        # 3) Soft-update every target network.
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for i in range(self.n_agents):
            for param, target_param in zip(self.actors[i].parameters(),
                                           self.actor_targets[i].parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(np.mean(actor_losses)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "actor_targets": [a.state_dict() for a in self.actor_targets],
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        for a, sd in zip(self.actors, ckpt["actors"]):
            a.load_state_dict(sd)
        for a, sd in zip(self.actor_targets, ckpt["actor_targets"]):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
try:
    from experiment import _common
    _MODEL_DIR = _common.MODEL_DIR
    _LOG_DIR = _common.LOG_DIR
except Exception:  # pragma: no cover - fall back to repo-relative defaults
    _MODEL_DIR = os.path.join("results", "models")
    _LOG_DIR = os.path.join("results", "logs")


def _deterministic_reward(env, model, uav_traverse, n_agents):
    """Deterministic rollout reward (no exploration noise) for best-model pick."""
    env.set_uav_traverse(uav_traverse)
    obs = env.reset()
    total = 0.0
    while True:
        actions = model.select_actions(obs, add_noise=False)
        obs, rewards, dones, info = env.step(np.asarray(actions, dtype=float))
        total += float(np.sum(rewards))
        if info["terminal"] or info["truncated"]:
            break
    # Only a COMPLETING rollout is a valid best-checkpoint candidate.
    if not info["terminal"]:
        return -float("inf")
    return total


def train_maddpg(cfg, uav_traverse, total_episode=2000, model_subdir="MADDPG",
                 model_dir=None, log_dir=None, log_every=10, use_a2g=True,
                 use_comm=True):
    """Train MADDPG on the injected traverse (unified algorithm interface).

    Returns ``{"ep_rewards": [...], "best_reward": float, "model": model}`` and
    writes ``results/logs/UAV_<N>_<Algo>_rewards.npz`` with the per-episode team
    reward (sum of all agent rewards).
    """
    model_dir = model_dir or _MODEL_DIR
    log_dir = log_dir or _LOG_DIR
    uav_num = int(cfg["uav_num"])
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MultiUAVInspectionEnv(cfg, safe_distance=0.1, use_a2g=use_a2g,
                                use_comm=use_comm)
    env.set_uav_traverse(uav_traverse)
    local_state_dim = env.local_obs_dim
    action_dim = env.action_dim
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    # Per-dimension exploration noise scale: v in [0, v_max], theta in radians.
    noise_scale = np.array([float(cfg["v_max"]), 1.0], dtype=np.float32)

    model = MADDPG(uav_num, local_state_dim, action_dim, max_action, min_action)
    replay = ReplayBuffer(model.global_state_dim, model.global_action_dim,
                          max_size=int(2e5))
    batch_size = model.batch_size

    save_dir = os.path.join(model_dir, "UAV_%d" % uav_num, model_subdir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "UAV_%d_%s.log" % (uav_num, model_subdir))

    expl_start, expl_end = 0.3, 0.02
    expl_decay = (expl_end / expl_start) ** (1.0 / max(int(total_episode), 1))
    best_reward = -float("inf")
    best_det_reward = -float("inf")
    eval_interval = 50
    ep_rewards = []

    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=== MADDPG training (injected traverse) ===")
    log("uav_num=%d total_episode=%d seed=%d local_state_dim=%d action_dim=%d "
        "global_state_dim=%d T=%d"
        % (uav_num, total_episode, seed, local_state_dim, action_dim,
           model.global_state_dim, env.T))

    for ep in range(int(total_episode)):
        obs = env.reset()
        ep_reward = 0.0
        steps = 0
        expl_scale = expl_start * (expl_decay ** ep)
        while True:
            actions = model.select_actions(obs, add_noise=True,
                                           noise_scale=expl_scale * noise_scale)

            next_obs, rewards, dones, info = env.step(actions)
            # Truncation (timeout) is a terminal FAILURE here -> done=True so the
            # critic does not bootstrap from the "dead" cutoff state.
            done = bool(info["terminal"]) or bool(info["truncated"])
            episode_done = done

            global_state = np.concatenate(obs)
            global_action = np.concatenate(actions)
            global_next_state = np.concatenate(next_obs)
            team_reward = float(np.sum(rewards))
            replay.add(global_state, global_action, team_reward,
                       global_next_state, done)

            if len(replay) >= batch_size:
                model.train(replay)

            ep_reward += team_reward
            steps += 1
            obs = next_obs
            if episode_done:
                break

        ep_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
        # Best checkpoint by a deterministic rollout (no exploration noise).
        if (ep + 1) % eval_interval == 0 or ep == 0:
            det_reward = _deterministic_reward(env, model, uav_traverse, uav_num)
            if det_reward > best_det_reward:
                best_det_reward = det_reward
                model.save(os.path.join(save_dir, "best.pth"))
                log("  [eval] ep %4d | det_reward %10.2f (new best)"
                    % (ep + 1, det_reward))

        if (ep + 1) % log_every == 0 or ep == 0:
            log("ep %4d | reward %10.2f | avg100 %10.2f | steps %3d | expl %.3f"
                % (ep + 1, ep_reward, float(np.mean(ep_rewards[-100:])),
                   steps, expl_scale))

    model.save(os.path.join(save_dir, "last.pth"))
    if best_det_reward == -float("inf"):
        model.save(os.path.join(save_dir, "best.pth"))
        log("  [eval] no completing rollout found; best.pth = last.pth fallback")
    np.savez(os.path.join(log_dir, "UAV_%d_%s_rewards.npz"
                          % (uav_num, model_subdir)),
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64))
    arr = np.asarray(ep_rewards, dtype=np.float64)
    log("=== done === best_reward=%.2f avg_last100=%.2f nan_count=%d"
        % (best_reward, float(np.mean(arr[-100:])), int(np.isnan(arr).sum())))
    log_file.close()
    return {"ep_rewards": ep_rewards, "best_reward": best_reward, "model": model}
