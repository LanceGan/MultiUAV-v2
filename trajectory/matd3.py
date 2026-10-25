"""Multi-Agent TD3 (MATD3) for the layer-3 trajectory problem (algorithm comparison).

Centralised Training with Decentralised Execution (CTDE): ``n_agents`` independent
per-agent actors (each maps one UAV's local observation to its ``[v, theta]``
action) plus a single centralised twin critic whose input is the *joint* state
(all agents' observations concatenated, ``n_agents * local_state_dim``) and the
*joint* action (all agents' actions concatenated, ``n_agents * action_dim``).

TD3 triple (the DDPG enhancements kept by MATD3): twin-Q minimum for the TD
target, target policy smoothing (``policy_noise`` / ``noise_clip`` scaled
per-dimension, with the angle smoothed in radians rather than its full ``2*pi``
span), and delayed actor / target updates (every ``policy_update_delay`` critic
updates).

Conventions (matching the unified algorithm interface):

* the replay reward is the *team* reward (sum of per-agent rewards);
* the replay ``done`` flag is ``info["terminal"]`` (true completion only — a
  horizon truncation bootstraps as ``done=0``);
* exploration noise is per-dimension ``N(0,1) * noise_scale`` with
  ``noise_scale = [v_max, 1.0]``, and actions are clipped to
  ``[min_action, max_action]``.
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
    """Per-agent decentralised policy: local obs -> tanh -> [min, max] action."""

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
    """Centralised twin critic (Q1, Q2) on concatenated joint state + action."""

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


class MATD3(object):
    def __init__(self, n_agents, local_state_dim, action_dim, max_action,
                 min_action, net_width=256, critic_width=512, a_lr=1e-4,
                 c_lr=1e-3, gamma=0.99, batch_size=256, tau=0.005,
                 policy_noise_scale=0.2, noise_clip_scale=0.5,
                 policy_update_delay=2):
        self.n_agents = int(n_agents)
        self.local_state_dim = int(local_state_dim)
        self.action_dim = int(action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_update_delay = policy_update_delay

        # Joint state / action dimensions (centralised critic input).
        self.state_dim = self.n_agents * self.local_state_dim
        self.joint_action_dim = self.n_agents * self.action_dim

        max_action = _to_tensor(max_action)
        min_action = _to_tensor(min_action)
        self.max_action = max_action
        self.min_action = min_action
        # Tiled bounds for the concatenated joint action (used in train()).
        self.joint_max_action = self.max_action.repeat(self.n_agents)
        self.joint_min_action = self.min_action.repeat(self.n_agents)
        # numpy copies for per-agent clipping in select_actions().
        self.max_action_np = max_action.cpu().numpy()
        self.min_action_np = min_action.cpu().numpy()

        # n_agents independent actors + targets + one optimizer each.
        self.actors = nn.ModuleList([
            Actor(local_state_dim, action_dim, net_width, max_action, min_action)
            for _ in range(self.n_agents)]).to(device)
        self.actor_targets = nn.ModuleList([copy.deepcopy(a) for a in self.actors])
        self.actor_optimizers = [
            torch.optim.Adam(a.parameters(), lr=a_lr) for a in self.actors]

        # One centralised twin critic.
        self.critic = Critic(self.state_dim, self.joint_action_dim,
                             critic_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        # Per-dimension target-policy smoothing scale, tiled over agents. The
        # angle is smoothed in RADIANS (0.5 rad ~ 29 deg), not by its full
        # 2*pi span, matching the parameter-shared TD3 convention.
        smooth_per_agent = torch.tensor([float(max_action[0]), 0.5],
                                        dtype=torch.float32, device=device)
        smooth_scale = smooth_per_agent.repeat(self.n_agents)
        self.policy_noise = policy_noise_scale * smooth_scale
        self.noise_clip = noise_clip_scale * smooth_scale

        self._update_counter = 0

    def select_actions(self, obs_list, add_noise=False, noise_scale=0.1):
        """Return one action ``[v, theta]`` per agent from its local observation."""
        noise_scale = np.asarray(noise_scale, dtype=np.float32)
        actions = []
        for i in range(self.n_agents):
            with torch.no_grad():
                s = torch.FloatTensor(
                    np.asarray(obs_list[i], dtype=np.float32).reshape(1, -1)).to(device)
                a = self.actors[i](s).cpu().numpy().flatten()
            if add_noise:
                noise = np.random.normal(0.0, 1.0, size=self.action_dim) * noise_scale
                a = np.clip(a + noise, self.min_action_np, self.max_action_np)
            else:
                a = np.clip(a, self.min_action_np, self.max_action_np)
            actions.append(a.astype(np.float32))
        return actions

    def train(self, replay_buffer):
        """One TD3 update on a sampled joint transition batch. Returns diagnostics."""
        s, a, r, s_prime, done = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Each agent's target actor maps its own slice of the joint next obs.
            next_actions = []
            for i in range(self.n_agents):
                obs_i = s_prime[:, i * self.local_state_dim:(i + 1) * self.local_state_dim]
                next_actions.append(self.actor_targets[i](obs_i))
            next_action = torch.cat(next_actions, dim=1)
            # Target policy smoothing (TD3).
            noise = torch.clamp(torch.randn_like(next_action) * self.policy_noise,
                                -self.noise_clip, self.noise_clip)
            next_action = torch.max(torch.min(next_action + noise, self.joint_max_action),
                                    self.joint_min_action)

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
            # Joint action under the current (non-target) actors; all agents'
            # actions carry gradient, and the joint policy gradient flows through
            # the centralised critic's Q1.
            new_actions = []
            for i in range(self.n_agents):
                obs_i = s[:, i * self.local_state_dim:(i + 1) * self.local_state_dim]
                new_actions.append(self.actors[i](obs_i))
            joint_action = torch.cat(new_actions, dim=1)
            actor_loss = -self.critic.Q1(s, joint_action).mean()

            for opt in self.actor_optimizers:
                opt.zero_grad()
            actor_loss.backward()
            for actor, opt in zip(self.actors, self.actor_optimizers):
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt.step()

            # Soft-update target networks.
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            for actor, actor_target in zip(self.actors, self.actor_targets):
                for param, target_param in zip(actor.parameters(),
                                               actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": (float(actor_loss.detach().cpu().item())
                           if actor_loss is not None else None),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        for i in range(self.n_agents):
            ckpt["actor_%d" % i] = self.actors[i].state_dict()
            ckpt["actor_target_%d" % i] = self.actor_targets[i].state_dict()
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(ckpt["actor_%d" % i])
            self.actor_targets[i].load_state_dict(ckpt["actor_target_%d" % i])


def _deterministic_reward(env, model, uav_traverse):
    """Deterministic rollout reward (no exploration noise) for best-checkpoint
    selection, so the saved policy is not noise-dependent."""
    env.set_uav_traverse(uav_traverse)
    obs = env.reset()
    total = 0.0
    while True:
        actions = model.select_actions(obs, add_noise=False)
        obs, rewards, dones, info = env.step(actions)
        total += float(np.sum(rewards))
        if info["terminal"] or info["truncated"]:
            break
    # Only a COMPLETING rollout is a valid best-checkpoint candidate.
    if not info["terminal"]:
        return -float("inf")
    return total


def train_matd3(cfg, uav_traverse, total_episode=2000, model_subdir="MATD3",
                model_dir=None, log_dir=None, log_every=10, use_a2g=True,
                use_comm=True):
    """Train MATD3 (CTDE) on the injected ``uav_traverse``.

    Returns ``{"ep_rewards": [...], "best_reward": float, "model": MATD3}`` and
    writes the reward history to ``results/logs/UAV_<N>_<Algo>_rewards.npz``.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = model_dir or os.path.join(root, "results", "models")
    log_dir = log_dir or os.path.join(root, "results", "logs")

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

    model = MATD3(uav_num, local_state_dim, action_dim, max_action, min_action)
    # One buffer of JOINT transitions: joint state / joint action / team reward.
    replay = ReplayBuffer(model.state_dim, model.joint_action_dim,
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

    log("=== MATD3 training (CTDE) ===")
    log("uav_num=%d total_episode=%d seed=%d local_state_dim=%d action_dim=%d T=%d"
        % (uav_num, total_episode, seed, local_state_dim, action_dim, env.T))

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

            # Store one joint transition per step.
            global_state = np.concatenate(obs)
            joint_action = np.concatenate(actions)
            next_global_state = np.concatenate(next_obs)
            team_reward = float(np.sum(rewards))
            replay.add(global_state, joint_action, team_reward,
                       next_global_state, done)

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

        # Best checkpoint by a DETERMINISTIC rollout (no noise), not the noisy
        # training reward.
        if (ep + 1) % eval_interval == 0 or ep == 0:
            det_reward = _deterministic_reward(env, model, uav_traverse)
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
    np.savez(os.path.join(log_dir, "UAV_%d_%s_rewards.npz" % (uav_num, model_subdir)),
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64))
    arr = np.asarray(ep_rewards, dtype=np.float64)
    log("=== done === best_reward=%.2f avg_last100=%.2f nan_count=%d"
        % (best_reward, float(np.mean(arr[-100:])), int(np.isnan(arr).sum())))
    log_file.close()

    return {"ep_rewards": ep_rewards, "best_reward": best_reward, "model": model}
