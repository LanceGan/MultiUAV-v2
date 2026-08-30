"""Shared layer-3 trajectory helpers for Task 10 (``run_traj`` / ``run_end2end``).

Contains the deterministic non-comm-aware *greedy* baseline policy, the TD3
training loop that trains on an *injected* ``uav_traverse`` (4D clustering +
ga_eqtsp), and a rollout-and-evaluate helper that writes the raw ``.npz`` log
plus the five real end-to-end metrics (computed by ``experiment.evaluate``).
"""

import math
import os

import numpy as np
import torch

from experiment import _common, evaluate
from trajectory import test as traj_test
from trajectory.env import MultiUAVInspectionEnv
from trajectory.replay import ReplayBuffer
from trajectory.td3 import SharedTD3


class GreedyPolicy(object):
    """Deterministic non-comm-aware waypoint follower (Task 10 baseline).

    Flies straight toward the current target at a fixed speed (``v_max``), with
    no radio-map awareness and no offload optimisation.  The heading is read
    from the normalised target-relative offset already present in each UAV's
    local observation (indices 2 and 3).
    """

    def __init__(self, length, width, v_max):
        self.length = float(length)
        self.width = float(width)
        self.v_max = float(v_max)

    def select_action(self, obs):
        dx = float(obs[2]) * self.length
        dy = float(obs[3]) * self.width
        theta = math.atan2(dy, dx)
        if theta < 0.0:
            theta += 2.0 * math.pi
        return np.array([self.v_max, theta], dtype=np.float32)


def _make_env(cfg, uav_traverse):
    env = MultiUAVInspectionEnv(cfg, safe_distance=0.1)
    env.set_uav_traverse(uav_traverse)
    return env


def _deterministic_reward(env, agents, uav_traverse, uav_num, min_action, max_action):
    """Deterministic rollout reward (no exploration noise) — used to select the
    best checkpoint so the saved policy is not noise-dependent.

    ``agents`` is a per-UAV list of ``SharedTD3`` policies: UAV ``i`` acts from
    ``agents[i]`` (independent learners — no parameter sharing).
    """
    env.set_uav_traverse(uav_traverse)
    obs = env.reset()
    total = 0.0
    while True:
        actions = [np.clip(agents[i].select_action(obs[i]), min_action, max_action)
                   for i in range(uav_num)]
        obs, rewards, dones, info = env.step(np.asarray(actions, dtype=float))
        total += float(np.sum(rewards))
        if info["terminal"] or info["truncated"]:
            break
    return total


def train_td3(cfg, uav_traverse, total_episode=2000, model_subdir="Proposed",
              model_dir=None, log_dir=None, log_every=10):
    """Train per-agent independent TD3 (IL MARL) on the injected traverse.

    Each UAV owns its own ``SharedTD3`` (actor + twin critic) and its own replay
    buffer, trained only on its own transitions — no parameter sharing, no
    centralized critic.  This fixes the "one UAV finishes, the other doesn't"
    failure: a single shared policy cannot simultaneously learn two UAVs'
    different 4D-clustered target sequences.

    Mirrors ``trajectory/train.py`` but *replaces* the smoke ``balanced_traverse``
    with the real clustered + sequenced assignment.  Returns the list of agents.
    """
    model_dir = model_dir or _common.MODEL_DIR
    log_dir = log_dir or _common.LOG_DIR
    uav_num = int(cfg["uav_num"])
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(cfg, uav_traverse)
    state_dim = env.local_obs_dim
    action_dim = env.action_dim
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    # Per-dimension exploration noise scale: v in [0, v_max], theta in radians
    # (NOT scaled by 2*pi, which made the angle noise ~108 deg and untunable).
    noise_scale = np.array([float(cfg["v_max"]), 1.0], dtype=np.float32)

    # Per-agent independent learners: one actor + twin critic + replay per UAV.
    agents = [SharedTD3(state_dim, action_dim, max_action, min_action)
              for _ in range(uav_num)]
    replays = [ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
               for _ in range(uav_num)]
    batch_size = agents[0].batch_size

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

    log("=== Task 10 %s per-agent independent TD3 training "
        "(injected 4D+ga_eqtsp traverse) ===" % model_subdir)
    log("uav_num=%d total_episode=%d seed=%d state_dim=%d action_dim=%d T=%d"
        % (uav_num, total_episode, seed, state_dim, action_dim, env.T))

    for ep in range(int(total_episode)):
        obs = env.reset()
        ep_reward = 0.0
        steps = 0
        expl_scale = expl_start * (expl_decay ** ep)
        while True:
            actions = []
            for i in range(uav_num):
                a = agents[i].select_action(obs[i])
                noise = np.random.normal(0.0, 1.0, size=action_dim) \
                    * (expl_scale * noise_scale)
                actions.append(np.clip(a + noise, min_action, max_action))

            next_obs, rewards, dones, info = env.step(actions)
            # ``done`` for replay bootstrapping = true completion only; a timeout
            # (truncation) must bootstrap as done=0 so the value is not biased down.
            done = bool(info["terminal"])
            episode_done = bool(info["terminal"]) or bool(info["truncated"])
            for i in range(uav_num):
                # Each UAV stores + trains on its OWN replay only (independent
                # learners — no pooled transitions, no shared critic).
                replays[i].add(obs[i], actions[i], rewards[i], next_obs[i], done)

                if len(replays[i]) >= batch_size:
                    agents[i].train(replays[i])

            ep_reward += float(np.sum(rewards))
            steps += 1
            obs = next_obs
            if episode_done:
                break

        ep_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
        # Select the BEST checkpoint by a DETERMINISTIC rollout (no noise), not
        # by the noisy training reward — a policy can be noise-dependent and
        # fail when the exploration noise is removed.  The deterministic reward
        # is the SUM over all UAVs, evaluated with each UAV's own agent.
        if (ep + 1) % eval_interval == 0 or ep == 0:
            det_reward = _deterministic_reward(
                env, agents, uav_traverse, uav_num, min_action, max_action)
            if det_reward > best_det_reward:
                best_det_reward = det_reward
                for i in range(uav_num):
                    agents[i].save(os.path.join(save_dir, "best_agent%d.pth" % i))
                log("  [eval] ep %4d | det_reward %10.2f (new best)"
                    % (ep + 1, det_reward))

        if (ep + 1) % log_every == 0 or ep == 0:
            log("ep %4d | reward %10.2f | avg100 %10.2f | steps %3d | expl %.3f"
                % (ep + 1, ep_reward, float(np.mean(ep_rewards[-100:])),
                   steps, expl_scale))

    for i in range(uav_num):
        agents[i].save(os.path.join(save_dir, "last_agent%d.pth" % i))
    np.savez(os.path.join(log_dir, "UAV_%d_%s_rewards.npz" % (uav_num, model_subdir)),
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64))
    arr = np.asarray(ep_rewards, dtype=np.float64)
    log("=== done === best_reward=%.2f avg_last100=%.2f nan_count=%d"
        % (best_reward, float(np.mean(arr[-100:])), int(np.isnan(arr).sum())))
    log_file.close()
    return agents


def load_td3(cfg, uav_traverse, model_subdir="Proposed", model_dir=None,
             model_episode="best"):
    """Load previously-trained per-agent TD3 checkpoints instead of retraining.

    Returns a LIST of ``SharedTD3`` agents (one per UAV), each loaded from
    ``<model_dir>/UAV_<N>/<model_subdir>/<episode>_agent<i>.pth``.

    Used by ``run_end2end`` to reuse the layer-3 ``Proposed`` models produced by
    ``run_traj`` (same 4D clustering + ga_eqtsp assignment) rather than training
    identical networks a second time.
    """
    model_dir = model_dir or _common.MODEL_DIR
    uav_num = int(cfg["uav_num"])
    env = _make_env(cfg, uav_traverse)
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    save_dir = os.path.join(model_dir, "UAV_%d" % uav_num, model_subdir)
    agents = []
    for i in range(uav_num):
        agent = SharedTD3(env.local_obs_dim, env.action_dim,
                          max_action, min_action)
        ckpt = traj_test.resolve_agent_model_path(save_dir, model_episode, i)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                "reuse of '%s' checkpoint requested but not found: %s "
                "(run experiment/run_traj.py first)" % (model_subdir, ckpt))
        agent.load(ckpt)
        agents.append(agent)
    print("reused %d checkpoint(s) -> %s" % (uav_num, save_dir))
    return agents


def rollout_evaluate(cfg, uav_traverse, policy, pipeline_run_id, method,
                     brief_name, log_dir=None):
    """Deterministic rollout -> ``.npz`` log + metrics JSONs -> metrics dict.

    ``policy`` is either a single stateless policy used by all UAVs (the greedy
    baseline) or a per-UAV LIST of ``SharedTD3`` agents (the proposed model).
    It is passed straight through to ``traj_test.rollout``.

    Writes three artifacts per run:
    * ``<log_dir>/<pipeline_run_id>.npz``            (rollout log for plotting),
    * ``<log_dir>/<pipeline_run_id>.metrics.json``   (plot-compatible metrics),
    * ``<log_dir>/<brief_name>.json``                (Task-10 brief filename).
    """
    log_dir = log_dir or _common.LOG_DIR
    uav_num = int(cfg["uav_num"])

    env = _make_env(cfg, uav_traverse)
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)

    log = traj_test.rollout(env, policy, uav_traverse,
                            min_action=min_action, max_action=max_action)

    npz_path = os.path.join(log_dir, "%s.npz" % pipeline_run_id)
    traj_test.save_log(npz_path, log, uav_num, cfg)

    metrics = evaluate.evaluate_log(np.load(npz_path))
    metrics.update({
        "run_id": pipeline_run_id,
        "uav_num": uav_num,
        "method": method,
        "seed": int(cfg["seed"]),
    })
    _common.save_json(os.path.join(log_dir, "%s.metrics.json" % pipeline_run_id),
                      metrics)

    brief = dict(metrics)
    brief["run_id"] = brief_name
    _common.save_json(os.path.join(log_dir, "%s.json" % brief_name), brief)
    print("saved rollout -> %s" % npz_path)
    return metrics
