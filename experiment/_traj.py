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


def train_td3(cfg, uav_traverse, total_episode=300, model_subdir="Proposed",
              model_dir=None, log_dir=None, log_every=10):
    """Train parameter-shared independent TD3 on the injected traverse.

    Mirrors ``trajectory/train.py`` but *replaces* the smoke ``balanced_traverse``
    with the real clustered + sequenced assignment.  Returns the trained model.
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
    action_range = max_action - min_action

    model = SharedTD3(state_dim, action_dim, max_action, min_action)
    replay = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    batch_size = model.batch_size

    save_dir = os.path.join(model_dir, "UAV_%d" % uav_num, model_subdir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "UAV_%d_%s.log" % (uav_num, model_subdir))

    expl_start, expl_end = 0.3, 0.02
    expl_decay = (expl_end / expl_start) ** (1.0 / max(int(total_episode), 1))
    best_reward = -float("inf")
    ep_rewards = []

    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=== Task 10 %s TD3 training (injected 4D+ga_eqtsp traverse) ==="
        % model_subdir)
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
                a = model.select_action(obs[i])
                noise = np.random.normal(0.0, 1.0, size=action_dim) \
                    * (expl_scale * action_range)
                actions.append(np.clip(a + noise, min_action, max_action))

            next_obs, rewards, dones, info = env.step(actions)
            terminal = bool(info["terminal"])
            for i in range(uav_num):
                replay.add(obs[i], actions[i], rewards[i], next_obs[i], terminal)

            if len(replay) >= batch_size:
                model.train(replay)

            ep_reward += float(np.sum(rewards))
            steps += 1
            obs = next_obs
            if terminal:
                break

        ep_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
            model.save(os.path.join(save_dir, "best.pth"))

        if (ep + 1) % log_every == 0 or ep == 0:
            log("ep %4d | reward %10.2f | avg100 %10.2f | steps %3d | expl %.3f"
                % (ep + 1, ep_reward, float(np.mean(ep_rewards[-100:])),
                   steps, expl_scale))

    model.save(os.path.join(save_dir, "last.pth"))
    np.savez(os.path.join(log_dir, "UAV_%d_%s_rewards.npz" % (uav_num, model_subdir)),
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64))
    arr = np.asarray(ep_rewards, dtype=np.float64)
    log("=== done === best_reward=%.2f avg_last100=%.2f nan_count=%d"
        % (best_reward, float(np.mean(arr[-100:])), int(np.isnan(arr).sum())))
    log_file.close()
    return model


def rollout_evaluate(cfg, uav_traverse, policy, pipeline_run_id, method,
                     brief_name, log_dir=None):
    """Deterministic rollout -> ``.npz`` log + metrics JSONs -> metrics dict.

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
