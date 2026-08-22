"""Layer-3 parameter-shared independent TD3 training loop (Task 7).

CLI::

    python trajectory/train.py --uav_num 2 --total_episode 300 --model_subdir Ours

Smoke training uses a SIMPLE balanced assignment (full clustering + sequence
integration is Task 10): the K points are split into N contiguous chunks and
``uav_traverse = {i: list(range(i*K//N, (i+1)*K//N))}`` is injected via
``env.set_uav_traverse``.

Every episode stores each UAV's individual transition into the shared buffer
using ``info["terminal"]`` (true completion only) as the replay done flag (a
horizon timeout bootstraps as done=0), and the shared networks train on the
pooled samples. The rollout loop stops on
``info["terminal"] or info["truncated"]``.
"""

import argparse
import os
import sys

# Make the project root importable when run as ``python trajectory/train.py``
# (the script's directory, not the root, is otherwise placed on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common.config import get_scenario
from trajectory.env import MultiUAVInspectionEnv
from trajectory.replay import ReplayBuffer
from trajectory.td3 import SharedTD3


def balanced_traverse(uav_num, n_points):
    """Split K points into N contiguous chunks (smoke-training assignment)."""
    return {i: list(range(i * n_points // uav_num, (i + 1) * n_points // uav_num))
            for i in range(uav_num)}


def main(uav_num=2, model_subdir="Ours", total_episode=2000, seed=None,
         model_dir="results/models", log_dir="results/logs"):
    cfg = get_scenario(uav_num)
    if seed is not None:
        cfg["seed"] = int(seed)
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    env = MultiUAVInspectionEnv(cfg, safe_distance=0.1)
    state_dim = env.local_obs_dim
    action_dim = env.action_dim

    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    action_range = max_action - min_action

    n_points = int(env.points.shape[0])
    env.set_uav_traverse(balanced_traverse(uav_num, n_points))

    model = SharedTD3(state_dim, action_dim, max_action, min_action)
    replay = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    batch_size = model.batch_size

    save_dir = os.path.join(model_dir, "UAV_%d" % uav_num, model_subdir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "UAV_%d_%s.log" % (uav_num, model_subdir))

    # Exploration: dimension-scaled Gaussian noise, decaying over episodes.
    expl_start = 0.3
    expl_end = 0.02
    expl_decay = (expl_end / expl_start) ** (1.0 / max(int(total_episode), 1))

    best_reward = -float("inf")
    total_steps = 0
    ep_rewards = []

    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=== Task 7 parameter-shared independent TD3 training ===")
    log("uav_num=%d total_episode=%d model_subdir=%s seed=%d"
        % (uav_num, total_episode, model_subdir, cfg["seed"]))
    log("state_dim=%d action_dim=%d n_points=%d T=%d"
        % (state_dim, action_dim, n_points, env.T))

    for ep in range(int(total_episode)):
        obs = env.reset()
        ep_reward = 0.0
        ep_energy = 0.0
        steps = 0
        success = 0.0
        expl_scale = expl_start * (expl_decay ** ep)

        while True:
            actions = []
            for i in range(uav_num):
                a = model.select_action(obs[i])
                noise = np.random.normal(0.0, 1.0, size=action_dim) \
                    * (expl_scale * action_range)
                a = np.clip(a + noise, min_action, max_action)
                actions.append(a)

            next_obs, rewards, dones, info = env.step(actions)
            # ``done`` for replay bootstrapping = true completion only; a timeout
            # (truncation) must bootstrap as done=0 so the value is not biased down.
            done = bool(info["terminal"])
            episode_done = bool(info["terminal"]) or bool(info["truncated"])
            for i in range(uav_num):
                replay.add(obs[i], actions[i], rewards[i],
                           next_obs[i], done)

            if len(replay) >= batch_size:
                model.train(replay)

            ep_reward += float(np.sum(rewards))
            steps += 1
            total_steps += 1
            obs = next_obs

            if episode_done:
                success = 0.0 if info["truncated"] else 1.0
                ep_energy = float(np.sum(info["energy"]))  # cumulative, final
                break

        ep_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
            model.save(os.path.join(save_dir, "best.pth"))

        if (ep + 1) % 50 == 0:
            model.save(os.path.join(save_dir, "checkpoint_ep%d.pth" % (ep + 1)))

        if (ep + 1) % 10 == 0 or ep == 0:
            avg100 = float(np.mean(ep_rewards[-100:]))
            log("ep %4d | reward %10.2f | avg100 %10.2f | success %5.2f | "
                "energy %8.2f | steps %3d | expl %.3f"
                % (ep + 1, ep_reward, avg100, success, ep_energy, steps,
                   expl_scale))

    model.save(os.path.join(save_dir, "last.pth"))

    # Structured reward history for Task 9 plotting (minimal, non-breaking).
    np.savez(os.path.join(log_dir, "UAV_%d_%s_rewards.npz" % (uav_num, model_subdir)),
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64))

    arr = np.asarray(ep_rewards, dtype=np.float64)
    nan_count = int(np.isnan(arr).sum())
    log("=== done === best_reward=%.2f avg_last100=%.2f nan_count=%d "
        "total_steps=%d" % (best_reward, float(np.mean(arr[-100:])),
                            nan_count, total_steps))
    log_file.close()

    return {"best_reward": best_reward, "nan_count": nan_count,
            "episode_rewards": ep_rewards}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train parameter-shared independent TD3 (Task 7).")
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--total_episode", type=int, default=2000)
    parser.add_argument("--model_subdir", type=str, default="Ours")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(uav_num=args.uav_num, model_subdir=args.model_subdir,
         total_episode=args.total_episode, seed=args.seed)
