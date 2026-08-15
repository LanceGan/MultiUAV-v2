"""Layer-3 rollout harness (Task 8).

Loads a trained parameter-shared independent TD3 checkpoint, runs a single
*deterministic* rollout (no exploration noise — pure policy) on the multi-UAV
environment, records every UAV's trajectory / per-step data buffer / uplink
rate / G2A outage, and saves the raw log plus the episode-level metrics to
``results/logs/<run_id>.npz``. The companion ``experiment/evaluate.py`` turns
that raw log into the five real end-to-end metrics.

CLI::

    python trajectory/test.py --uav_num 2 --model_subdir Ours
    python trajectory/test.py --uav_num 2 --model_subdir Ours --model_episode final
    python trajectory/test.py --uav_num 2 --model_subdir Ours --run_id my_run
"""

import argparse
import glob
import os
import sys

# Make the project root importable when run as ``python trajectory/test.py``
# (the script's directory, not the root, is otherwise placed on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common.config import get_scenario
from trajectory.env import MultiUAVInspectionEnv
from trajectory.td3 import SharedTD3


def balanced_traverse(uav_num, n_points):
    """Split K points into N contiguous chunks (matches the Task 7 smoke run)."""
    return {i: list(range(i * n_points // uav_num, (i + 1) * n_points // uav_num))
            for i in range(uav_num)}


def resolve_model_path(model_dir, model_episode):
    """Map a ``--model_episode`` label to a concrete checkpoint path.

    ``best`` -> best.pth, ``final``/``last`` -> last.pth, ``stable`` -> the
    highest-numbered periodic checkpoint, anything else -> a literal filename.
    """
    name = (model_episode or "best").strip()
    if name == "best":
        fname = "best.pth"
    elif name in ("final", "last"):
        fname = "last.pth"
    elif name == "stable":
        ckpts = sorted(glob.glob(os.path.join(model_dir, "checkpoint_ep*.pth")))
        fname = os.path.basename(ckpts[-1]) if ckpts else "last.pth"
    else:
        fname = name if name.endswith(".pth") else name + ".pth"
    return os.path.join(model_dir, fname)


def rollout(env, model, uav_traverse, min_action=None, max_action=None):
    """Run one deterministic rollout and return the recorded log dict.

    Returns a dict with (per-UAV lists/arrays): ``trajectories`` (N, 2) per UAV,
    ``times`` (completion step, -1 if unfinished), ``data_buffers`` (Mbit),
    ``outage_steps`` (G2A outage probability), ``rates`` (A2G Mbps), plus the
    episode-level ``energies``, ``tau_history``, ``outage_duration``,
    ``offloaded_data``, ``rate_sum``, ``rate_steps``.
    """
    uav_num = env.uav_num
    if min_action is None:
        min_action = np.zeros(env.action_dim, dtype=np.float32)
    if max_action is None:
        max_action = np.array([env.v_max, 2.0 * np.pi], dtype=np.float32)
    min_action = np.asarray(min_action, dtype=np.float32)
    max_action = np.asarray(max_action, dtype=np.float32)

    env.set_uav_traverse(uav_traverse)
    obs = env.reset()

    # Per-step records (aligned with the active steps of each UAV).
    data_buffers = [[float(env._data_buffer[i])] for i in range(uav_num)]
    rates = [[] for _ in range(uav_num)]
    outage_steps = [[] for _ in range(uav_num)]

    while True:
        active_before = [not bool(env._done[i]) for i in range(uav_num)]

        actions = []
        for i in range(uav_num):
            a = model.select_action(obs[i])          # pure policy, no noise
            a = np.clip(a, min_action, max_action)
            actions.append(a)

        obs, rewards, dones, info = env.step(actions)

        for i in range(uav_num):
            if active_before[i]:
                data_buffers[i].append(float(env._data_buffer[i]))
                rates[i].append(float(info["rate"][i]))
                outage_steps[i].append(float(info["outage"][i]))

        if info["terminal"] or info["truncated"]:
            break

    trajectories = [np.asarray(env._trajectories[i], dtype=np.float64)
                    for i in range(uav_num)]
    times = np.asarray([t if t is not None else -1.0 for t in info["times"]],
                       dtype=np.float64)

    return {
        "trajectories": trajectories,
        "times": times,
        "data_buffers": [np.asarray(b, dtype=np.float64) for b in data_buffers],
        "outage_steps": [np.asarray(o, dtype=np.float64) for o in outage_steps],
        "rates": [np.asarray(r, dtype=np.float64) for r in rates],
        "energies": np.asarray(info["energy"], dtype=np.float64),
        "tau_history": [np.asarray(t, dtype=np.float64)
                        for t in info["tau_history"]],
        "outage_duration": np.asarray(info["outage_duration"], dtype=np.float64),
        "offloaded_data": np.asarray(info["offloaded_data"], dtype=np.float64),
        "rate_sum": np.asarray(info["rate_sum"], dtype=np.float64),
        "rate_steps": np.asarray(info["rate_steps"], dtype=np.int64),
        "done": np.asarray(info["done"], dtype=bool),
        "truncated": bool(info["truncated"]),
        "t": int(info["t"]),
    }


def save_log(path, log, uav_num, cfg):
    """Serialize a rollout log dict to an .npz file (per-UAV variable-length
    sequences are stored under indexed keys, e.g. ``traj_0``, ``rate_1``)."""
    data = {
        "uav_num": np.asarray(uav_num, dtype=np.int64),
        "delta_t": np.asarray(float(cfg["delta_t"]), dtype=np.float64),
        "T": np.asarray(int(cfg.get("T", 200)), dtype=np.int64),
        "times": log["times"],
        "energies": log["energies"],
        "outage_duration": log["outage_duration"],
        "offloaded_data": log["offloaded_data"],
        "rate_sum": log["rate_sum"],
        "rate_steps": log["rate_steps"],
        "done": log["done"],
        "truncated": np.asarray(log["truncated"]),
        "t": np.asarray(log["t"], dtype=np.int64),
    }
    for i in range(uav_num):
        data["traj_%d" % i] = log["trajectories"][i]
        data["buffer_%d" % i] = log["data_buffers"][i]
        data["outage_%d" % i] = log["outage_steps"][i]
        data["rate_%d" % i] = log["rates"][i]
        data["tau_%d" % i] = log["tau_history"][i]
    np.savez(path, **data)


def main(uav_num=2, model_subdir="Ours", model_episode="best", run_id=None,
         model_dir="results/models", log_dir="results/logs"):
    cfg = get_scenario(uav_num)
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    env = MultiUAVInspectionEnv(cfg, safe_distance=0.1)
    n_points = int(env.points.shape[0])
    traverse = balanced_traverse(uav_num, n_points)
    env.set_uav_traverse(traverse)

    state_dim = env.local_obs_dim
    action_dim = env.action_dim
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)

    model = SharedTD3(state_dim, action_dim, max_action, min_action)
    ckpt = resolve_model_path(
        os.path.join(model_dir, "UAV_%d" % uav_num, model_subdir),
        model_episode)
    if not os.path.exists(ckpt):
        raise FileNotFoundError("model checkpoint not found: %s" % ckpt)
    model.load(ckpt)
    print("loaded checkpoint -> %s" % ckpt)

    if run_id is None:
        run_id = "UAV_%d_%s_%s" % (uav_num, model_subdir, model_episode)
    os.makedirs(log_dir, exist_ok=True)

    log = rollout(env, model, traverse,
                  min_action=min_action, max_action=max_action)

    npz_path = os.path.join(log_dir, "%s.npz" % run_id)
    save_log(npz_path, log, uav_num, cfg)

    print("run_id        = %s" % run_id)
    print("terminated    = %s (truncated=%s, steps=%d)"
          % ("truncated" if log["truncated"] else "completed",
             log["truncated"], log["t"]))
    print("saved rollout log -> %s" % npz_path)
    return {"run_id": run_id, "npz_path": npz_path, "log": log}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a deterministic rollout with a trained TD3 (Task 8).")
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--model_subdir", type=str, default="Ours")
    parser.add_argument("--model_episode", type=str, default="best")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="results/models")
    parser.add_argument("--log_dir", type=str, default="results/logs")
    args = parser.parse_args()
    main(uav_num=args.uav_num, model_subdir=args.model_subdir,
         model_episode=args.model_episode, run_id=args.run_id,
         model_dir=args.model_dir, log_dir=args.log_dir)
