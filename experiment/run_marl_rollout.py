"""Deterministic rollout comparison for the four MARL algorithms.

Loads each algorithm's ``best.pth`` and runs ONE deterministic rollout (no
exploration noise) on the SAME environment / 4D-clustering / GA_EQTSP sequence,
then writes the full rollout ``.npz`` + ``.metrics.json`` (same pipeline as the
greedy baseline and the SCA baseline) so ``plots/plot_results.py`` renders them
alongside the other methods.  Also prints a compact summary table.

CLI::

    python experiment/run_marl_rollout.py               # all 4, N=2
    python experiment/run_marl_rollout.py --algorithms MADDPG MAPPO
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiment import _common, evaluate
from sequence import ga_eqtsp
from trajectory import test as traj_test
from trajectory.env import MultiUAVInspectionEnv
from trajectory.td3 import SharedTD3
from trajectory import maddpg, matd3, mappo

ALGORITHMS = ("ILD3", "MADDPG", "MATD3", "MAPPO")


def _select(model, obs_list):
    """Pick actions for all UAVs; works for both ``select_actions(obs_list)`` and
    ``select_action(obs)`` interfaces."""
    if hasattr(model, "select_actions"):
        return model.select_actions(obs_list)
    return [model.select_action(o) for o in obs_list]


def _make_model(algo, n_agents, local_state_dim, action_dim, max_action, min_action):
    if algo == "ILD3":
        return SharedTD3(local_state_dim, action_dim, max_action, min_action)
    if algo == "MADDPG":
        return maddpg.MADDPG(n_agents, local_state_dim, action_dim, max_action, min_action)
    if algo == "MATD3":
        return matd3.MATD3(n_agents, local_state_dim, action_dim, max_action, min_action)
    if algo == "MAPPO":
        return mappo.MAPPO(n_agents, local_state_dim, action_dim, max_action, min_action)
    raise ValueError("unknown algorithm: %s" % algo)


def marl_rollout(env, model, uav_traverse):
    """Deterministic rollout -> full log dict (mirrors ``traj_test.rollout``)."""
    uav_num = env.uav_num
    env.set_uav_traverse(uav_traverse)
    obs = env.reset()

    min_action = np.zeros(env.action_dim, dtype=np.float32)
    max_action = np.array([env.v_max, 2.0 * np.pi], dtype=np.float32)

    data_buffers = [[float(env._data_buffer[i])] for i in range(uav_num)]
    rates = [[] for _ in range(uav_num)]
    outage_steps = [[] for _ in range(uav_num)]

    while True:
        active_before = [not bool(env._done[i]) for i in range(uav_num)]
        actions = [np.clip(a, min_action, max_action) for a in _select(model, obs)]
        obs, rewards, dones, info = env.step(np.asarray(actions, dtype=float))
        for i in range(uav_num):
            if active_before[i]:
                data_buffers[i].append(float(env._data_buffer[i]))
                rates[i].append(float(info["rate"][i]))
                outage_steps[i].append(float(info["outage"][i]))
        if info["terminal"] or info["truncated"]:
            break

    return {
        "trajectories": [np.asarray(env._trajectories[i], dtype=np.float64)
                         for i in range(uav_num)],
        "times": np.asarray([t if t is not None else -1.0 for t in info["times"]],
                            dtype=np.float64),
        "data_buffers": [np.asarray(b, dtype=np.float64) for b in data_buffers],
        "outage_steps": [np.asarray(o, dtype=np.float64) for o in outage_steps],
        "rates": [np.asarray(r, dtype=np.float64) for r in rates],
        "energies": np.asarray(info["energy"], dtype=np.float64),
        "tau_history": [np.asarray(t, dtype=np.float64) for t in info["tau_history"]],
        "outage_duration": np.asarray(info["outage_duration"], dtype=np.float64),
        "offloaded_data": np.asarray(info["offloaded_data"], dtype=np.float64),
        "rate_sum": np.asarray(info["rate_sum"], dtype=np.float64),
        "rate_steps": np.asarray(info["rate_steps"], dtype=np.int64),
        "done": np.asarray(info["done"], dtype=bool),
        "truncated": bool(info["truncated"]),
        "t": int(info["t"]),
    }


def run(uav_num=2, seed=42, algorithms=ALGORITHMS, log_dir=None):
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
    seed = int(cfg["seed"])
    labels = _common.cluster_labels("4d", points, data_sizes, uav_num, seed)
    clusters = _common.split_clusters(labels, points, data_sizes, uav_num)
    orders = []
    for cpts, _gidx, cds in clusters:
        _, o, _ = ga_eqtsp.ga_eqtsp(cpts, cfg["ini_loc"], cfg["end_loc"], cds, cfg)
        orders.append(o)
    traverse = _common.build_traverse(clusters, orders)

    env = MultiUAVInspectionEnv(cfg, safe_distance=0.1)
    local_state_dim = env.local_obs_dim
    action_dim = env.action_dim
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)

    model_dir = os.path.join("results", "models", "UAV_%d" % uav_num)
    print("%-8s %-9s %-8s %-8s %-8s %-8s %s" % ("algo", "all_done", "makespan", "outage", "delay", "rate", "times"))
    for algo in algorithms:
        model = _make_model(algo, uav_num, local_state_dim, action_dim,
                            max_action, min_action)
        model.load(os.path.join(model_dir, algo, "best.pth"))

        log = marl_rollout(env, model, traverse)

        run_id = "UAV_%d_%s" % (uav_num, algo)
        npz_path = os.path.join(log_dir, "%s.npz" % run_id)
        traj_test.save_log(npz_path, log, uav_num, cfg)

        metrics = evaluate.evaluate_log(np.load(npz_path))
        metrics.update({"run_id": run_id, "uav_num": uav_num,
                        "method": algo, "seed": seed})
        _common.save_json(os.path.join(log_dir, "%s.metrics.json" % run_id),
                          metrics)

        times = [float(t) for t in log["times"]]
        all_done = all(t >= 0 for t in times)
        print("%-8s %-9s %-8.1f %-8.1f %-8.1f %-8.2f %s"
              % (algo, str(all_done), metrics["makespan"],
                 metrics["outage_duration"], metrics["total_delay"],
                 metrics["avg_uplink_rate"], [int(t) for t in times]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic rollout comparison of MARL algorithms.")
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithms", type=str, nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args(argv)
    return run(uav_num=args.uav_num, seed=args.seed, algorithms=args.algorithms,
               log_dir=args.log_dir)


if __name__ == "__main__":
    main()
