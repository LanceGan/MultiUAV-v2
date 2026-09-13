"""Deterministic rollout performance comparison for the four MARL algorithms.

Loads each algorithm's ``best.pth`` (the checkpoint the trainers already picked
by their own criterion) and runs ONE deterministic rollout (no exploration
noise) on the SAME environment / 4D-clustering / GA_EQTSP sequence, reporting
the final task metrics: whether both UAVs finish, makespan, outage duration,
transmission delay, and mean uplink rate.

This complements the training-reward-curve comparison: a reward curve shows
training stability, while this shows the actual task performance of the saved
policy.

CLI::

    python experiment/run_marl_rollout.py               # all 4, N=2
    python experiment/run_marl_rollout.py --algorithms MADDPG MAPPO
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiment import _common
from sequence import ga_eqtsp
from trajectory.env import MultiUAVInspectionEnv
from trajectory.td3 import SharedTD3
from trajectory import maddpg, matd3, mappo

ALGORITHMS = ("ILD3", "MADDPG", "MATD3", "MAPPO")


def _select(model, obs_list):
    """Pick actions for all UAVs; works for both `select_actions(obs_list)` and
    `select_action(obs)` interfaces."""
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


def _rollout_metrics(env, model, uav_traverse):
    env.set_uav_traverse(uav_traverse)
    obs = env.reset()
    while True:
        actions = _select(model, obs)
        obs, rewards, dones, info = env.step(actions)
        if info["terminal"] or info["truncated"]:
            break

    uav_num = env.uav_num
    delta_t = env.delta_t
    times = [float(t) if t is not None else -1.0 for t in info["times"]]
    finished = [t >= 0 for t in times]
    all_done = all(finished)
    makespan = (max(times) * delta_t) if all_done else (env.T * delta_t)
    outage = float(sum(info["outage_duration"]))
    delay = float(sum(sum(ts) for ts in info["tau_history"]))
    rate = float(sum(info["rate_sum"])) / max(float(sum(info["rate_steps"])), 1.0)
    energy = float(sum(info["energy"]))
    return dict(
        all_done=all_done,
        times=times,
        makespan=makespan,
        outage=outage,
        delay=delay,
        rate=rate,
        energy=energy,
    )


def run(uav_num=2, seed=42, algorithms=ALGORITHMS):
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
        m = _rollout_metrics(env, model, traverse)
        print("%-8s %-9s %-8.1f %-8.1f %-8.1f %-8.2f %s"
              % (algo, str(m["all_done"]), m["makespan"], m["outage"],
                 m["delay"], m["rate"], [int(t) for t in m["times"]]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic rollout comparison of MARL algorithms.")
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithms", type=str, nargs="+", default=list(ALGORITHMS))
    args = parser.parse_args(argv)
    return run(uav_num=args.uav_num, seed=args.seed, algorithms=args.algorithms)


if __name__ == "__main__":
    main()
