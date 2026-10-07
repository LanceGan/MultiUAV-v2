"""Layer-3 SCA baseline: communication-aware trajectory via SCA (SQP).

For each UAV, plans a comm-optimised route (deviating from straight lines to fly
through high-rate / low-outage regions) over the same 4D-clustered + ga_eqtsp
assignment as the greedy baseline and the DRL, then replays it through the env
to obtain the same five end-to-end metrics (makespan / total_delay /
avg_uplink_rate / outage_duration / total_energy).

Writes ``results/logs/UAV_<N>_SCA.npz`` + ``.metrics.json`` (plot-compatible)
and ``results/logs/traj_sca_uav<N>.json``.

CLI::

    python experiment/run_sca.py --uav_nums 2    # smoke
    python experiment/run_sca.py                 # full grid
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiment import _common, evaluate
from experiment.run_traj import fixed_traverse
from trajectory import test as traj_test
from trajectory.env import MultiUAVInspectionEnv
from trajectory.sca import ChannelField, SCAPolicy, plan_route


def sca_rollout(env, policies, uav_traverse):
    """Deterministic rollout with one policy per UAV (mirrors ``traj_test.rollout``)."""
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
        actions = [np.clip(policies[i].select_action(obs[i]), min_action, max_action)
                   for i in range(uav_num)]
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


def run(uav_nums=(2, 3, 4), seed=42, log_dir=None, m=5, n_restarts=3,
        use_a2g=False):
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    results = []
    for uav_num in uav_nums:
        cfg, points, data_sizes, traverse = fixed_traverse(uav_num, seed)

        field = ChannelField(cfg)

        env = MultiUAVInspectionEnv(cfg, safe_distance=0.1)
        policies = []
        for i in range(uav_num):
            gidx = traverse.get(i, [])
            # Waypoints = ini + inspection points in visit order (no end: the
            # env stops after the last point).  Segment k carries the data of
            # its *destination* point (env "pre-load en route" semantics).
            waypoints = [np.asarray(cfg["ini_loc"], dtype=float)]
            waypoints += [np.asarray(points[g], dtype=float) for g in gidx]
            data_per_segment = [float(data_sizes[g]) for g in gidx]
            if len(gidx) == 0:
                policies.append(None)
                continue
            path = plan_route(field, waypoints, data_per_segment, cfg,
                              m=m, n_restarts=n_restarts, seed=seed + i,
                              use_a2g=use_a2g)
            policies.append(SCAPolicy(path, cfg["length"], cfg["width"],
                                      cfg["v_max"]))

        # Replay through the env (per-UAV policies; done UAVs use a dummy).
        dummy = SCAPolicy(np.asarray([cfg["ini_loc"]], dtype=float),
                          cfg["length"], cfg["width"], cfg["v_max"])
        log = sca_rollout(env, [p if p is not None else dummy for p in policies],
                          traverse)

        run_id = "UAV_%d_SCA" % uav_num
        npz_path = os.path.join(log_dir, "%s.npz" % run_id)
        traj_test.save_log(npz_path, log, uav_num, cfg)

        metrics = evaluate.evaluate_log(np.load(npz_path))
        metrics.update({"run_id": run_id, "uav_num": uav_num,
                        "method": "SCA", "seed": int(cfg["seed"])})
        _common.save_json(os.path.join(log_dir, "%s.metrics.json" % run_id),
                          metrics)
        _common.save_json(os.path.join(log_dir, "traj_sca_uav%d.json" % uav_num),
                          dict(metrics, run_id="traj_sca_uav%d" % uav_num))
        print("saved rollout -> %s" % npz_path)
        results.append(metrics)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-3 SCA trajectory baseline.")
    parser.add_argument("--uav_nums", type=str, default="2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n_restarts", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--use_a2g", action="store_true",
                        help="use the A2G uplink map (dual-map tier); default is G2A-only")
    args = parser.parse_args(argv)
    uav_nums = tuple(int(x) for x in args.uav_nums.split(",") if x.strip())
    return run(uav_nums=uav_nums, seed=args.seed, log_dir=args.log_dir,
               m=args.m, n_restarts=args.n_restarts, use_a2g=args.use_a2g)


if __name__ == "__main__":
    main()
