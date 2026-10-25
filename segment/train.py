"""Train the single-segment goal/resource-conditioned TD3 (new design).

Segments are extracted from the layer-1/2 assignment (4D clustering + GA_EQTSP)
as (start, goal, payload) triples — a placeholder source that the ALNS outer
layer will later replace; the TD3 itself trains exactly per the new design.

CLI::

    python segment/train.py --uav_num 2 --total_episode 2000
    python segment/train.py --uav_num 2 --plot-only        # reuse a saved model
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common
from experiment.run_traj import fixed_traverse
from trajectory.replay import ReplayBuffer
from segment.env import SingleSegmentEnv, CommField
from segment.td3 import TD3
from segment.ddpg import DDPG
from segment.sac import SAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "models", "segment")
LOG_DIR = os.path.join(ROOT, "results", "logs")
FIG_DIR = os.path.join(ROOT, "results", "figs")


def extract_segments(cfg, points, data_sizes, traverse):
    """Turn each UAV's route into (start, goal, payload) segment triples.

    Per the new design, the segment ``A -> B`` carries *A*'s data (the source
    point's data, uploaded before arriving at B); the first segment (depot ->
    first point) carries nothing, the last (last point -> end) carries the last
    point's data.
    """
    ini = np.asarray(cfg["ini_loc"], dtype=float)
    end = np.asarray(cfg["end_loc"], dtype=float)

    segments = []
    for i in range(int(cfg["uav_num"])):
        gidx = traverse.get(i, [])
        if not gidx:
            continue
        wps = [ini] + [np.asarray(points[g], dtype=float) for g in gidx] + [end]
        datas = [0.0] + [float(data_sizes[g]) for g in gidx]
        for r in range(len(wps) - 1):
            segments.append((wps[r], wps[r + 1], datas[r]))
    return segments


def all_pairs_segments(cfg, points, data_sizes):
    """All *directed* segments as (start, goal, payload) triples.

    The training pool for the general goal/resource-conditioned policy: every
    ordered pair of points (``A -> B`` carries A's data, so ``A->B`` != ``B->A``)
    plus depot->point (payload 0) and point->end (payload = source data).
    Total = K + K*(K-1) + K = K*(K+1) directed edges.
    """
    ini = np.asarray(cfg["ini_loc"], dtype=float)
    end = np.asarray(cfg["end_loc"], dtype=float)
    pts = [np.asarray(p, dtype=float) for p in points]
    K = len(pts)

    segments = []
    for j in range(K):                       # depot -> point (nothing uploaded yet)
        segments.append((ini, pts[j], 0.0))
    for i in range(K):                       # point -> point (source i's data)
        for j in range(K):
            if i != j:
                segments.append((pts[i], pts[j], float(data_sizes[i])))
    for i in range(K):                       # point -> end (source i's data)
        segments.append((pts[i], end, float(data_sizes[i])))
    return segments


def _deterministic_rollout(env, model, min_action, max_action):
    """Return (total_reward, feasible) for a no-noise rollout."""
    env.reset()
    total = 0.0
    while True:
        a = np.clip(model.select_action(env._state()), min_action, max_action)
        s_, r, done, info = env.step(a)
        total += r
        if done:
            break
    return total, bool(info["feasible"])


def train(uav_num=2, total_episode=2000, seed=42, log_every=50, eval_interval=50,
          pool="all_pairs", c_lr=1e-3, policy_delay=2, model_tag="",
          held_out_frac=0.2, algo="TD3"):
    cfg, points, data_sizes, traverse = fixed_traverse(uav_num, seed)
    if pool == "route":
        segments = extract_segments(cfg, points, data_sizes, traverse)
    else:
        segments = all_pairs_segments(cfg, points, data_sizes)
    if not segments:
        raise RuntimeError("no segments extracted")
    print("training pool: %d directed edges (%s)" % (len(segments), pool))

    # Held-out split (generalization test): hold out a fixed fraction of edges,
    # train only on the rest, then evaluate on the held-out set.
    if held_out_frac > 0 and pool == "all_pairs":
        rng_split = np.random.RandomState(seed + 12345)
        n_held = int(len(segments) * held_out_frac)
        held_idx = set(rng_split.choice(len(segments), n_held, replace=False).tolist())
        train_segments = [s for i, s in enumerate(segments) if i not in held_idx]
        held_segments = [s for i, s in enumerate(segments) if i in held_idx]
        print("held-out split: %d train / %d held"
              % (len(train_segments), len(held_segments)))
    else:
        train_segments = segments
        held_segments = []

    print("precomputing communication field ...")
    field = CommField(cfg)

    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = SingleSegmentEnv(cfg, segments[0][0], segments[0][1],
                                 segments[0][2]).state_dim
    action_dim = 2
    v_max = float(cfg["v_max"])
    max_action = np.array([v_max, 2.0 * math.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    noise_scale = np.array([v_max, 1.0], dtype=np.float32)

    if algo == "TD3":
        model = TD3(state_dim, action_dim, max_action, min_action,
                    c_lr=c_lr, policy_delay=policy_delay)
    elif algo == "DDPG":
        model = DDPG(state_dim, action_dim, max_action, min_action)
    elif algo == "SAC":
        model = SAC(state_dim, action_dim, max_action, min_action)
    else:
        raise ValueError("unknown algo: %s" % algo)
    replay = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    batch_size = model.batch_size

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    suffix = ("_" + model_tag) if model_tag else ""
    best_path = os.path.join(MODEL_DIR, "uav%d%s_best.pth" % (uav_num, suffix))
    last_path = os.path.join(MODEL_DIR, "uav%d%s_last.pth" % (uav_num, suffix))

    expl_start, expl_end = 0.3, 0.02
    expl_decay = (expl_end / expl_start) ** (1.0 / max(int(total_episode), 1))
    best_score = -float("inf")
    ep_rewards = []
    ep_feasible = []

    rng = np.random.RandomState(seed)
    # Deterministic-eval probe: evenly spaced across the pool (mixes easy
    # payload-0 depot edges, hard point->point edges, and point->end edges).
    probe_idx = np.linspace(0, len(train_segments) - 1, 6).astype(int)
    probe = [train_segments[i] for i in probe_idx]

    for ep in range(int(total_episode)):
        start, goal, data = train_segments[rng.randint(len(train_segments))]
        env = SingleSegmentEnv(cfg, start, goal, data, field=field)
        s = env.reset()
        ep_reward = 0.0
        expl = expl_start * (expl_decay ** ep)
        while True:
            if algo == "SAC":
                a = model.sample_action(s)
            else:
                a = model.select_action(s) + np.random.normal(0.0, expl, size=2) * noise_scale
                a = np.clip(a, min_action, max_action)
            s_, r, done, info = env.step(a)
            replay.add(s, a, r, s_, float(done))
            if len(replay) >= batch_size:
                model.train(replay)
            ep_reward += r
            s = s_
            if done:
                break

        ep_rewards.append(ep_reward)
        ep_feasible.append(bool(info["feasible"]))

        if (ep + 1) % eval_interval == 0 or ep == 0:
            # Score = deterministic reward, feasible-only (never pick an
            # infeasible policy as best).
            scores = []
            for (ps, pg, pd) in probe:
                e = SingleSegmentEnv(cfg, ps, pg, pd, field=field)
                rew, feas = _deterministic_rollout(e, model, min_action, max_action)
                scores.append(rew if feas else -float("inf"))
            score = float(np.mean(scores)) if any(s > -float("inf") for s in scores) else -float("inf")
            if score > best_score:
                best_score = score
                model.save(best_path)
                print("  [eval] ep %4d | det_score %10.2f (new best)" % (ep + 1, score))

        if (ep + 1) % log_every == 0 or ep == 0:
            print("ep %4d | reward %10.2f | avg100 %10.2f | expl %.3f"
                  % (ep + 1, ep_reward, float(np.mean(ep_rewards[-100:])), expl))

    model.save(last_path)
    np.savez(os.path.join(LOG_DIR, "UAV_%d_%s_rewards.npz" % (uav_num, algo)),
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64),
             ep_feasible=np.asarray(ep_feasible, dtype=bool))
    print("=== done === best_score=%.2f avg_last100=%.2f feasible_rate=%.1f%%" %
          (best_score, float(np.mean(ep_rewards[-100:])),
           100.0 * float(np.mean(ep_feasible))))

    # Held-out generalization evaluation (use the eval-selected best model).
    if held_segments:
        model.load(best_path)
        feas_count = 0
        for (ps, pg, pd) in held_segments:
            e = SingleSegmentEnv(cfg, ps, pg, pd, field=field)
            _, feas = _deterministic_rollout(e, model, min_action, max_action)
            feas_count += bool(feas)
        print("held-out feasibility (best model): %d/%d = %.1f%%"
              % (feas_count, len(held_segments), 100.0 * feas_count / len(held_segments)))
    return model, cfg, segments


def rollout_path(env, model, min_action, max_action):
    """Deterministic rollout -> (positions, feasible)."""
    env.reset()
    xs, ys = [env.pos[0]], [env.pos[1]]
    while True:
        a = np.clip(model.select_action(env._state()), min_action, max_action)
        s_, r, done, info = env.step(a)
        xs.append(env.pos[0])
        ys.append(env.pos[1])
        if done:
            break
    return np.asarray(xs), np.asarray(ys), bool(info["feasible"])


def plot(uav_num=2, seed=42):
    """Plot UAV 0's single-segment trajectories over the G2A outage + A2G SINR
    maps (same fields/rendering as the old ``plot_results`` trajectory figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from radio_map import radio_map_A2G, radio_map_G2A
    from trajectory.env import DEFAULT_UAV_H

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    UNIT_TO_KM = 0.1
    UAV_H_KM = DEFAULT_UAV_H * UNIT_TO_KM
    GRID_RES = 60

    cfg, points, data_sizes, traverse = fixed_traverse(uav_num, seed)
    segments = extract_segments(cfg, points, data_sizes, traverse)

    state_dim = SingleSegmentEnv(cfg, segments[0][0], segments[0][1],
                                 segments[0][2]).state_dim
    v_max = float(cfg["v_max"])
    max_action = np.array([v_max, 2.0 * math.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)

    model = TD3(state_dim, 2, max_action, min_action)
    model.load(os.path.join(MODEL_DIR, "uav%d_best.pth" % uav_num))

    # Query the G2A outage and A2G SINR fields (km grid, batch API).
    extent_km = float(cfg["length"]) * UNIT_TO_KM
    xs = np.linspace(0.0, extent_km, GRID_RES)
    ys = np.linspace(0.0, extent_km, GRID_RES)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, UAV_H_KM)], axis=1)
    _, g2a_vals = radio_map_G2A.getPointMiniOutage(locs)
    _, a2g_vals = radio_map_A2G.getPointMiniOutage(locs)
    g2a_grid = np.asarray(g2a_vals, dtype=float).reshape(GRID_RES, GRID_RES)
    a2g_grid = np.asarray(a2g_vals, dtype=float).reshape(GRID_RES, GRID_RES)

    # UAV 0's route (depot -> points -> end).
    gidx = traverse.get(0, [])
    ini = np.asarray(cfg["ini_loc"], dtype=float)
    end = np.asarray(cfg["end_loc"], dtype=float)
    wps = [ini] + [np.asarray(points[g], dtype=float) for g in gidx] + [end]
    datas = [0.0] + [float(data_sizes[g]) for g in gidx]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(wps) - 1, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left: G2A outage + trajectories.
    ax = axes[0]
    cf = ax.contourf(xs, ys, g2a_grid, levels=np.linspace(0.0, 1.0, 11), cmap="Reds")
    fig.colorbar(cf, ax=ax, label="G2A 中断概率")
    # Right: A2G SINR + trajectories.
    ax = axes[1]
    cf = ax.contourf(xs, ys, a2g_grid, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="A2G SINR (dB)")

    for ax in axes:
        for r in range(len(wps) - 1):
            env = SingleSegmentEnv(cfg, wps[r], wps[r + 1], datas[r])
            px, py, feas = rollout_path(env, model, min_action, max_action)
            ax.plot(px * UNIT_TO_KM, py * UNIT_TO_KM, color=colors[r], lw=1.6,
                    label=("seg %d" % r) if r < len(wps) - 2 else "last")
        for g in gidx:
            ax.plot(points[g][0] * UNIT_TO_KM, points[g][1] * UNIT_TO_KM,
                    "ko", ms=4, zorder=5)
        ax.plot(ini[0] * UNIT_TO_KM, ini[1] * UNIT_TO_KM, "r^", ms=10, label="start")
        ax.plot(end[0] * UNIT_TO_KM, end[1] * UNIT_TO_KM, "rs", ms=10, label="end")
        ax.set_xlim(0, extent_km)
        ax.set_ylim(0, extent_km)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

    axes[0].set_title("G2A 中断图 + 单段轨迹 (UAV 0)")
    axes[1].set_title("A2G SINR 图 + 单段轨迹 (UAV 0)")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()

    out = os.path.join(FIG_DIR, "segment_trajectory_uav%d.png" % uav_num)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved -> %s" % out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--total_episode", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", type=str, default="all_pairs",
                        choices=["all_pairs", "route"])
    parser.add_argument("--c_lr", type=float, default=1e-3)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--model_tag", type=str, default="")
    parser.add_argument("--held_out_frac", type=float, default=0.2)
    parser.add_argument("--algo", type=str, default="TD3",
                        choices=["TD3", "DDPG", "SAC"])
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args(argv)

    if args.plot_only:
        plot(uav_num=args.uav_num, seed=args.seed)
    else:
        model, cfg, segments = train(uav_num=args.uav_num,
                                     total_episode=args.total_episode,
                                     seed=args.seed, pool=args.pool,
                                     c_lr=args.c_lr,
                                     policy_delay=args.policy_delay,
                                     model_tag=args.model_tag,
                                     held_out_frac=args.held_out_frac,
                                     algo=args.algo)
        if not args.no_plot:
            plot(uav_num=args.uav_num, seed=args.seed)
    return 0


if __name__ == "__main__":
    main()
