"""Export per-step UAV trajectories and end-to-end metrics from a sealed
exclusive-MARL run by deterministically replaying a checkpoint.

The replay reuses the exact helpers from ``run_exclusive_marl_pilot``
(including the teacher-action blend used during training evaluation) so the
exported metrics match ``summary.json`` best_metrics.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import run_exclusive_marl_pilot as pilot
from segment.stc.e2e_exclusive import ExclusiveSectorFixedRouteEnv
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario


def _as_list(value):
    return np.asarray(value).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="results/formal_win")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best_constraint.pth")
    parser.add_argument("--analysis-root", default="results/formal_win_analysis")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_root) / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError("run directory not found: %s" % run_dir)
    cfg = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )

    algorithm = cfg["algorithm"]
    uav_num = int(cfg["uav_num"])
    seed = int(cfg["seed"])
    horizon_steps = int(cfg["horizon_steps"])
    arrival_radius_m = float(cfg["arrival_radius_m"])
    safe_distance_m = float(cfg["safe_distance_m"])
    outage_probability_threshold = float(cfg["outage_probability_threshold"])
    outage_max_s = float(cfg["outage_max_s"])
    spacing_m = float(cfg["spacing_m"])
    safety_outage_threshold = float(cfg["safety_outage_threshold"])
    teacher_action_blend = float(cfg["teacher_action_blend"])

    pilot._set_device(pilot._device())

    scenario = load_stc_scenario(uav_num, task_seed=seed, map_seed=seed)
    field = PerSectorRadioField.from_scenario(scenario, spacing_m=spacing_m)
    routes = {int(k): [int(t) for t in v] for k, v in cfg["routes"].items()}
    env = ExclusiveSectorFixedRouteEnv(
        scenario,
        field,
        routes,
        horizon_steps=horizon_steps,
        arrival_radius_m=arrival_radius_m,
        safe_distance_m=safe_distance_m,
        outage_probability_threshold=outage_probability_threshold,
        outage_max_s=outage_max_s,
    )
    model = pilot._make_model(
        algorithm,
        env,
        actor_lr=float(cfg["actor_lr"]),
        critic_lr=float(cfg["critic_lr"]),
        demo_margin=float(cfg["demo_margin"]),
        teacher_anchor_weight=float(cfg["teacher_anchor_weight"]),
    )
    ckpt = run_dir / args.checkpoint
    if not ckpt.is_file():
        raise FileNotFoundError("checkpoint not found: %s" % ckpt)
    model.load(str(ckpt))

    obs = env.reset()
    frames = []
    while True:
        actions = pilot._deterministic_actions(algorithm, model, obs)
        if teacher_action_blend > 0.0 and algorithm in ("SharedTD3", "MAPPO"):
            teacher_actions = env.project_safe_actions(
                pilot._guided_actions(env),
                outage_threshold=safety_outage_threshold,
                update_stats=False,
            )
            actions = pilot._blend_teacher_actions(
                actions, teacher_actions, teacher_action_blend
            )
        actions = env.project_safe_actions(
            actions, outage_threshold=safety_outage_threshold
        )
        obs, rewards, terminal, info = env.step(actions)
        frames.append({
            "t": int(env.t),
            "positions_m": _as_list(env.positions),
            "actions": _as_list(env.last_actions),
            "buffers_bits": _as_list(env.buffers),
            "energy_j": _as_list(env.energy_j),
            "sectors": _as_list(env.sectors),
            "a2g_efficiency": _as_list(env.last_efficiency),
            "g2a_outage": _as_list(env.last_outage),
            "current_contiguous_outage_s": _as_list(
                info["current_contiguous_outage_s"]
            ),
            "uploaded_bits": _as_list(info["uploaded_bits"]),
            "route_index": _as_list(info["route_index"]),
            "done": _as_list(info["done"]),
            "collisions": int(info["collisions"]),
            "out_of_bounds": int(info["out_of_bounds"]),
            "sector_conflicts": int(info["sector_conflicts"]),
            "safety_projections": int(info["safety_projections"]),
            "upload_guard_projections": int(info["upload_guard_projections"]),
            "rewards": [float(r) for r in rewards],
        })
        if terminal or info["truncated"]:
            break

    metrics = env.metrics()

    out_dir = Path(args.analysis_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem

    report = {
        "run_id": args.run_id,
        "algorithm": algorithm,
        "uav_num": uav_num,
        "checkpoint": args.checkpoint,
        "teacher_action_blend": teacher_action_blend,
        "safety_outage_threshold": safety_outage_threshold,
        "horizon_steps": horizon_steps,
        "steps": len(frames),
        "metrics": metrics,
        "frames": frames,
    }
    json_path = out_dir / ("trajectory_%s.json" % stem)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    arrays = {
        "positions_m": np.asarray(
            [f["positions_m"] for f in frames], dtype=float
        ),
        "actions": np.asarray([f["actions"] for f in frames], dtype=float),
        "buffers_bits": np.asarray(
            [f["buffers_bits"] for f in frames], dtype=float
        ),
        "energy_j": np.asarray([f["energy_j"] for f in frames], dtype=float),
        "sectors": np.asarray([f["sectors"] for f in frames], dtype=int),
        "a2g_efficiency": np.asarray(
            [f["a2g_efficiency"] for f in frames], dtype=float
        ),
        "g2a_outage": np.asarray(
            [f["g2a_outage"] for f in frames], dtype=float
        ),
        "current_contiguous_outage_s": np.asarray(
            [f["current_contiguous_outage_s"] for f in frames], dtype=float
        ),
        "uploaded_bits": np.asarray(
            [f["uploaded_bits"] for f in frames], dtype=float
        ),
        "route_index": np.asarray(
            [f["route_index"] for f in frames], dtype=int
        ),
        "done": np.asarray([f["done"] for f in frames], dtype=bool),
        "rewards": np.asarray([f["rewards"] for f in frames], dtype=float),
    }
    npz_path = out_dir / ("trajectory_%s.npz" % stem)
    np.savez(npz_path, **arrays)

    print(
        "exported run=%s checkpoint=%s steps=%d feasible=%s"
        % (args.run_id, args.checkpoint, len(frames), metrics["feasible"])
    )
    print("json=%s" % json_path)
    print("npz=%s" % npz_path)


if __name__ == "__main__":
    main()
