"""Limited-budget GG-MATD3-style end-to-end trajectory baseline.

The reproduction keeps MATD3's centralised twin critic, decentralised actors,
continuous actions, dense reward, distance-aware speed taper, and local
collision repulsion. The UAV count is fixed by the STC scenario, so Pan et al.'s
GA stage is not applicable. Training and evaluation use the current STC task
data and shared-bandwidth radio field.
"""

import argparse
import io
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.stc_common import json_bytes, json_safe, small_scenario
from segment.stc.e2e import (
    SharedBandwidthE2EEnv,
    apply_greedy_speed,
    apply_repulsion,
)
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from trajectory import matd3 as matd3_module
from trajectory import replay as replay_module


EXPECTED = (
    "records.json", "summary.json", "progress.json",
    "training_curve.npz", "best.pth", "last.pth",
)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reference(run_dir):
    path = Path(run_dir) / "raw_records.json"
    records = json_load(path)
    for record in records:
        ref = record.get("reference", {}).get("reference", {})
        if ref.get("t_ref_s") and ref.get("d_ref_s"):
            return float(ref["t_ref_s"]), float(ref["d_ref_s"])
    raise ValueError("no frozen objective reference found in %s" % path)


def json_load(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _score(metrics, reward):
    return (
        int(bool(metrics["feasible"])),
        int(metrics["uploaded_tasks"]),
        int(metrics["visited_tasks"]),
        -float(metrics["makespan_s"]),
        float(reward),
    )


def _save_model(rc, model, path, artifact_name):
    model.save(path)
    rc.write_bytes(artifact_name, Path(path).read_bytes())


def _run_episode(env, model, *, train_noise=None, greedy=True, repulsion=True,
                 update_every=1, replay=None):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        raw = model.select_actions(
            obs,
            add_noise=train_noise is not None,
            noise_scale=(train_noise if train_noise is not None else 0.0))
        actions = np.asarray(raw, dtype=float)
        if greedy:
            actions = apply_greedy_speed(env, actions)
        if repulsion:
            actions = apply_repulsion(env, actions)
        actions = np.clip(actions, env.action_low, env.action_high)

        next_obs, rewards, _dones, info = env.step(actions)
        total_reward += float(np.sum(rewards))
        done = bool(info["terminal"] or info["truncated"])
        if replay is not None:
            replay.add(
                np.concatenate(obs), np.concatenate(actions),
                float(np.sum(rewards)), np.concatenate(next_obs), done)
            if (len(replay) >= model.batch_size
                    and steps % max(int(update_every), 1) == 0):
                model.train(replay)
        obs = next_obs
        steps += 1
        if done:
            break
    return total_reward, env.metrics(), info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-num", type=int, default=2)
    parser.add_argument("--map-seed", type=int, default=42)
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--spacing-m", type=float, default=200.0)
    parser.add_argument("--horizon-steps", type=int, default=1000)
    parser.add_argument("--time-limit-s", type=float, default=None)
    parser.add_argument("--reference-run", default="results/runs/formal-u2-s42")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tasks", type=int, default=6)
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.eval_interval <= 0:
        parser.error("--eval-interval must be positive")

    device = _device()
    matd3_module.device = device
    replay_module.device = device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenario = load_stc_scenario(
        args.uav_num, task_seed=args.task_seed, map_seed=args.map_seed)
    if args.smoke:
        scenario = small_scenario(scenario, args.smoke_tasks)
    field = PerSectorRadioField.from_scenario(
        scenario, spacing_m=args.spacing_m)
    env = SharedBandwidthE2EEnv(
        scenario, field, horizon_steps=args.horizon_steps)
    reference_t, reference_d = _reference(args.reference_run)

    max_action = np.asarray(env.action_high, dtype=np.float32)
    min_action = np.asarray(env.action_low, dtype=np.float32)
    model = matd3_module.MATD3(
        env.n_uav, env.local_obs_dim, env.action_dim,
        max_action, min_action, a_lr=3e-4, c_lr=3e-4,
        gamma=0.95, tau=0.01)
    replay = replay_module.ReplayBuffer(
        model.state_dim, model.joint_action_dim, max_size=200_000)

    cfg = {
        "uav_num": int(scenario.n_uav),
        "n_tasks": int(scenario.n_tasks),
        "map_seed": int(args.map_seed),
        "task_seed": int(args.task_seed),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "update_every": int(args.update_every),
        "eval_interval": int(args.eval_interval),
        "spacing_m": float(args.spacing_m),
        "horizon_steps": int(args.horizon_steps),
        "time_limit_s": args.time_limit_s,
        "reference_run": str(Path(args.reference_run).resolve()),
        "t_ref_s": reference_t,
        "d_ref_s": reference_d,
        "device": str(device),
        "algorithm": "GG-MATD3-style limited-budget reproduction",
        "ga_uav_count": "not applicable: U fixed by scenario",
        "greedy_speed": True,
        "collision_repulsion": True,
    }
    rc = RunContext.create(
        "results/runs", "stc_gg_matd3_e2e", cfg,
        run_id=args.run_id, expected=EXPECTED)
    rc.register_input("resolved_config", data=json_bytes(cfg))
    rc.write_bytes("resolved_config.json", json_bytes(cfg))

    train_rewards = []
    train_steps = []
    train_visited = []
    train_uploaded = []
    eval_records = []
    best_score = None
    best_metrics = None
    best_reward = -float("inf")
    best_path = rc.run_dir / "best.pth"
    last_path = rc.run_dir / "last.pth"
    start = time.perf_counter()
    completed_episodes = 0
    training_stopped_by_time = False

    def write_progress():
        rc.write_bytes("progress.json", json_bytes({
            "completed_episodes": completed_episodes,
            "target_episodes": args.episodes,
            "best_score": list(best_score) if best_score else None,
            "best_metrics": best_metrics,
            "evaluations": eval_records,
        }))

    try:
        for ep in range(args.episodes):
            if (args.time_limit_s is not None
                    and time.perf_counter() - start > args.time_limit_s):
                training_stopped_by_time = True
                break
            frac = ep / max(args.episodes - 1, 1)
            noise = 0.3 * ((0.02 / 0.3) ** frac)
            reward, metrics, info = _run_episode(
                env, model, train_noise=noise,
                update_every=args.update_every, replay=replay)
            completed_episodes = ep + 1
            train_rewards.append(float(reward))
            train_steps.append(int(info["t"]))
            train_visited.append(int(metrics["visited_tasks"]))
            train_uploaded.append(int(metrics["uploaded_tasks"]))
            if (completed_episodes == 1
                    or completed_episodes % args.eval_interval == 0):
                eval_reward, eval_metrics, _ = _run_episode(
                    env, model, train_noise=None)
                score = _score(eval_metrics, eval_reward)
                eval_records.append({
                    "episode": completed_episodes,
                    "reward": float(eval_reward),
                    "metrics": eval_metrics,
                })
                if best_score is None or score > best_score:
                    best_score = score
                    best_metrics = eval_metrics
                    best_reward = float(eval_reward)
                    _save_model(rc, model, best_path, "best.pth")
                print(
                    "[GG-MATD3] ep=%d train_reward=%.2f "
                    "visited=%d/%d uploaded=%d/%d eval_feasible=%s"
                    % (completed_episodes, reward,
                       metrics["visited_tasks"], scenario.n_tasks,
                       metrics["uploaded_tasks"], scenario.n_tasks,
                       eval_metrics["feasible"]), flush=True)
                write_progress()

        _save_model(rc, model, last_path, "last.pth")
        if best_score is None:
            _save_model(rc, model, best_path, "best.pth")
            best_metrics = env.metrics()
        else:
            model.load(best_path)
        final_reward, final_metrics, _ = _run_episode(
            env, model, train_noise=None)
        feasible = bool(final_metrics["feasible"])
        objective = (
            0.5 * final_metrics["makespan_s"] / reference_t
            + 0.5 * final_metrics["total_delay_s"] / reference_d
            if feasible else None)
        record = {
            "method": "gg_matd3_e2e_style",
            "label": "GG-MATD3-style end-to-end (limited budget)",
            "status": "evaluated",
            "feasible": feasible,
            "objective": objective,
            "makespan_s": final_metrics["makespan_s"],
            "total_delay_s": final_metrics["total_delay_s"],
            "max_outage_ratio": final_metrics["max_outage_ratio"],
            "visited_tasks": final_metrics["visited_tasks"],
            "uploaded_tasks": final_metrics["uploaded_tasks"],
            "collisions": final_metrics["collisions"],
            "out_of_bounds": final_metrics["out_of_bounds"],
            "steps": final_metrics["steps"],
            "eval_reward": float(final_reward),
            "best_eval_reward": float(best_reward),
            "completed_episodes": completed_episodes,
            "target_episodes": int(args.episodes),
            "training_stopped_by_time": training_stopped_by_time,
            "training_runtime_s": time.perf_counter() - start,
            "greedy_speed": True,
            "collision_repulsion": True,
            "ga_uav_count": "not applicable",
            "n_uav": scenario.n_uav,
            "n_tasks": scenario.n_tasks,
            "map_seed": scenario.map_seed,
            "task_seed": scenario.task_seed,
            "method_seed": args.seed,
            "reproduction_note": (
                "Style reproduction; UAV count is fixed by STC scenario and "
                "training is evaluated under the shared-bandwidth task model."),
        }
        summary = {
            "record_count": 1,
            "completed_episodes": completed_episodes,
            "target_episodes": int(args.episodes),
            "feasible": feasible,
            "best_eval": best_metrics,
        }
        rc.write_bytes("records.json", json_bytes([record]))
        rc.write_bytes("summary.json", json_bytes(summary))
        curve_buf = io.BytesIO()
        np.savez(
            curve_buf,
            rewards=np.asarray(train_rewards, dtype=np.float64),
            steps=np.asarray(train_steps, dtype=np.int64),
            visited=np.asarray(train_visited, dtype=np.int64),
            uploaded=np.asarray(train_uploaded, dtype=np.int64),
        )
        rc.write_bytes("training_curve.npz", curve_buf.getvalue())
        write_progress()
        rc.finalize("success")
    except BaseException:
        try:
            model.save(last_path)
        except Exception:
            pass
        rc.finalize("failed")
        raise
    print("run written to %s" % rc.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
