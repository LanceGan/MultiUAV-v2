"""Short-budget MADRL integration pilot on the exclusive-sector environment.

This runner is deliberately scoped to one fixed scenario (U=2/s42) and a short
budget.  It validates that SharedTD3, MADDPG, MATD3, and MAPPO can train,
checkpoint, reload, and evaluate against the accepted environment.  It is not a
formal performance experiment.
"""

import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import _common
from segment.stc.e2e_exclusive import ExclusiveSectorFixedRouteEnv
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from sequence import ga_eqtsp
from trajectory import maddpg, ma_d3qn, matd3, mappo, replay as replay_module, td3
from trajectory.replay import ReplayBuffer
from trajectory.td3 import SharedTD3
from trajectory.td3_bc import DemoReplayBuffer, SharedTD3BC, linear_decay


ALGORITHMS = ("SharedTD3", "MADDPG", "MATD3", "MAPPO", "MA-D3QN")
EXPECTED = (
    "summary.json",
    "records.json",
    "training_curve.npz",
    "best.pth",
    "best_constraint.pth",
    "best_completion.pth",
    "last.pth",
)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_device(device):
    td3.device = device
    maddpg.device = device
    matd3.device = device
    mappo.device = device
    ma_d3qn.device = device
    replay_module.device = device


def _fixed_routes(uav_num, seed):
    cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
    labels = _common.cluster_labels("4d", points, data_sizes, uav_num, seed)
    clusters = _common.split_clusters(labels, points, data_sizes, uav_num)
    orders = []
    for cluster_points, _global_index, cluster_data in clusters:
        _, order, _ = ga_eqtsp.ga_eqtsp(
            cluster_points, cfg["ini_loc"], cfg["end_loc"], cluster_data, cfg
        )
        orders.append(order)
    traverse = _common.build_traverse(clusters, orders)
    return {u: list(traverse[u]) for u in range(uav_num)}


def _load_routes(path, uav_num):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    routes = payload.get("routes", payload)
    routes = {int(k): [int(t) for t in v] for k, v in routes.items()}
    if sorted(routes) != list(range(uav_num)):
        raise ValueError("route cache does not contain every UAV")
    return routes


def _make_model(
    algorithm, env, actor_lr=1e-4, critic_lr=1e-3, demo_margin=0.8,
    teacher_anchor_weight=0.0,
):
    max_action = np.asarray(env.action_high, dtype=np.float32)
    min_action = np.asarray(env.action_low, dtype=np.float32)
    if algorithm == "SharedTD3":
        return SharedTD3BC(
            env.local_obs_dim, env.action_dim, max_action, min_action,
            a_lr=float(actor_lr), c_lr=float(critic_lr),
            teacher_anchor_weight=teacher_anchor_weight,
        )
    if algorithm == "MADDPG":
        return maddpg.MADDPG(
            env.n_uav, env.local_obs_dim, env.action_dim,
            max_action, min_action,
            a_lr=float(actor_lr), c_lr=float(critic_lr),
        )
    if algorithm == "MATD3":
        return matd3.MATD3(
            env.n_uav, env.local_obs_dim, env.action_dim,
            max_action, min_action,
            a_lr=float(actor_lr), c_lr=float(critic_lr),
        )
    if algorithm == "MAPPO":
        return mappo.MAPPO(
            env.n_uav, env.local_obs_dim, env.action_dim,
            max_action, min_action, rollout_steps=env.T,
            a_lr=float(actor_lr), c_lr=float(critic_lr),
            teacher_weight=teacher_anchor_weight,
        )
    if algorithm == "MA-D3QN":
        codec = ma_d3qn.DiscreteActionCodec(env.v_max)
        model = ma_d3qn.MAD3QNAgent(
            env.local_obs_dim, len(codec), learning_rate=actor_lr,
            margin=demo_margin,
        )
        model.codec = codec
        return model
    raise ValueError("unknown algorithm: %s" % algorithm)


def _select_actions(algorithm, model, obs, exploration_scale):
    if algorithm == "MA-D3QN":
        actions = []
        for local_obs in obs:
            index = model.select_action(
                local_obs, epsilon=float(exploration_scale)
            )
            actions.append(model.codec.decode(index))
        return actions
    if algorithm == "SharedTD3":
        actions = []
        for local_obs in obs:
            action = model.select_action(local_obs)
            noise = np.random.normal(0.0, 1.0, size=model.action_dim) * np.asarray(
                [float(model.max_action[0]), 1.0], dtype=np.float32
            )
            action = np.clip(
                action + exploration_scale * noise,
                model.min_action.cpu().numpy(),
                model.max_action.cpu().numpy(),
            )
            actions.append(action)
        return actions
    if algorithm == "MAPPO":
        actions, log_probs, value = model.act(obs, add_noise=True)
        return actions, log_probs, value
    return model.select_actions(
        obs,
        add_noise=True,
        noise_scale=exploration_scale * np.asarray(
            [float(model.max_action_np[0]), 1.0], dtype=np.float32
        ),
    )


def _guided_actions(env):
    return [env.guide_action(i) for i in range(env.n_uav)]


def _blend_teacher_actions(policy_actions, teacher_actions, teacher_blend):
    """Blend policy and safe-teacher actions in ``[speed, relative_heading]``."""
    alpha = float(teacher_blend)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("teacher_blend must be in [0, 1]")
    blended = []
    for policy, teacher in zip(policy_actions, teacher_actions):
        policy = np.asarray(policy, dtype=float).reshape(2)
        teacher = np.asarray(teacher, dtype=float).reshape(2)
        speed = (1.0 - alpha) * policy[0] + alpha * teacher[0]
        heading = (1.0 - alpha) * policy[1] + alpha * teacher[1]
        heading = (heading + np.pi) % (2.0 * np.pi) - np.pi
        blended.append(np.asarray([speed, heading], dtype=float))
    return blended


def _mappo_guide_step(model, obs, actions):
    """Evaluate fixed MapGuide actions under the current MAPPO behaviour policy."""
    global_state = torch.tensor(
        np.concatenate(obs), dtype=torch.float32, device=mappo.device
    ).reshape(1, -1)
    with torch.no_grad():
        value = float(model.critic(global_state).item())
        log_probs = []
        for i in range(model.n_agents):
            local_state = torch.tensor(
                np.asarray(obs[i], dtype=np.float32),
                dtype=torch.float32,
                device=mappo.device,
            ).reshape(1, -1)
            mean, std = model.actors[i](local_state)
            action = torch.tensor(
                np.asarray(actions[i], dtype=np.float32),
                dtype=torch.float32,
                device=mappo.device,
            ).reshape(1, -1)
            log_probs.append(float(Normal(mean, std).log_prob(action).sum(-1).item()))
    return log_probs, value


def _deterministic_actions(algorithm, model, obs):
    if algorithm == "MA-D3QN":
        return [
            model.codec.decode(model.select_action(local_obs, epsilon=0.0))
            for local_obs in obs
        ]
    if algorithm == "SharedTD3":
        return [model.select_action(local_obs) for local_obs in obs]
    return model.select_actions(obs, add_noise=False)


def _remaining_distance(env):
    return float(sum(
        np.linalg.norm(env.positions[i] - env._target_position(i))
        for i in range(env.n_uav)
        if not env.done[i]
    ))


def _evaluate(algorithm, model, env, safety_outage_threshold,
              teacher_action_blend=0.0):
    obs = env.reset()
    total_reward = 0.0
    while True:
        actions = _deterministic_actions(algorithm, model, obs)
        if teacher_action_blend > 0.0 and algorithm in ("SharedTD3", "MAPPO"):
            teacher_actions = env.project_safe_actions(
                _guided_actions(env),
                outage_threshold=safety_outage_threshold,
                update_stats=False,
            )
            actions = _blend_teacher_actions(
                actions, teacher_actions, teacher_action_blend
            )
        actions = env.project_safe_actions(
            actions, outage_threshold=safety_outage_threshold
        )
        obs, rewards, terminal, info = env.step(actions)
        total_reward += float(np.sum(rewards))
        if terminal or info["truncated"]:
            break
    metrics = env.metrics()
    metrics["return"] = total_reward
    metrics["remaining_distance_m"] = _remaining_distance(env)
    return metrics


def _score(metrics):
    outage = np.asarray(metrics["max_contiguous_outage_s"], dtype=float)
    outage_excess = np.maximum(outage - float(metrics["outage_max_s"]), 0.0)
    completion = np.asarray(metrics["completion_step"], dtype=float)
    return (
        int(bool(metrics["feasible"])),
        int(metrics["visited_tasks"]),
        int(metrics["uploaded_tasks"]),
        -int(np.sum(completion < 0.0)),
        -int(np.sum(outage_excess > 0.0)),
        -float(np.sum(outage_excess)),
        -float(np.max(outage, initial=0.0)),
        -float(np.sum(outage)),
        -int(metrics["out_of_bounds"]),
        -int(metrics["collisions"]),
        -int(
            metrics["early_service_arrivals"]
            + metrics["early_terminal_arrivals"]
        ),
        -float(metrics["makespan_s"]),
        -float(metrics["total_delay_s"]),
        -float(metrics["remaining_distance_m"]),
        float(metrics["return"]),
    )


def _completion_score(metrics):
    outage = np.asarray(metrics["max_contiguous_outage_s"], dtype=float)
    outage_excess = np.maximum(outage - float(metrics["outage_max_s"]), 0.0)
    completion = np.asarray(metrics["completion_step"], dtype=float)
    return (
        int(bool(metrics["feasible"])),
        int(metrics["uploaded_tasks"]),
        int(metrics["visited_tasks"]),
        -int(np.sum(completion < 0.0)),
        -int(np.sum(outage_excess > 0.0)),
        -float(np.sum(outage_excess)),
        -float(np.max(outage, initial=0.0)),
        -int(metrics["out_of_bounds"]),
        -int(metrics["collisions"]),
        -int(
            metrics["early_service_arrivals"]
            + metrics["early_terminal_arrivals"]
        ),
        -float(metrics["remaining_distance_m"]),
        float(metrics["return"]),
    )


def _train_off_policy(algorithm, model, env, episodes, evaluation_interval,
                      best_path, best_completion_path, last_path,
                      guide_prob, guide_min_prob,
                      bc_weight, bc_min_weight, bc_decay_episodes,
                      demo_fraction, safety_outage_threshold,
                      teacher_anchor_weight, teacher_action_blend):
    if algorithm == "SharedTD3":
        replay = DemoReplayBuffer(
            env.local_obs_dim, env.action_dim, max_size=100_000,
            demo_fraction=demo_fraction,
        )
    else:
        replay = ReplayBuffer(
            model.state_dim if algorithm == "MATD3" else model.global_state_dim,
            (model.joint_action_dim
             if algorithm == "MATD3" else model.global_action_dim),
            max_size=100_000,
        )

    rewards_history = []
    visited_history = []
    uploaded_history = []
    step_history = []
    guided_history = []
    bc_weight_history = []
    term_history = {}
    eval_records = []
    best_score = None
    best_metrics = None
    best_completion_score = None
    best_completion_metrics = None
    start = time.perf_counter()

    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0.0
        steps = 0
        fraction = episode / max(episodes - 1, 1)
        exploration = 0.3 * ((0.02 / 0.3) ** fraction)
        episode_guide_probability = (
            guide_prob + (guide_min_prob - guide_prob) * fraction
        )
        guided_run = bool(np.random.rand() < episode_guide_probability)
        if algorithm == "SharedTD3":
            model.bc_weight = linear_decay(
                bc_weight, episode, bc_decay_episodes,
                floor=bc_min_weight,
            )
            bc_weight_history.append(float(model.bc_weight))

        while True:
            teacher_actions = env.project_safe_actions(
                _guided_actions(env),
                outage_threshold=safety_outage_threshold,
                update_stats=False,
            )
            actions = (
                [np.asarray(action, dtype=float).copy()
                 for action in teacher_actions]
                if guided_run
                else _select_actions(algorithm, model, obs, exploration)
            )
            if not guided_run and teacher_action_blend > 0.0 and (
                    algorithm == "SharedTD3"):
                actions = _blend_teacher_actions(
                    actions, teacher_actions, teacher_action_blend
                )
            actions = env.project_safe_actions(
                actions, outage_threshold=safety_outage_threshold
            )
            next_obs, rewards, terminal, info = env.step(actions)
            done = bool(terminal or info["truncated"])
            team_reward = float(np.sum(rewards))

            if algorithm == "SharedTD3":
                for uav_id in range(env.n_uav):
                    replay.add(
                        obs[uav_id], actions[uav_id], rewards[uav_id],
                        next_obs[uav_id], done, demo=guided_run,
                        teacher_action=teacher_actions[uav_id],
                    )
            else:
                replay.add(
                    np.concatenate(obs), np.concatenate(actions),
                    team_reward, np.concatenate(next_obs), done,
                )
            if len(replay) >= model.batch_size:
                model.train(replay)

            episode_reward += team_reward
            steps += 1
            obs = next_obs
            if done:
                break

        metrics = env.metrics()
        rewards_history.append(episode_reward)
        visited_history.append(metrics["visited_tasks"])
        uploaded_history.append(metrics["uploaded_tasks"])
        step_history.append(steps)
        guided_history.append(int(guided_run))
        for name, values in env.reward_term_totals.items():
            term_history.setdefault(name, []).append(values.copy())

        if ((episode + 1) % evaluation_interval == 0
                or episode == 0 or episode + 1 == episodes):
            eval_metrics = _evaluate(
                algorithm, model, env, safety_outage_threshold,
                teacher_action_blend=teacher_action_blend,
            )
            score = _score(eval_metrics)
            completion_score = _completion_score(eval_metrics)
            eval_records.append({
                "episode": episode + 1,
                "metrics": eval_metrics,
                "score": list(score),
                "completion_score": list(completion_score),
            })
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = eval_metrics
                model.save(best_path)
            if (
                best_completion_score is None
                or completion_score > best_completion_score
            ):
                best_completion_score = completion_score
                best_completion_metrics = eval_metrics
                model.save(best_completion_path)
            print(
                "[%s] ep=%d/%d reward=%.2f train_steps=%d "
                "visited=%d/%d uploaded=%d/%d eval_feasible=%s"
                % (
                    algorithm, episode + 1, episodes, episode_reward, steps,
                    metrics["visited_tasks"], env.n_tasks,
                    metrics["uploaded_tasks"], env.n_tasks,
                    eval_metrics["feasible"],
                ),
                flush=True,
            )

    model.save(last_path)
    if best_score is None:
        model.save(best_path)
        best_metrics = env.metrics()
        best_metrics["return"] = 0.0
        best_metrics["remaining_distance_m"] = _remaining_distance(env)
    if best_completion_score is None:
        model.save(best_completion_path)
        best_completion_metrics = dict(best_metrics)

    return {
        "rewards": rewards_history,
        "visited": visited_history,
        "uploaded": uploaded_history,
        "steps": step_history,
        "guided": guided_history,
        "bc_weight": bc_weight_history,
        "reward_terms": term_history,
        "evaluations": eval_records,
        "best_metrics": best_metrics,
        "best_completion_metrics": best_completion_metrics,
        "elapsed_s": time.perf_counter() - start,
    }


def _train_mappo(model, env, episodes, evaluation_interval, best_path,
                 best_completion_path, last_path, guide_prob, guide_min_prob,
                 safety_outage_threshold, teacher_action_blend):
    replay = mappo._RolloutBuffer(
        env.n_uav, env.local_obs_dim, env.action_dim, env.T
    )
    rewards_history = []
    visited_history = []
    uploaded_history = []
    step_history = []
    guided_history = []
    term_history = {}
    eval_records = []
    best_score = None
    best_metrics = None
    best_completion_score = None
    best_completion_metrics = None
    start = time.perf_counter()

    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0.0
        steps = 0
        fraction = episode / max(episodes - 1, 1)
        episode_guide_probability = (
            guide_prob + (guide_min_prob - guide_prob) * fraction
        )
        guided_run = bool(np.random.rand() < episode_guide_probability)
        while True:
            teacher_actions = env.project_safe_actions(
                _guided_actions(env),
                outage_threshold=safety_outage_threshold,
                update_stats=False,
            )
            if guided_run:
                actions = [
                    np.asarray(action, dtype=float).copy()
                    for action in teacher_actions
                ]
                log_probs, value = _mappo_guide_step(model, obs, actions)
            else:
                actions, log_probs, value = model.act(obs, add_noise=True)
                if teacher_action_blend > 0.0:
                    actions = _blend_teacher_actions(
                        actions, teacher_actions, teacher_action_blend
                    )
                    log_probs, value = _mappo_guide_step(
                        model, obs, actions
                    )
            projected = env.project_safe_actions(
                actions, outage_threshold=safety_outage_threshold
            )
            if not np.allclose(projected, actions):
                actions = projected
                log_probs, value = _mappo_guide_step(model, obs, actions)
            else:
                actions = projected
            next_obs, rewards, terminal, info = env.step(actions)
            done = bool(terminal or info["truncated"])
            team_reward = float(np.sum(rewards))
            replay.store(
                obs, actions, log_probs, team_reward, value, done, next_obs,
                teacher_actions=teacher_actions,
            )
            if replay.is_full():
                model.train(replay)
                replay = mappo._RolloutBuffer(
                    env.n_uav, env.local_obs_dim, env.action_dim, env.T
                )
            episode_reward += team_reward
            steps += 1
            obs = next_obs
            if done:
                break

        if len(replay):
            model.train(replay)
            replay = mappo._RolloutBuffer(
                env.n_uav, env.local_obs_dim, env.action_dim, env.T
            )

        metrics = env.metrics()
        rewards_history.append(episode_reward)
        visited_history.append(metrics["visited_tasks"])
        uploaded_history.append(metrics["uploaded_tasks"])
        step_history.append(steps)
        guided_history.append(int(guided_run))
        for name, values in env.reward_term_totals.items():
            term_history.setdefault(name, []).append(values.copy())

        if ((episode + 1) % evaluation_interval == 0
                or episode == 0 or episode + 1 == episodes):
            eval_metrics = _evaluate(
                "MAPPO", model, env, safety_outage_threshold,
                teacher_action_blend=teacher_action_blend,
            )
            score = _score(eval_metrics)
            completion_score = _completion_score(eval_metrics)
            eval_records.append({
                "episode": episode + 1,
                "metrics": eval_metrics,
                "score": list(score),
                "completion_score": list(completion_score),
            })
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = eval_metrics
                model.save(best_path)
            if (
                best_completion_score is None
                or completion_score > best_completion_score
            ):
                best_completion_score = completion_score
                best_completion_metrics = eval_metrics
                model.save(best_completion_path)
            print(
                "[MAPPO] ep=%d/%d reward=%.2f train_steps=%d "
                "visited=%d/%d uploaded=%d/%d eval_feasible=%s"
                % (
                    episode + 1, episodes, episode_reward, steps,
                    metrics["visited_tasks"], env.n_tasks,
                    metrics["uploaded_tasks"], env.n_tasks,
                    eval_metrics["feasible"],
                ),
                flush=True,
            )

    model.save(last_path)
    if best_score is None:
        model.save(best_path)
        best_metrics = env.metrics()
        best_metrics["return"] = 0.0
        best_metrics["remaining_distance_m"] = _remaining_distance(env)
    if best_completion_score is None:
        model.save(best_completion_path)
        best_completion_metrics = dict(best_metrics)
    return {
        "rewards": rewards_history,
        "visited": visited_history,
        "uploaded": uploaded_history,
        "steps": step_history,
        "guided": guided_history,
        "reward_terms": term_history,
        "evaluations": eval_records,
        "best_metrics": best_metrics,
        "best_completion_metrics": best_completion_metrics,
        "elapsed_s": time.perf_counter() - start,
    }


def _train_ma_d3qn(
    model, env, episodes, evaluation_interval, best_path,
    best_completion_path, last_path, guide_prob, guide_min_prob,
    bc_weight, bc_min_weight, bc_decay_episodes, demo_fraction,
    safety_outage_threshold, n_step, action_repeat,
    pretrain_demo_episodes, pretrain_updates,
):
    buffer = ma_d3qn.DiscreteReplayBuffer(
        env.local_obs_dim, capacity=200_000, demo_fraction=demo_fraction
    )
    rewards_history = []
    visited_history = []
    uploaded_history = []
    step_history = []
    guided_history = []
    term_history = {}
    eval_records = []
    best_score = None
    best_metrics = None
    best_completion_score = None
    best_completion_metrics = None
    start = time.perf_counter()

    def run_episode(episode, guided_run, learning):
        obs = env.reset()
        episode_reward = 0.0
        actual_steps = 0
        fraction = episode / max(episodes - 1, 1)
        epsilon = 0.3 + (0.02 - 0.3) * fraction
        assemblers = [
            ma_d3qn.NStepAssembler(n_step, model.gamma)
            for _ in range(env.n_uav)
        ]
        while True:
            if guided_run:
                actions = _guided_actions(env)
            else:
                actions = _select_actions("MA-D3QN", model, obs, epsilon)
            actions = env.project_safe_actions(
                actions, outage_threshold=safety_outage_threshold
            )
            indices = [model.codec.encode(action) for action in actions]
            discrete_actions = np.asarray(
                [model.codec.decode(index) for index in indices], dtype=float
            )

            next_obs = obs
            team_reward = 0.0
            done = False
            for _ in range(max(int(action_repeat), 1)):
                next_obs, rewards, terminal, info = env.step(discrete_actions)
                team_reward += float(np.sum(rewards))
                actual_steps += 1
                done = bool(terminal or info["truncated"])
                if done:
                    break

            for uav_id in range(env.n_uav):
                records = assemblers[uav_id].push(
                    obs[uav_id], indices[uav_id], team_reward,
                    next_obs[uav_id], done,
                )
                for record in records:
                    buffer.add(record, demo=guided_run)
            if learning and len(buffer) >= model.batch_size:
                model.train(buffer)
            episode_reward += team_reward
            obs = next_obs
            if done:
                break
        return episode_reward, actual_steps, env.metrics()

    model.bc_weight = float(bc_weight)
    for demo_episode in range(int(pretrain_demo_episodes)):
        run_episode(demo_episode, guided_run=True, learning=False)
    for _ in range(int(pretrain_updates)):
        model.train(buffer)

    for episode in range(episodes):
        fraction = episode / max(episodes - 1, 1)
        episode_guide_probability = (
            guide_prob + (guide_min_prob - guide_prob) * fraction
        )
        guided_run = bool(np.random.rand() < episode_guide_probability)
        model.bc_weight = linear_decay(
            bc_weight, episode, bc_decay_episodes, floor=bc_min_weight
        )
        episode_reward, steps, metrics = run_episode(
            episode, guided_run=guided_run, learning=True
        )
        rewards_history.append(episode_reward)
        visited_history.append(metrics["visited_tasks"])
        uploaded_history.append(metrics["uploaded_tasks"])
        step_history.append(steps)
        guided_history.append(int(guided_run))
        for name, values in env.reward_term_totals.items():
            term_history.setdefault(name, []).append(values.copy())

        if ((episode + 1) % evaluation_interval == 0
                or episode == 0 or episode + 1 == episodes):
            eval_metrics = _evaluate(
                "MA-D3QN", model, env, safety_outage_threshold
            )
            score = _score(eval_metrics)
            completion_score = _completion_score(eval_metrics)
            eval_records.append({
                "episode": episode + 1,
                "metrics": eval_metrics,
                "score": list(score),
                "completion_score": list(completion_score),
            })
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = eval_metrics
                model.save(best_path)
            if (
                best_completion_score is None
                or completion_score > best_completion_score
            ):
                best_completion_score = completion_score
                best_completion_metrics = eval_metrics
                model.save(best_completion_path)
            print(
                "[MA-D3QN] ep=%d/%d reward=%.2f steps=%d "
                "visited=%d/%d uploaded=%d/%d eval_feasible=%s"
                % (
                    episode + 1, episodes, episode_reward, steps,
                    metrics["visited_tasks"], env.n_tasks,
                    metrics["uploaded_tasks"], env.n_tasks,
                    eval_metrics["feasible"],
                ),
                flush=True,
            )

    model.save(last_path)
    if best_score is None:
        model.save(best_path)
        best_metrics = env.metrics()
        best_metrics["return"] = 0.0
        best_metrics["remaining_distance_m"] = _remaining_distance(env)
    if best_completion_score is None:
        model.save(best_completion_path)
        best_completion_metrics = dict(best_metrics)
    return {
        "rewards": rewards_history,
        "visited": visited_history,
        "uploaded": uploaded_history,
        "steps": step_history,
        "guided": guided_history,
        "reward_terms": term_history,
        "evaluations": eval_records,
        "best_metrics": best_metrics,
        "best_completion_metrics": best_completion_metrics,
        "elapsed_s": time.perf_counter() - start,
    }


def run_algorithm(args, algorithm):
    device = _device()
    _set_device(device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenario = load_stc_scenario(
        args.uav_num, task_seed=args.seed, map_seed=args.seed
    )
    field = PerSectorRadioField.from_scenario(scenario, spacing_m=args.spacing_m)
    if args.routes_json:
        routes = _load_routes(args.routes_json, args.uav_num)
        route_source = str(Path(args.routes_json).resolve())
    else:
        routes = _fixed_routes(args.uav_num, args.seed)
        route_source = "computed"
    env = ExclusiveSectorFixedRouteEnv(
        scenario,
        field,
        routes,
        horizon_steps=args.horizon_steps,
        arrival_radius_m=args.arrival_radius_m,
        safe_distance_m=args.safe_distance_m,
        outage_probability_threshold=args.outage_probability_threshold,
        outage_max_s=args.outage_max_s,
    )
    model = _make_model(
        algorithm, env,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        demo_margin=args.demo_margin,
        teacher_anchor_weight=args.teacher_anchor_weight,
    )
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError("initial checkpoint not found: %s" % init_path)
        model.load(str(init_path))
        print("loaded initial checkpoint -> %s" % init_path.resolve(), flush=True)

    run_id = args.run_id or "pilot-exclusive-u2-s42-%s-e%d-s%d" % (
        algorithm.lower(), args.episodes, args.horizon_steps
    )
    config = {
        "experiment": "exclusive MADRL short integration pilot",
        "scenario_label": "s42",
        "uav_num": int(args.uav_num),
        "algorithm": algorithm,
        "episodes": int(args.episodes),
        "horizon_steps": int(args.horizon_steps),
        "arrival_radius_m": float(args.arrival_radius_m),
        "safe_distance_m": float(args.safe_distance_m),
        "outage_probability_threshold": float(args.outage_probability_threshold),
        "outage_max_s": float(args.outage_max_s),
        "spacing_m": float(args.spacing_m),
        "seed": int(args.seed),
        "device": str(device),
        "mapguide_probability": float(args.guide_prob),
        "mapguide_min_probability": float(args.guide_min_prob),
        "bc_weight": float(args.bc_weight),
        "bc_min_weight": float(args.bc_min_weight),
        "bc_decay_episodes": int(args.bc_decay_episodes),
        "demo_fraction": float(args.demo_fraction),
        "teacher_anchor_weight": float(args.teacher_anchor_weight),
        "teacher_action_blend": float(args.teacher_action_blend),
        "actor_lr": float(args.actor_lr),
        "critic_lr": float(args.critic_lr),
        "safety_outage_threshold": float(args.safety_outage_threshold),
        "n_step": int(args.n_step),
        "action_repeat": int(args.action_repeat),
        "pretrain_demo_episodes": int(args.pretrain_demo_episodes),
        "pretrain_updates": int(args.pretrain_updates),
        "demo_margin": float(args.demo_margin),
        "init_checkpoint": (
            str(Path(args.init_checkpoint).resolve())
            if args.init_checkpoint else None
        ),
        "routes": {str(k): v for k, v in routes.items()},
        "route_source": route_source,
        "assignment": "4D",
        "sequencing": "GA-EQTSP",
    }
    rc = RunContext.create(
        args.run_root, "exclusive_marl_pilot", config,
        run_id=run_id, expected=EXPECTED
    )
    rc.write_json("resolved_config", config)
    rc.register_input("task_positions_m", data=scenario.task_positions_m)
    rc.register_input("task_data_bits", data=scenario.task_data_bits)
    rc.register_input("routes", data=routes)
    if args.init_checkpoint:
        rc.register_input("initial_checkpoint", path=args.init_checkpoint)

    best_path = rc.run_dir / "best.pth"
    best_constraint_path = rc.run_dir / "best_constraint.pth"
    best_completion_path = rc.run_dir / "best_completion.pth"
    last_path = rc.run_dir / "last.pth"
    if algorithm == "MA-D3QN":
        result = _train_ma_d3qn(
            model, env, args.episodes, args.eval_interval,
            best_path, best_completion_path, last_path,
            args.guide_prob, args.guide_min_prob,
            args.bc_weight, args.bc_min_weight, args.bc_decay_episodes,
            args.demo_fraction, args.safety_outage_threshold,
            args.n_step, args.action_repeat,
            args.pretrain_demo_episodes, args.pretrain_updates,
        )
    elif algorithm == "MAPPO":
        result = _train_mappo(
            model, env, args.episodes, args.eval_interval,
            best_path, best_completion_path, last_path,
            args.guide_prob, args.guide_min_prob,
            args.safety_outage_threshold,
            args.teacher_action_blend,
        )
    else:
        result = _train_off_policy(
            algorithm, model, env, args.episodes, args.eval_interval,
            best_path, best_completion_path, last_path,
            args.guide_prob, args.guide_min_prob,
            args.bc_weight, args.bc_min_weight, args.bc_decay_episodes,
            args.demo_fraction, args.safety_outage_threshold,
            args.teacher_anchor_weight,
            args.teacher_action_blend,
        )
    model.save(last_path)
    # ``best.pth`` is the constraint-prioritised checkpoint. Keep an explicit
    # copy under the descriptive name expected by the run contract.
    best_constraint_path.write_bytes(best_path.read_bytes())

    curve_bytes = io.BytesIO()
    curve_arrays = {
        "episode_rewards": np.asarray(result["rewards"], dtype=float),
        "visited": np.asarray(result["visited"], dtype=int),
        "uploaded": np.asarray(result["uploaded"], dtype=int),
        "steps": np.asarray(result["steps"], dtype=int),
        "guided": np.asarray(result["guided"], dtype=int),
    }
    for name, values in result["reward_terms"].items():
        curve_arrays["reward_term_%s" % name] = np.asarray(values, dtype=float)
    if "bc_weight" in result:
        curve_arrays["bc_weight"] = np.asarray(result["bc_weight"], dtype=float)
    np.savez(curve_bytes, **curve_arrays)
    summary = {
        "run_id": run_id,
        "algorithm": algorithm,
        "scenario_label": "s42",
        "uav_num": int(args.uav_num),
        "episodes": int(args.episodes),
        "horizon_steps": int(args.horizon_steps),
        "elapsed_s": float(result["elapsed_s"]),
        "mapguide_probability": float(args.guide_prob),
        "mapguide_min_probability": float(args.guide_min_prob),
        "guided_episodes": int(np.sum(result["guided"])),
        "bc_weight_initial": float(args.bc_weight),
        "bc_weight_final": (
            float(result["bc_weight"][-1]) if result.get("bc_weight") else None
        ),
        "teacher_anchor_weight": float(args.teacher_anchor_weight),
        "teacher_action_blend": float(args.teacher_action_blend),
        "best_metrics": result["best_metrics"],
        "best_completion_metrics": result["best_completion_metrics"],
        "evaluations": result["evaluations"],
        "routes": {str(k): v for k, v in routes.items()},
    }
    rc.write_json("summary", summary)
    rc.write_json("records", result["evaluations"])
    rc.write_bytes("training_curve.npz", curve_bytes.getvalue())
    rc.write_bytes("best.pth", best_path.read_bytes())
    rc.write_bytes("best_constraint.pth", best_constraint_path.read_bytes())
    rc.write_bytes("best_completion.pth", best_completion_path.read_bytes())
    rc.write_bytes("last.pth", last_path.read_bytes())
    manifest = rc.finalize("success")

    metrics = result["best_metrics"]
    completion_metrics = result["best_completion_metrics"]
    print(
        "finished algorithm=%s run_id=%s best_visited=%d/%d "
        "best_uploaded=%d/%d feasible=%s completion_best=%d/%d elapsed_s=%.1f"
        % (
            algorithm, run_id, metrics["visited_tasks"], env.n_tasks,
            metrics["uploaded_tasks"], env.n_tasks, metrics["feasible"],
            completion_metrics["uploaded_tasks"], env.n_tasks,
            result["elapsed_s"],
        ),
        flush=True,
    )
    print("manifest=%s" % manifest, flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--uav-num", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon-steps", type=int, default=200)
    parser.add_argument("--arrival-radius-m", type=float, default=10.0)
    parser.add_argument("--safe-distance-m", type=float, default=30.0)
    parser.add_argument("--spacing-m", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--guide-prob", type=float, default=0.5)
    parser.add_argument("--outage-probability-threshold", type=float, default=0.5)
    parser.add_argument("--outage-max-s", type=float, default=20.0)
    parser.add_argument("--guide-min-prob", type=float, default=0.5)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--bc-min-weight", type=float, default=0.25)
    parser.add_argument("--bc-decay-episodes", type=int, default=600)
    parser.add_argument("--demo-fraction", type=float, default=0.25)
    parser.add_argument("--teacher-anchor-weight", type=float, default=0.0)
    parser.add_argument("--teacher-action-blend", type=float, default=0.0)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--safety-outage-threshold", type=float, default=0.2)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--pretrain-demo-episodes", type=int, default=40)
    parser.add_argument("--pretrain-updates", type=int, default=20000)
    parser.add_argument("--demo-margin", type=float, default=0.8)
    parser.add_argument("--run-root", default="results/runs_pilot")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--routes-json", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args(argv)
    if not (0.0 <= args.guide_min_prob <= args.guide_prob <= 1.0):
        parser.error("require 0 <= guide-min-prob <= guide-prob <= 1")
    if not (0.0 <= args.bc_min_weight <= args.bc_weight):
        parser.error("require 0 <= bc-min-weight <= bc-weight")
    if not (0.0 <= args.demo_fraction <= 1.0):
        parser.error("require 0 <= demo-fraction <= 1")
    if args.teacher_anchor_weight < 0.0:
        parser.error("teacher-anchor-weight must be non-negative")
    if not 0.0 <= args.teacher_action_blend <= 1.0:
        parser.error("teacher-action-blend must be in [0, 1]")
    if not (0.0 <= args.safety_outage_threshold <= 1.0):
        parser.error("require 0 <= safety-outage-threshold <= 1")
    if args.n_step <= 0 or args.action_repeat <= 0:
        parser.error("n-step and action-repeat must be positive")
    if args.pretrain_demo_episodes < 0 or args.pretrain_updates < 0:
        parser.error("pretrain values must be non-negative")
    return run_algorithm(args, args.algorithm)


if __name__ == "__main__":
    main()
