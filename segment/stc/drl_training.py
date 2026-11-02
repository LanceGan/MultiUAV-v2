"""SI 单段 DRL 训练辅助（奖励包装 / 边集切分 / held-out 评估 / 训练循环）。

训练环境复用 :class:`segment.stc.sac_candidates.StcSacEnv`（SI 单段 rollout
环境，``step`` 不含奖励）；本模块在其外包一层奖励，结构沿用项目定案
（CLAUDE.md「新设计实现进度-关键决策」，量级与旧 100m 设计一致）：

- 对称前进 ``+60·Δdist``（Δdist 为 SI 米 ×0.01，即 100m 归一；远离为负，无套利）
- 上传奖励 ``+1·本步已传 Mbit``
- 时间惩罚 ``−1/步``；中断惩罚 ``−1·p_out/步``（最优扇区 G2A 中断，单机代理口径）
- 终局：到达且传完 ``+400``；到达未传完或超时 ``−1000``（可行性优先）

目标条件化：每 episode 从合法边集随机抽一条边（源点数据按边语义：depot 边 0、
其余携带源点数据）；``--held-out-frac`` 切出的边训练期从不采样，仅用于每
``eval_interval`` 集的确定性泛化评估（口径 = 到达且传完的边占比），best
checkpoint 按该口径选取。产出的 SAC checkpoint 与
:func:`segment.stc.sac_candidates._load_sac_policy` 的兼容性校验完全一致
（7 维状态、动作 ``[v, θ]``、``v∈[0, v_max]``）。
"""

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from segment.stc.generators import _node_data
from segment.stc.sac_candidates import StcSacEnv, legal_edges

ALGOS = ("SAC", "TD3", "DDPG")

# 奖励常量（与旧设计同量级；Δdist 的 0.01 把 SI 米换算回 100m 归一）
PROGRESS_W = 60.0 * 0.01
UPLOAD_W = 1.0 / 1e6
TIME_PENALTY = 1.0
OUTAGE_W = 1.0
R_FEASIBLE = 400.0
R_INFEASIBLE = -1000.0


class SingleSegmentRewardEnv:
    """给 :class:`StcSacEnv` 包一层奖励；``step`` 返回 ``(s, r, done, info)``。"""

    def __init__(self, scenario, field, source, target, data_bits):
        self.field = field
        self.env = StcSacEnv(scenario, field, source, target, data_bits)

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self):
        return self.env.reset()

    def _dist(self):
        return float(np.linalg.norm(self.env.pos - self.env.goal))

    def step(self, action):
        prev_dist = self._dist()
        prev_buf = self.env.data_buffer
        s, done = self.env.step(action)
        new_dist = self._dist()
        uploaded = prev_buf - self.env.data_buffer
        outage = float(np.min(self.field.sample(self.env.pos).g2a_outage))
        r = (PROGRESS_W * (prev_dist - new_dist)
             + uploaded * UPLOAD_W
             - TIME_PENALTY
             - OUTAGE_W * outage)
        arrived = new_dist <= self.env.arrive_radius
        drained = self.env.data_buffer <= 0.0
        feasible = False
        if done:
            feasible = bool(arrived and drained)
            r += R_FEASIBLE if feasible else R_INFEASIBLE
        info = {"arrived": bool(arrived), "drained": bool(drained),
                "feasible": feasible}
        return s, r, done, info


def split_edges(n_tasks, held_out_frac, seed):
    """确定性切分合法边集 → ``(train_edges, heldout_edges)``（不相交、无遗漏）。"""
    edges = list(legal_edges(n_tasks))
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(edges))
    n_held = int(round(float(held_out_frac) * len(edges)))
    held_idx = set(order[:n_held].tolist())
    train = [edges[i] for i in range(len(edges)) if i not in held_idx]
    held = [edges[i] for i in sorted(held_idx)]
    return train, held


def edge_env(scenario, field, edge):
    src, tgt = edge
    return SingleSegmentRewardEnv(scenario, field, src, tgt,
                                  _node_data(scenario, src))


def make_model(algo, v_max):
    max_action = np.array([float(v_max), 2.0 * math.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    if algo == "SAC":
        from segment.sac import SAC
        return SAC(7, 2, max_action, min_action), max_action, min_action
    if algo == "TD3":
        from segment.td3 import TD3
        return TD3(7, 2, max_action, min_action), max_action, min_action
    if algo == "DDPG":
        from segment.ddpg import DDPG
        return DDPG(7, 2, max_action, min_action), max_action, min_action
    raise ValueError("unknown algo %r (expected one of %s)" % (algo, ALGOS))


def deterministic_rollout(model, env, max_action, min_action):
    """确定性 rollout 一条边 → ``(ep_reward, feasible)``。"""
    s = env.reset()
    total = 0.0
    while True:
        a = np.clip(model.select_action(s), min_action, max_action)
        s, r, done, info = env.step(a)
        total += r
        if done:
            return total, bool(info["feasible"])


def heldout_feasible_rate(model, scenario, field, edges, max_action, min_action):
    """held-out 边上的确定性可行率（到达且传完的边占比）。"""
    if not edges:
        return 0.0
    ok = 0
    for edge in edges:
        _, feas = deterministic_rollout(model, edge_env(scenario, field, edge),
                                        max_action, min_action)
        ok += int(feas)
    return ok / len(edges)


def _git_head():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True).stdout
        return out.decode("utf-8", errors="replace").strip() or "unknown"
    except Exception:
        return "unknown"


def _edges_hash(edges):
    payload = json.dumps(sorted([list(e) for e in edges]), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train(scenario, field, *, algo="SAC", total_episodes=3000, train_seed=42,
          held_out_frac=0.2, eval_interval=50, out_dir="results/models/stc_drl",
          run_id="run", log_every=50, replay_size=int(2e5)):
    """目标条件化单段训练主循环；返回产物路径与汇总指标。"""
    import torch
    from trajectory.replay import ReplayBuffer
    from segment.stc.evaluator import content_hash

    if algo not in ALGOS:
        raise ValueError("unknown algo %r" % algo)
    torch.manual_seed(int(train_seed))
    rng = np.random.RandomState(int(train_seed))

    train_edges, held_edges = split_edges(scenario.n_tasks, held_out_frac,
                                          train_seed)
    if not train_edges:
        raise ValueError("empty train edge set")
    # 无 held-out 时退化为训练边均匀探针（仍是确定性评估，只是不测泛化）
    eval_edges = held_edges or [train_edges[i] for i in
                                np.linspace(0, len(train_edges) - 1,
                                            min(6, len(train_edges))).astype(int)]

    model, max_action, min_action = make_model(algo, scenario.limits.v_max_mps)
    replay = ReplayBuffer(7, 2, max_size=replay_size)
    batch_size = model.batch_size
    noise_scale = np.array([scenario.limits.v_max_mps, 1.0], dtype=np.float32)
    expl_start, expl_end = 0.3, 0.02
    expl_decay = (expl_end / expl_start) ** (1.0 / max(int(total_episodes), 1))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "%s_u%d_%s" % (algo.lower(), scenario.n_uav, run_id)
    best_path = out / (stem + "_best.pth")
    last_path = out / (stem + "_last.pth")

    ep_rewards, ep_feasible, eval_curve = [], [], []
    best_rate = -1.0
    t0 = time.perf_counter()

    for ep in range(int(total_episodes)):
        edge = train_edges[rng.randint(len(train_edges))]
        env = edge_env(scenario, field, edge)
        s = env.reset()
        ep_reward = 0.0
        expl = expl_start * (expl_decay ** ep)
        info = {"feasible": False}
        while True:
            if algo == "SAC":
                a = model.sample_action(s)
            else:
                a = model.select_action(s) + rng.normal(0.0, expl, size=2) * noise_scale
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

        if (ep + 1) % int(eval_interval) == 0 or ep == 0:
            rate = heldout_feasible_rate(model, scenario, field, eval_edges,
                                         max_action, min_action)
            eval_curve.append((ep + 1, rate))
            if rate > best_rate:
                best_rate = rate
                model.save(str(best_path))
                print("  [eval] ep %4d | held-out feasible %5.1f%% (new best)"
                      % (ep + 1, 100.0 * rate))
        if (ep + 1) % int(log_every) == 0 or ep == 0:
            print("ep %4d | reward %10.2f | avg100 %10.2f | feas100 %5.1f%% | expl %.3f"
                  % (ep + 1, ep_reward, float(np.mean(ep_rewards[-100:])),
                     100.0 * float(np.mean(ep_feasible[-100:])), expl))

    model.save(str(last_path))
    if best_rate < 0:  # eval 从未触发（理论上 ep==0 必触发，保险）
        model.save(str(best_path))

    npz_path = out / (stem + "_rewards.npz")
    np.savez(npz_path,
             ep_rewards=np.asarray(ep_rewards, dtype=np.float64),
             ep_feasible=np.asarray(ep_feasible, dtype=bool),
             eval_heldout_feasible=np.asarray(eval_curve, dtype=np.float64))
    provenance = {
        "git_head": _git_head(),
        "algo": algo, "total_episodes": int(total_episodes),
        "train_seed": int(train_seed), "held_out_frac": float(held_out_frac),
        "eval_interval": int(eval_interval), "run_id": run_id,
        "scenario_hash": content_hash(scenario),
        "world_hash": getattr(field, "world_hash", None),
        "n_train_edges": len(train_edges), "n_heldout_edges": len(held_edges),
        "train_edges_hash": _edges_hash(train_edges),
        "heldout_edges_hash": _edges_hash(held_edges),
        "best_heldout_feasible": best_rate,
        "wall_time_s": time.perf_counter() - t0,
        "reward_constants": {"progress": "60*Δdist*0.01", "upload": "+1/Mbit",
                             "time": -TIME_PENALTY, "outage": -OUTAGE_W,
                             "feasible": R_FEASIBLE, "infeasible": R_INFEASIBLE},
    }
    prov_path = out / (stem + "_provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return {"best_path": str(best_path), "last_path": str(last_path),
            "npz_path": str(npz_path), "provenance_path": str(prov_path),
            "best_heldout_feasible": best_rate,
            "ep_rewards": ep_rewards, "ep_feasible": ep_feasible,
            "eval_curve": eval_curve}
