"""SAC single-segment candidate production/loading (Task 9).

Rolls a checkpoint-bound SAC policy into per-edge :class:`SegmentCandidate`
objects for the STC SI single-segment environment (metres / seconds / bits /
hertz, deterministic :class:`~segment.stc.radio.PerSectorRadioField`).  The
loader validates checkpoint compatibility: it must be a SAC policy loadable
against the SI single-segment environment (7-dim state, 2-dim ``[v, theta]``
action with ``v in [0, v_max_mps]`` metres/second).  A missing or legacy
(100-m / Mbit statistical-field) checkpoint yields an explicit ``unavailable``
result with a reason -- never a fabricated candidate and never a deterministic
trajectory relabelled as SAC.
"""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from segment.stc.generators import (
    _build_candidate,
    _node_data,
    _node_pos,
    drl_generator_label,
)

# Absolute-data reference for the state feature (bits), mirroring the legacy
# DATA_REF=30 Mbit converted to SI.
DATA_REF_SI = 30e6


@dataclass(frozen=True)
class SacCandidateResult:
    status: str            # "available" | "unavailable"
    candidates: dict       # {(source, target): tuple[SegmentCandidate, ...]}
    reason: str            # "" when available, else the explicit failure reason
    checkpoint_hash: str   # SHA-256 of the checkpoint bytes when available
    checkpoint_path: str


def legal_edges(n_tasks):
    """Exactly ``K(K+1)`` legal directed task edges (mirrors ``library._legal_edges``)."""
    edges = []
    for i in range(n_tasks):
        edges.append((-1, i))                    # depot -> task (zero payload)
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                edges.append((i, j))             # task -> task (source payload)
    for i in range(n_tasks):
        edges.append((i, n_tasks))               # task -> terminal (source payload)
    return edges


class StcSacEnv:
    """SI single-segment rollout environment driving a SAC policy into a trace.

    The state is 7-dimensional (position / goal / data-buffer / time features,
    all in SI units) and the action is ``[v (m/s), theta (rad)]``.  Upload drains
    the buffer at the per-step max-sector full-bandwidth rate (the single-UAV
    proxy), exactly mirroring the training dynamics of an SI SAC policy.
    """

    def __init__(self, scenario, field, source, target, data_bits):
        self.scenario = scenario
        self.field = field
        limits = scenario.limits
        self.v_max = limits.v_max_mps
        self.dt = limits.delta_t_s
        self.arrive_radius = limits.arrive_radius_m
        self.length = scenario.length_m
        self.width = scenario.width_m
        self.bandwidth = limits.bandwidth_hz
        self.start = _node_pos(scenario, source)
        self.goal = _node_pos(scenario, target)
        self.data_size = float(data_bits)
        dist = float(np.linalg.norm(self.goal - self.start))
        self.max_steps = int(max(60.0, dist / (self.v_max * self.dt) * 5.0))
        self.reset()

    @property
    def state_dim(self):
        return 7

    def reset(self):
        self.pos = self.start.copy()
        self.data_buffer = self.data_size
        self.t = 0
        self.positions = [self.pos.copy()]
        return self._state()

    def _state(self):
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        return np.array([
            self.pos[0] / max(self.length, 1e-9),
            self.pos[1] / max(self.width, 1e-9),
            dx / max(self.length, 1e-9),
            dy / max(self.width, 1e-9),
            self.data_buffer / max(self.data_size, 1e-6),
            self.data_buffer / DATA_REF_SI,
            self.t / max(self.max_steps, 1),
        ], dtype=np.float32)

    def step(self, action):
        v = float(np.clip(action[0], 0.0, self.v_max))
        theta = float(action[1])
        self.pos = np.array([
            min(max(self.pos[0] + v * math.cos(theta) * self.dt, 0.0), self.length),
            min(max(self.pos[1] + v * math.sin(theta) * self.dt, 0.0), self.width),
        ])
        self.positions.append(self.pos.copy())
        s = self.field.sample(self.pos)
        rate = float(np.max(s.a2g_efficiency)) * self.bandwidth
        self.data_buffer = max(0.0, self.data_buffer - rate * self.dt)
        self.t += 1
        done = (float(np.linalg.norm(self.pos - self.goal)) <= self.arrive_radius
                or self.t >= self.max_steps)
        return self._state(), done


def _load_sac_policy(checkpoint_path, scenario):
    """Load and validate a SAC policy against the STC SI environment.

    Returns ``(policy, digest, reason)`` where ``policy`` is a
    :class:`~segment.sac.SAC` (``None`` on failure) and ``reason`` explains any
    incompatibility.  Validation: torch-loadable SAC actor (7/2 dims), and
    SI action bounds ``v in [0, v_max_mps]`` (the action-scale/bias buffers
    encode the training units, so a legacy 100-m/Mbit checkpoint is rejected).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        return None, "", "SAC checkpoint file is missing: %s" % checkpoint_path

    import torch
    try:
        ckpt = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        return None, "", "SAC checkpoint is not torch-loadable: %r" % exc
    if not isinstance(ckpt, dict) or "actor" not in ckpt:
        return None, "", "SAC checkpoint is missing the actor state_dict"
    actor = ckpt["actor"]
    for key in ("l1.weight", "l2.weight", "mean_layer.weight", "log_std_layer.weight"):
        if key not in actor:
            return None, "", "checkpoint is not a SAC actor (missing %r)" % key
    if int(actor["l1.weight"].shape[-1]) != 7 or int(actor["mean_layer.weight"].shape[0]) != 2:
        return None, "", "SAC checkpoint state/action dims are not (7, 2)"

    from segment.sac import SAC
    v_max = float(scenario.limits.v_max_mps)
    policy = SAC(7, 2, np.array([v_max, 2.0 * math.pi], dtype=np.float32),
                 np.array([0.0, 0.0], dtype=np.float32))
    try:
        policy.load(str(path))
    except Exception as exc:
        return None, "", "SAC checkpoint does not load into the STC policy: %r" % exc

    scale = np.asarray(policy.actor.action_scale.detach().cpu().numpy(), dtype=float)
    bias = np.asarray(policy.actor.action_bias.detach().cpu().numpy(), dtype=float)
    expected_scale = np.array([v_max / 2.0, math.pi])
    expected_bias = np.array([v_max / 2.0, math.pi])
    if not (np.allclose(scale, expected_scale, rtol=1e-2, atol=1e-2)
            and np.allclose(bias, expected_bias, rtol=1e-2, atol=1e-2)):
        return None, "", (
            "SAC checkpoint action bounds %s are not the STC SI single-segment "
            "environment (expected v in [0, %.1f] m/s); a legacy 100-m/Mbit "
            "statistical-field checkpoint is incompatible" % (list(scale), v_max))

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return policy, digest, ""


def _rollout_drl_candidate(scenario, field, policy, source, target, data_bits,
                           generator):
    """Deterministic rollout of a DRL policy into one SI ``SegmentCandidate``."""
    env = StcSacEnv(scenario, field, source, target, data_bits)
    state = env.reset()
    v_max = scenario.limits.v_max_mps
    while True:
        action = np.clip(policy.select_action(state),
                         [0.0, 0.0], [v_max, 2.0 * math.pi])
        state, done = env.step(action)
        if done:
            break
    positions = list(env.positions)
    positions[-1] = _node_pos(scenario, target).copy()  # snap the anchor to target
    hold = tuple(positions[-2]) if len(positions) >= 2 else tuple(positions[0])
    return _build_candidate(scenario, field, source, target, generator,
                            positions, hold, data_bits, True,
                            candidate_id="sac_%d_%d" % (source, target))


def _produce_drl_segment_candidates(checkpoint_path, scenario, field, model):
    if checkpoint_path is None or str(checkpoint_path) == "":
        return SacCandidateResult(
            "unavailable", {}, "%s checkpoint is not configured (no path)" % model,
            "", str(checkpoint_path or ""))
    policy, digest, reason = _load_sac_policy(checkpoint_path, scenario)
    if policy is None:
        return SacCandidateResult("unavailable", {}, reason, "",
                                  str(checkpoint_path))
    generator = drl_generator_label(model, digest)
    candidates = {}
    for (src, tgt) in legal_edges(scenario.n_tasks):
        data_bits = _node_data(scenario, src)
        candidates[(src, tgt)] = (_rollout_drl_candidate(
            scenario, field, policy, src, tgt, data_bits, generator),)
    return SacCandidateResult("available", candidates, "", digest,
                              str(checkpoint_path))


def produce_sac_segment_candidates(checkpoint_path, scenario, field):
    """Produce per-edge SAC candidates from a checkpoint, or an explicit
    ``unavailable`` result when the checkpoint is missing/incompatible."""
    return _produce_drl_segment_candidates(checkpoint_path, scenario, field,
                                           model="sac")


def load_sac_candidates(checkpoint_path, scenario, field):
    """Alias of :func:`produce_sac_segment_candidates` (loading entry point)."""
    return produce_sac_segment_candidates(checkpoint_path, scenario, field)
