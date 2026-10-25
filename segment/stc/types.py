"""Immutable SI-unit data contracts for STC-ALNS (Task 2).

All quantities are SI: metres, seconds, bits, hertz, watt.  Legacy 100-m units,
Mbit and MHz are converted only at the scenario/radio boundary (``scenario.py``
/ ``radio.py``), never inside the core package.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioLimits:
    delta_t_s: float
    v_max_mps: float
    uav_height_m: float
    arrive_radius_m: float
    max_wait_s: float
    mission_deadline_s: float
    bandwidth_hz: float
    uplink_power_w: float
    eps_out: float
    w_t: float
    w_d: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(
                    "limit %s must be positive finite, got %r" % (name, val))


@dataclass(frozen=True)
class ReferenceValues:
    """Frozen normalization reference (spec §5.7).

    ``t_ref_s`` and ``d_ref_s`` are positive SI seconds.  ``source_hash`` binds
    the reference to the pre-registered baseline that produced it so an
    experiment family can verify it never changes.
    """
    t_ref_s: float
    d_ref_s: float
    source_hash: str

    def __post_init__(self):
        for name in ("t_ref_s", "d_ref_s"):
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(
                    "reference %s must be positive finite, got %r" % (name, val))
        if not self.source_hash:
            raise ValueError("source_hash must be a non-empty string")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Single authoritative scoring path ``w_t*T/T_ref + w_d*D/D_ref`` (spec §5.7)."""
    reference: ReferenceValues
    w_t: float
    w_d: float

    def __post_init__(self):
        for name in ("w_t", "w_d"):
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(
                    "weight %s must be positive finite, got %r" % (name, val))

    def score(self, makespan_s, total_delay_s):
        """Normalized objective.  The only place the objective formula lives."""
        return (self.w_t * makespan_s / self.reference.t_ref_s
                + self.w_d * total_delay_s / self.reference.d_ref_s)


@dataclass(frozen=True)
class RadioSample:
    """Per-sector A2G spectral efficiency and G2A outage at one position."""
    a2g_efficiency: np.ndarray   # (n_sectors,) bit/s/Hz
    g2a_outage: np.ndarray       # (n_sectors,) probability


@dataclass(frozen=True, eq=False)
class SegmentCandidate:
    candidate_id: str
    source: int
    target: int
    generator: str
    positions_m: np.ndarray      # (T, 2)
    full_rate_bps: np.ndarray    # (T, n_sectors)
    sector_ids: np.ndarray       # (T,) preferred sector per slot
    g2a_outage: np.ndarray       # (T, n_sectors)
    flight_steps: int
    hold_position_m: tuple
    source_data_bits: float
    rankable: bool = True   # False for relaxation references (never ranked)

    def __post_init__(self):
        T = len(self.positions_m)
        if (len(self.full_rate_bps) != T or len(self.sector_ids) != T
                or len(self.g2a_outage) != T):
            raise ValueError("trace arrays must share the same first dimension")


@dataclass(frozen=True)
class SegmentLibrary:
    candidates: dict   # (source, target) -> tuple[SegmentCandidate, ...]
    map_seed: int
    task_seed: int
    world_hash: str


@dataclass(frozen=True)
class PlanSolution:
    routes: tuple            # tuple of tuple of task ids (0..K-1)
    candidate_ids: tuple     # tuple of tuple of candidate ids (per traversed edge)

    def __post_init__(self):
        flat = [t for r in self.routes for t in r]
        if len(flat) != len(set(flat)):
            raise ValueError("tasks must appear at most once (no duplicates)")


@dataclass(frozen=True)
class TimelineSlot:
    """One synchronous evaluator slot for one UAV.

    All quantities are SI.  ``edge_id`` is the route-edge index for this UAV
    (``uav_id`` lives on the enclosing :class:`UAVTimeline`), so a later task can
    key hotspots by ``(uav_id, route_edge_index, source, target, candidate_id)``.

    ``pre_buffer_bits`` is the buffer at the *start* of the slot (before this
    slot's upload); ``active_uploader`` is exactly ``pre_buffer_bits > 0``, so a
    buffer that drains to zero mid-slot still consumed bandwidth and caused
    conflict.  ``rate_bps`` is the exclusive (full-bandwidth) rate at the serving
    sector; ``uploaded_bits`` is the actual upload
    ``min(pre_buffer_bits, rate_bps * allocated_bandwidth_hz/B * dt)``.
    """
    time_start_s: float
    time_end_s: float
    edge_id: int
    candidate_id: str
    position_m: tuple
    airborne: bool
    pre_buffer_bits: float
    post_buffer_bits: float
    active_uploader: bool
    serving_sector: int
    allocated_bandwidth_hz: float
    rate_bps: float
    uploaded_bits: float
    g2a_outage: float
    holding: bool
    service_event: str


@dataclass(frozen=True)
class UAVTimeline:
    uav_id: int
    slots: tuple   # tuple[TimelineSlot, ...]


@dataclass(frozen=True)
class ViolationVector:
    unserved_tasks: int
    upload_failures: int
    outage_excess: float
    deadline_exceeded: int
    wait_exceeded: int


@dataclass(frozen=True)
class EvaluationResult:
    feasible: bool
    makespan_s: float
    total_delay_s: float
    outage_ratio: tuple
    violations: ViolationVector
    timelines: tuple
    objective: float = float("inf")


@dataclass(frozen=True)
class ConflictReport:
    total_overlap_s: float
    lost_bits: float
    per_uav_loss: tuple
    edge_hotspots: dict
    pairwise_windows: tuple


@dataclass(frozen=True)
class SearchResult:
    best_solution: PlanSolution
    best_result: EvaluationResult
    search_trace: tuple
    evaluator_calls: dict


@dataclass(frozen=True)
class STCScenario:
    n_uav: int
    n_tasks: int
    task_positions_m: tuple    # tuple of (x, y) metres
    task_data_bits: tuple      # tuple of float bits
    depot_m: tuple
    terminal_m: tuple
    limits: ScenarioLimits
    length_m: float            # region extent (metres)
    width_m: float
    n_bs: int                  # number of base stations
    task_seed: int
    map_seed: int

    @property
    def n_sectors(self):
        return self.n_bs * 3   # three antenna sectors per BS
