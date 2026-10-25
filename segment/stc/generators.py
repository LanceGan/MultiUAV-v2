"""Legal SI-unit single-segment trace generators (Task 4).

Each generator produces one or more ``SegmentCandidate`` traces.  Traces are
piecewise-straight at ``v_max``, sampled from the per-sector radio field; the
upload hold (if needed) happens at an explicit point *outside* the next task's
arrival radius, and the trace always ends at the shared service anchor.
"""

import numpy as np

from segment.stc.types import SegmentCandidate


def _node_pos(scenario, node_id):
    if node_id == -1:
        return np.asarray(scenario.depot_m, dtype=float)
    if node_id == scenario.n_tasks:
        return np.asarray(scenario.terminal_m, dtype=float)
    return np.asarray(scenario.task_positions_m[node_id], dtype=float)


def _node_data(scenario, node_id):
    if 0 <= node_id < scenario.n_tasks:
        return float(scenario.task_data_bits[node_id])
    return 0.0


def _trace(scenario, field, src_m, tgt_m, data_bits, via=()):
    """Build (positions_m, hold_position_m) for src -> via... -> tgt.

    Flies straight legs at v_max, stopping the final leg at the arrival-radius
    boundary; appends enough holding steps to finish the upload there; then a
    final connector into the anchor.
    """
    limits = scenario.limits
    pts = [np.asarray(src_m, float)] + [np.asarray(v, float) for v in via] + [np.asarray(tgt_m, float)]

    fly = []
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        d = float(np.linalg.norm(b - a))
        direction = (b - a) / max(d, 1e-9)
        is_last = (k == len(pts) - 2)
        stop = max(0.0, d - limits.arrive_radius_m) if is_last else d
        n = int(np.ceil(stop / (limits.v_max_mps * limits.delta_t_s)))
        leg = [a + i * limits.v_max_mps * limits.delta_t_s * direction for i in range(n + 1)]
        if n >= 1:
            leg[-1] = a + stop * direction
        fly.extend(leg if not fly else leg[1:])

    hold_position = fly[-1] if fly else pts[0]
    # flight trace ends at the hold point; the final connector enters the anchor.
    # The hold itself is contention-dependent and added by the evaluator (Task 6).
    positions = fly + [pts[-1]]
    return positions, hold_position


def _build_candidate(scenario, field, source, target, generator, positions,
                     hold_position, data_bits, rankable, candidate_id=None):
    full_rate = np.zeros((len(positions), field.n_sectors))
    sectors = np.zeros(len(positions), dtype=int)
    outage = np.zeros((len(positions), field.n_sectors))
    for t, pos in enumerate(positions):
        s = field.sample(pos)
        full_rate[t] = s.a2g_efficiency * scenario.limits.bandwidth_hz
        sectors[t] = int(np.argmax(s.a2g_efficiency))
        outage[t] = s.g2a_outage
    cid = candidate_id or "%s_%d_%d" % (generator, source, target)
    return SegmentCandidate(
        candidate_id=cid,
        source=source, target=target, generator=generator,
        positions_m=np.asarray(positions, dtype=float),
        full_rate_bps=full_rate, sector_ids=sectors, g2a_outage=outage,
        flight_steps=len(positions), hold_position_m=tuple(hold_position),
        source_data_bits=data_bits, rankable=rankable,
    )


def generate_straight_relaxation(scenario, field, source, target):
    """Relaxation lower bound: straight at v_max, upload ignored, non-rankable."""
    src = _node_pos(scenario, source)
    tgt = _node_pos(scenario, target)
    data = _node_data(scenario, source)
    positions, hold = _trace(scenario, field, src, tgt, 0.0, via=())
    # relaxation: hold is not applied (ignore upload), so trim any hold steps
    return _build_candidate(scenario, field, source, target, "straight_relaxation",
                            positions, hold, data, rankable=False)


def generate_straight_hold(scenario, field, source, target):
    """Legal naive baseline: straight flight, hold before the arrival radius."""
    src = _node_pos(scenario, source)
    tgt = _node_pos(scenario, target)
    data = _node_data(scenario, source)
    positions, hold = _trace(scenario, field, src, tgt, data, via=())
    return _build_candidate(scenario, field, source, target, "straight_hold",
                            positions, hold, data, rankable=True)


def generate_comm_waypoints(scenario, field, source, target,
                            grid_shape=(10, 10), max_waypoints=1):
    """Discrete relay-point search: direct candidate + one-relay candidates."""
    src = _node_pos(scenario, source)
    tgt = _node_pos(scenario, target)
    data = _node_data(scenario, source)

    candidates = []
    # direct candidate (same trace as straight_hold, relabelled)
    positions, hold = _trace(scenario, field, src, tgt, data, via=())
    candidates.append(_build_candidate(scenario, field, source, target, "comm_direct",
                                       positions, hold, data, rankable=True))

    margin = scenario.limits.arrive_radius_m
    xs = np.linspace(margin, scenario.length_m - margin, grid_shape[0])
    ys = np.linspace(margin, scenario.width_m - margin, grid_shape[1])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            relay = np.array([x, y])
            positions, hold = _trace(scenario, field, src, tgt, data, via=(relay,))
            candidates.append(_build_candidate(
                scenario, field, source, target, "comm_waypoint",
                positions, hold, data, rankable=True,
                candidate_id="comm_waypoint_%d_%d_%d_%d" % (source, target, i, j)))
    return candidates
