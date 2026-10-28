"""Slot-synchronous shared-bandwidth state machine (authoritative evaluator).

Every formal metric comes from here.  Per slot it activates due UAVs, advances
each along its selected candidate trace, holds at the arrival boundary until the
buffer drains, associates one serving sector per airborne UAV, splits each
sector's bandwidth among *active uploaders* (pre-slot non-empty buffers only),
updates buffers, applies service events after upload/motion, and records a typed
:class:`TimelineSlot` per UAV.

Slot timing: slot ``n`` covers ``[n*dt, (n+1)*dt)``, so a UAV that completes in
the first slot has completion time ``(0+1)*dt = dt`` (never zero).  Raw metrics
(T, D, outage, violations) are computed exactly once; the normalized objective
is scored only through :class:`ObjectiveSpec`.

The single physics path is :func:`_run_slots`, consumed by both
:func:`evaluate_solution` (full replay from slot zero) and
:func:`simulate_from_state` (resume an already-running state from a later slot),
so the incremental evaluator never re-implements the slot machine.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass, replace

import numpy as np

from segment.stc.types import (
    BalanceMetrics,
    EvaluationResult,
    ObjectiveSpec,
    PlanSolution,
    ReferenceValues,
    TimelineSlot,
    UAVTimeline,
    ViolationVector,
)


def select_serving_sector(sample, has_payload, reliability_limit):
    """Serving sector per §5.5: reliable-then-rate when uploading, min-outage otherwise."""
    if has_payload:
        ok = sample.g2a_outage <= reliability_limit
        if np.any(ok):
            return int(np.argmax(np.where(ok, sample.a2g_efficiency, -np.inf)))
        return int(np.argmin(sample.g2a_outage))
    return int(np.argmin(sample.g2a_outage))


def _require_objective_spec(objective_spec):
    if not isinstance(objective_spec, ObjectiveSpec):
        raise ValueError(
            "evaluate_solution requires an ObjectiveSpec; missing/nonpositive "
            "references cannot be used for scoring")


@dataclass
class SimulationState:
    """Mutable slot-machine state; the single authoritative physics path.

    ``start_slot`` is the first slot index this state has not yet processed
    (0 for a fresh replay, later for an incremental resume).  ``snapshot()``
    returns an independent copy whose ``slots_by_uav`` lists share the frozen
    ``TimelineSlot`` records but are otherwise free to append to.
    """
    airborne: list
    done: list
    edge: list
    tidx: list
    buf: list
    completion: list
    edge_hold: list
    outage_time: list
    air_time: list
    upload_time: list
    slots_by_uav: list
    unserved: int
    deadline_exceeded: int
    wait_exceeded: int
    start_slot: int

    def snapshot(self):
        return SimulationState(
            airborne=list(self.airborne), done=list(self.done),
            edge=list(self.edge), tidx=list(self.tidx), buf=list(self.buf),
            completion=list(self.completion), edge_hold=list(self.edge_hold),
            outage_time=list(self.outage_time), air_time=list(self.air_time),
            upload_time=list(self.upload_time),
            slots_by_uav=[list(s) for s in self.slots_by_uav],
            unserved=self.unserved, deadline_exceeded=self.deadline_exceeded,
            wait_exceeded=self.wait_exceeded, start_slot=self.start_slot)

    def light_snapshot(self, start_slot=None):
        """Cheap snapshot without the recorded ``slots_by_uav`` (resume seed).

        The recorded timeline prefix is reconstructed separately from the base
        result, so it is intentionally omitted here to keep per-slot memory
        bounded to O(U).
        """
        return SimulationState(
            airborne=list(self.airborne), done=list(self.done),
            edge=list(self.edge), tidx=list(self.tidx), buf=list(self.buf),
            completion=list(self.completion), edge_hold=list(self.edge_hold),
            outage_time=list(self.outage_time), air_time=list(self.air_time),
            upload_time=list(self.upload_time), slots_by_uav=[],
            unserved=self.unserved, deadline_exceeded=self.deadline_exceeded,
            wait_exceeded=self.wait_exceeded,
            start_slot=self.start_slot if start_slot is None else start_slot)


def _build_cand_seqs(scenario, library, solution):
    """Per-UAV edge + candidate sequences (an empty route is an idle UAV)."""
    n_uav = scenario.n_uav
    K = scenario.n_tasks
    uav_cands = []
    uav_edges = []
    for u in range(n_uav):
        route = solution.routes[u]
        cids = solution.candidate_ids[u]
        if route:
            edges = ([(-1, route[0])]
                     + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
                     + [(route[-1], K)])
            seq = []
            for e, (src, tgt) in enumerate(edges):
                cand = next(c for c in library[(src, tgt)] if c.candidate_id == cids[e])
                seq.append(cand)
            uav_cands.append(seq)
            uav_edges.append(edges)
        else:
            uav_cands.append(())
            uav_edges.append(())
    return uav_cands, uav_edges


def _init_state(scenario, solution):
    n_uav = scenario.n_uav
    airborne = [bool(solution.routes[u]) for u in range(n_uav)]
    done = [not bool(solution.routes[u]) for u in range(n_uav)]
    completion = [0.0 if not solution.routes[u] else None for u in range(n_uav)]
    return SimulationState(
        airborne=airborne, done=done,
        edge=[0] * n_uav, tidx=[0] * n_uav, buf=[0.0] * n_uav,
        completion=completion, edge_hold=[0.0] * n_uav,
        outage_time=[0.0] * n_uav, air_time=[0.0] * n_uav,
        upload_time=[0.0] * n_uav, slots_by_uav=[[] for _ in range(n_uav)],
        unserved=scenario.n_tasks, deadline_exceeded=0, wait_exceeded=0,
        start_slot=0)


def _run_slots(scenario, library, solution, field, state, keep_timelines,
               capture=None, sequences=None):
    """Advance the authoritative slot machine from ``state.start_slot`` to the end.

    ``capture``, when given, is called as ``capture(n, state)`` at the start of
    each processed slot (before any state mutation), so an incremental caller
    can seed later resume points.
    """
    uav_cands, uav_edges = (sequences if sequences is not None else
                            _build_cand_seqs(scenario, library, solution))
    limits = scenario.limits
    n_uav = scenario.n_uav
    K = scenario.n_tasks
    dt = limits.delta_t_s
    # Process only slots whose end is within the mission deadline. Including
    # the slot starting at the deadline allowed late completion to pass.
    max_slots = int(limits.mission_deadline_s / dt + 1e-9)

    for n in range(state.start_slot, max_slots):
        if capture is not None:
            capture(n, state)
        t_start = n * dt
        t_end = (n + 1) * dt

        # pre-slot buffer (buffer at the start of this slot, before upload)
        pre_buf = [state.buf[u] for u in range(n_uav)]

        # 2. advance / hold
        holding = [False] * n_uav
        for u in range(n_uav):
            if state.airborne[u] and not state.done[u]:
                cand = uav_cands[u][state.edge[u]]
                L = len(cand.positions_m)
                if state.tidx[u] < L - 2:
                    state.tidx[u] += 1
                elif state.tidx[u] == L - 2:
                    if pre_buf[u] <= 0.0:
                        state.tidx[u] += 1          # enter the service anchor
                    else:
                        holding[u] = True           # wait outside the arrival radius
                        state.edge_hold[u] += dt
                        if state.edge_hold[u] > limits.max_wait_s + 1e-9:
                            state.wait_exceeded += 1

        # 3. associate sectors + accrue outage over the full airborne time
        sectors = [None] * n_uav
        slot_outage = [0.0] * n_uav
        for u in range(n_uav):
            if state.airborne[u] and not state.done[u]:
                cand = uav_cands[u][state.edge[u]]
                p = cand.positions_m[state.tidx[u]]
                s = field.sample(p)
                sec = select_serving_sector(s, pre_buf[u] > 0.0, limits.eps_out)
                sectors[u] = sec
                slot_outage[u] = float(s.g2a_outage[sec])
                state.outage_time[u] += slot_outage[u] * dt
                state.air_time[u] += dt

        # 4. active uploaders (pre-slot buffer) + bandwidth split
        active = [state.airborne[u] and not state.done[u] and pre_buf[u] > 0.0
                  for u in range(n_uav)]
        co = {}
        for u in range(n_uav):
            if active[u]:
                co[sectors[u]] = co.get(sectors[u], 0) + 1

        # 5. upload + update buffer
        alloc = [0.0] * n_uav
        rate_full = [0.0] * n_uav
        uploaded = [0.0] * n_uav
        post_buf = list(pre_buf)
        for u in range(n_uav):
            if state.airborne[u] and not state.done[u]:
                cand = uav_cands[u][state.edge[u]]
                sec = sectors[u]
                rate_full[u] = float(cand.full_rate_bps[state.tidx[u]][sec])
                if active[u]:
                    alloc[u] = limits.bandwidth_hz / co[sec]
                    rate = rate_full[u] * (alloc[u] / limits.bandwidth_hz)
                    up = min(pre_buf[u], rate * dt)
                    uploaded[u] = up
                    state.buf[u] = pre_buf[u] - up
                    state.upload_time[u] += dt
                else:
                    state.buf[u] = pre_buf[u]  # == 0 (buffers never go negative)
                post_buf[u] = state.buf[u]

        # service event (after upload/motion); applied below
        service_event = [""] * n_uav
        for u in range(n_uav):
            if state.airborne[u] and not state.done[u]:
                cand = uav_cands[u][state.edge[u]]
                if state.tidx[u] == len(cand.positions_m) - 1 and post_buf[u] <= 0.0:
                    _, tgt = uav_edges[u][state.edge[u]]
                    service_event[u] = ("complete" if state.edge[u] ==
                                        len(uav_cands[u]) - 1 else "serve")

        # record typed TimelineSlot
        if keep_timelines:
            for u in range(n_uav):
                if state.airborne[u] and not state.done[u]:
                    cand = uav_cands[u][state.edge[u]]
                    p = cand.positions_m[state.tidx[u]]
                    state.slots_by_uav[u].append(TimelineSlot(
                        time_start_s=t_start,
                        time_end_s=t_end,
                        edge_id=state.edge[u],
                        candidate_id=cand.candidate_id,
                        source=cand.source,
                        target=cand.target,
                        position_m=(float(p[0]), float(p[1])),
                        airborne=True,
                        pre_buffer_bits=pre_buf[u],
                        post_buffer_bits=post_buf[u],
                        active_uploader=active[u],
                        serving_sector=sectors[u],
                        allocated_bandwidth_hz=alloc[u],
                        rate_bps=rate_full[u],
                        uploaded_bits=uploaded[u],
                        g2a_outage=slot_outage[u],
                        holding=holding[u],
                        service_event=service_event[u],
                    ))

        # 6. apply service events (buffer load / edge advance / terminal done)
        for u in range(n_uav):
            if state.airborne[u] and not state.done[u] and service_event[u]:
                _, tgt = uav_edges[u][state.edge[u]]
                state.edge_hold[u] = 0.0
                if service_event[u] == "complete":
                    state.done[u] = True
                    state.completion[u] = t_end
                else:
                    state.unserved -= 1
                    state.buf[u] = float(scenario.task_data_bits[tgt])
                    state.edge[u] += 1
                    state.tidx[u] = 0


def _build_result(scenario, library, solution, field, objective_spec, state,
                  keep_timelines):
    limits = scenario.limits
    n_uav = scenario.n_uav

    # post-loop failure accounting (deadline with residual buffer)
    upload_failures = 0
    for u in range(n_uav):
        if state.airborne[u] and not state.done[u]:
            state.deadline_exceeded += 1
            if state.buf[u] > 0.0:
                upload_failures += 1

    outage_ratio = tuple(ot / max(at, 1e-9)
                         for ot, at in zip(state.outage_time, state.air_time))
    feasible = (state.unserved == 0 and upload_failures == 0
                and state.deadline_exceeded == 0
                and state.wait_exceeded == 0
                and all(c is not None for c in state.completion)
                and all(r <= limits.eps_out for r in outage_ratio))
    makespan = max((c for c in state.completion if c is not None), default=0.0)
    total_delay = sum(state.upload_time)
    violations = ViolationVector(
        unserved_tasks=state.unserved,
        upload_failures=upload_failures,
        outage_excess=max(0.0, max(outage_ratio) - limits.eps_out) if outage_ratio else 0.0,
        deadline_exceeded=state.deadline_exceeded,
        wait_exceeded=state.wait_exceeded,
    )

    if keep_timelines:
        timelines = tuple(UAVTimeline(uav_id=u, slots=tuple(state.slots_by_uav[u]))
                          for u in range(n_uav))
    else:
        timelines = ()

    objective = float("inf") if not feasible else objective_spec.score(makespan, total_delay)

    # Task-balance reporting (spec §1.5): secondary, never folded into the
    # objective.  A UAV that never completes before the deadline records NaN.
    completion_times = tuple(
        (float(state.completion[u]) if state.completion[u] is not None else float("nan"))
        for u in range(n_uav))
    comp_arr = np.asarray(completion_times, dtype=float)
    if np.all(np.isnan(comp_arr)):
        t_avg = 0.0
        sigma_t = 0.0
        t_range = 0.0
    else:
        t_avg = float(np.nanmean(comp_arr))
        sigma_t = float(np.nanstd(comp_arr))
        t_range = float(np.nanmax(comp_arr) - np.nanmin(comp_arr))
    balance = BalanceMetrics(completion_times_s=completion_times,
                             t_avg_s=t_avg, sigma_t_s=sigma_t, t_range_s=t_range)

    return EvaluationResult(
        feasible=feasible, makespan_s=makespan, total_delay_s=total_delay,
        outage_ratio=outage_ratio, violations=violations, timelines=timelines,
        objective=objective, balance=balance,
    )


def evaluate_solution(scenario, library, solution, field, objective_spec,
                      keep_timelines=True):
    """Authoritative full replay from slot zero."""
    _require_objective_spec(objective_spec)
    state = _init_state(scenario, solution)
    _run_slots(scenario, library, solution, field, state, keep_timelines)
    return _build_result(scenario, library, solution, field, objective_spec,
                         state, keep_timelines)


def evaluate_direct_segment(scenario, candidate, field, objective_spec,
                            keep_timelines=True):
    """Replay one edge from its source at slot zero through the SAME machine.

    The source has already been serviced: its payload is present before the
    first slot. Completing this edge ends replay without loading target data.
    No route edge or airborne slot is synthesized.
    """
    _require_objective_spec(objective_spec)
    sc = replace(scenario, n_uav=1)
    solution = PlanSolution(routes=((candidate.source,),),
                            candidate_ids=((candidate.candidate_id,),))
    state = _init_state(sc, solution)
    state.unserved = 0
    state.buf[0] = (0.0 if candidate.source == -1
                    else float(candidate.source_data_bits))
    _run_slots(sc, None, solution, field, state, keep_timelines,
               sequences=([[candidate]], [[(candidate.source, candidate.target)]]))
    return _build_result(sc, None, solution, field, objective_spec,
                         state, keep_timelines)


def simulate_from_state(scenario, library, solution, field, objective_spec,
                        state, keep_timelines=True, capture=None):
    """Resume the authoritative slot machine from ``state.start_slot`` onward.

    The physics is identical to :func:`evaluate_solution`; only the starting
    slot differs, so an incremental caller can skip the provably-unchanged
    prefix of the timeline.
    """
    _require_objective_spec(objective_spec)
    _run_slots(scenario, library, solution, field, state, keep_timelines, capture)
    return _build_result(scenario, library, solution, field, objective_spec,
                         state, keep_timelines)


def evaluate_with_snapshots(scenario, library, solution, field, objective_spec,
                            keep_timelines=True):
    """Full replay that also returns lightweight per-slot state snapshots.

    The snapshots map a slot index to a :class:`SimulationState` (without
    ``slots_by_uav``), letting the incremental evaluator resume from any slot.
    """
    _require_objective_spec(objective_spec)
    state = _init_state(scenario, solution)
    snapshots = {}

    def capture(n, st):
        snapshots[n] = st.light_snapshot(n)

    _run_slots(scenario, library, solution, field, state, keep_timelines, capture)
    result = _build_result(scenario, library, solution, field, objective_spec,
                           state, keep_timelines)
    return result, snapshots


def objective(result, objective_spec):
    """Normalized objective through the single scoring path (``ObjectiveSpec``)."""
    return objective_spec.score(result.makespan_s, result.total_delay_s)


def feasibility_rank(result):
    """Lexicographic feasibility rank (lower is better), §7.2."""
    v = result.violations
    return (v.unserved_tasks, v.upload_failures, v.outage_excess,
            v.deadline_exceeded, v.wait_exceeded, result.objective)


def _direct_candidate(library, src, tgt):
    from segment.stc.slsqp_refine import candidate_cost
    cands = library[(src, tgt)]
    return min(cands, key=lambda c: candidate_cost(c, 0.5))


def content_hash(value):
    """Canonical, complete scenario/candidate content identity (including arrays)."""
    def canonical(obj):
        if is_dataclass(obj):
            return canonical(asdict(obj))
        if isinstance(obj, np.ndarray):
            return canonical(obj.tolist())
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: canonical(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [canonical(v) for v in obj]
        return obj
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def library_content_hash(library):
    return content_hash([[src, tgt, sorted((content_hash(c) for c in cs))]
                         for (src, tgt), cs in sorted(library.items())])


def _reference_hash(scenario, ref_trace, t_ref_s, d_ref_s):
    return content_hash({"scenario": scenario, "selected_candidates": ref_trace,
                         "t_ref_s": t_ref_s, "d_ref_s": d_ref_s})


def compute_reference(scenario, library):
    """Reference (T_ref, D_ref) from a greedy NN route split into n_uav groups.

    Returns a frozen :class:`ReferenceValues` binding the reference to the inputs
    that produced it via ``source_hash``.
    """
    from segment.stc.slsqp_refine import candidate_cost
    order = []
    remaining = list(range(scenario.n_tasks))
    cur = np.asarray(scenario.depot_m, float)
    while remaining:
        nxt = min(remaining, key=lambda j: float(np.linalg.norm(
            np.asarray(scenario.task_positions_m[j]) - cur)))
        order.append(nxt)
        cur = np.asarray(scenario.task_positions_m[nxt])
        remaining.remove(nxt)

    max_flight = 0.0
    total_upload = 0.0
    ref_trace = []
    for g in np.array_split(order, scenario.n_uav):
        g = list(g)
        if not g:
            continue
        seq = ([(-1, g[0])]
               + [(g[i], g[i + 1]) for i in range(len(g) - 1)]
               + [(g[-1], scenario.n_tasks)])
        ft = 0.0
        up = 0.0
        for src, tgt in seq:
            c = _direct_candidate(library, src, tgt)
            ft += c.flight_steps * scenario.limits.delta_t_s
            up += c.source_data_bits / max(float(np.mean(c.full_rate_bps.max(axis=1))), 1e-9)
            ref_trace.append((int(src), int(tgt), c.candidate_id, content_hash(c)))
        max_flight = max(max_flight, ft)
        total_upload += up
    return ReferenceValues(t_ref_s=float(max_flight), d_ref_s=float(total_upload),
                           source_hash=_reference_hash(scenario, ref_trace,
                                                       float(max_flight), float(total_upload)))


def default_objective_spec(scenario, library):
    """Frozen reference + scenario weights; the single search entry point."""
    return ObjectiveSpec(reference=compute_reference(scenario, library),
                         w_t=scenario.limits.w_t, w_d=scenario.limits.w_d)
