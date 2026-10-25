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
"""

import hashlib

import numpy as np

from segment.stc.types import (
    EvaluationResult,
    ObjectiveSpec,
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


def evaluate_solution(scenario, library, solution, field, objective_spec,
                      keep_timelines=True):
    _require_objective_spec(objective_spec)
    limits = scenario.limits
    n_uav = scenario.n_uav
    K = scenario.n_tasks
    dt = limits.delta_t_s
    max_slots = int(limits.mission_deadline_s / dt) + 1

    # Build per-UAV edge + candidate sequences.
    uav_cands = []
    uav_edges = []
    for u in range(n_uav):
        route = solution.routes[u]
        cids = solution.candidate_ids[u]
        if route:
            edges = ([(-1, route[0])]
                     + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
                     + [(route[-1], K)])
        else:
            edges = [(-1, K)]  # empty route: depot -> terminal directly
        seq = []
        for e, (src, tgt) in enumerate(edges):
            cand = next(c for c in library[(src, tgt)] if c.candidate_id == cids[e])
            seq.append(cand)
        uav_cands.append(seq)
        uav_edges.append(edges)

    airborne = [False] * n_uav
    done = [False] * n_uav
    edge = [0] * n_uav
    tidx = [0] * n_uav
    buf = [0.0] * n_uav
    # departure_steps are in departure_step_s units; convert to slot indices
    departure_slots = [int(round(solution.departure_steps[u] * limits.departure_step_s / dt))
                       for u in range(n_uav)]
    completion = [None] * n_uav
    edge_hold = [0.0] * n_uav
    outage_time = [0.0] * n_uav
    air_time = [0.0] * n_uav
    upload_time = [0.0] * n_uav
    slots_by_uav = [[] for _ in range(n_uav)]

    unserved = K
    deadline_exceeded = 0
    wait_exceeded = 0

    for n in range(max_slots):
        t_start = n * dt
        t_end = (n + 1) * dt

        # 1. activate due UAVs
        for u in range(n_uav):
            if not airborne[u] and not done[u] and departure_slots[u] <= n:
                airborne[u] = True

        # pre-slot buffer (buffer at the start of this slot, before upload)
        pre_buf = [buf[u] for u in range(n_uav)]

        # 2. advance / hold
        holding = [False] * n_uav
        for u in range(n_uav):
            if airborne[u] and not done[u]:
                cand = uav_cands[u][edge[u]]
                L = len(cand.positions_m)
                if tidx[u] < L - 2:
                    tidx[u] += 1
                elif tidx[u] == L - 2:
                    if pre_buf[u] <= 0.0:
                        tidx[u] += 1          # enter the service anchor
                    else:
                        holding[u] = True     # wait outside the arrival radius
                        edge_hold[u] += dt
                        if edge_hold[u] > limits.max_wait_s + 1e-9:
                            wait_exceeded += 1

        # 3. associate sectors + accrue outage over the full airborne time
        sectors = [None] * n_uav
        slot_outage = [0.0] * n_uav
        for u in range(n_uav):
            if airborne[u] and not done[u]:
                cand = uav_cands[u][edge[u]]
                p = cand.positions_m[tidx[u]]
                s = field.sample(p)
                sec = select_serving_sector(s, pre_buf[u] > 0.0, limits.eps_out)
                sectors[u] = sec
                slot_outage[u] = float(s.g2a_outage[sec])
                outage_time[u] += slot_outage[u] * dt
                air_time[u] += dt

        # 4. active uploaders (pre-slot buffer) + bandwidth split
        active = [airborne[u] and not done[u] and pre_buf[u] > 0.0
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
            if airborne[u] and not done[u]:
                cand = uav_cands[u][edge[u]]
                sec = sectors[u]
                rate_full[u] = float(cand.full_rate_bps[tidx[u]][sec])
                if active[u]:
                    alloc[u] = limits.bandwidth_hz / co[sec]
                    rate = rate_full[u] * (alloc[u] / limits.bandwidth_hz)
                    up = min(pre_buf[u], rate * dt)
                    uploaded[u] = up
                    buf[u] = pre_buf[u] - up
                    upload_time[u] += dt
                else:
                    buf[u] = pre_buf[u]  # == 0 (buffers never go negative)
                post_buf[u] = buf[u]

        # service event (after upload/motion); applied below
        service_event = [""] * n_uav
        for u in range(n_uav):
            if airborne[u] and not done[u]:
                cand = uav_cands[u][edge[u]]
                if tidx[u] == len(cand.positions_m) - 1 and post_buf[u] <= 0.0:
                    _, tgt = uav_edges[u][edge[u]]
                    service_event[u] = "complete" if tgt == K else "serve"

        # record typed TimelineSlot
        if keep_timelines:
            for u in range(n_uav):
                if airborne[u] and not done[u]:
                    cand = uav_cands[u][edge[u]]
                    p = cand.positions_m[tidx[u]]
                    slots_by_uav[u].append(TimelineSlot(
                        time_start_s=t_start,
                        time_end_s=t_end,
                        edge_id=edge[u],
                        candidate_id=cand.candidate_id,
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
            if airborne[u] and not done[u] and service_event[u]:
                _, tgt = uav_edges[u][edge[u]]
                edge_hold[u] = 0.0
                if tgt == K:
                    done[u] = True
                    completion[u] = t_end
                else:
                    unserved -= 1
                    buf[u] = float(scenario.task_data_bits[tgt])
                    edge[u] += 1
                    tidx[u] = 0

    # post-loop failure accounting (deadline with residual buffer)
    upload_failures = 0
    for u in range(n_uav):
        if airborne[u] and not done[u]:
            deadline_exceeded += 1
            if buf[u] > 0.0:
                upload_failures += 1

    outage_ratio = tuple(ot / max(at, 1e-9) for ot, at in zip(outage_time, air_time))
    feasible = (unserved == 0 and upload_failures == 0 and deadline_exceeded == 0
                and wait_exceeded == 0
                and all(c is not None for c in completion)
                and all(r <= limits.eps_out for r in outage_ratio))
    makespan = max((c for c in completion if c is not None), default=0.0)
    total_delay = sum(upload_time)
    violations = ViolationVector(
        unserved_tasks=unserved,
        upload_failures=upload_failures,
        outage_excess=max(0.0, max(outage_ratio) - limits.eps_out) if outage_ratio else 0.0,
        deadline_exceeded=deadline_exceeded,
        wait_exceeded=wait_exceeded,
    )

    if keep_timelines:
        timelines = tuple(UAVTimeline(uav_id=u, slots=tuple(slots_by_uav[u]))
                          for u in range(n_uav))
    else:
        timelines = ()

    objective = float("inf") if not feasible else objective_spec.score(makespan, total_delay)
    return EvaluationResult(
        feasible=feasible, makespan_s=makespan, total_delay_s=total_delay,
        outage_ratio=outage_ratio, violations=violations, timelines=timelines,
        objective=objective,
    )


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


def _reference_hash(scenario, ref_trace):
    h = hashlib.sha256()

    def add(obj):
        h.update(repr(obj).encode("utf-8"))

    add(scenario.n_uav); add(scenario.n_tasks)
    add(scenario.map_seed); add(scenario.task_seed)
    add(scenario.depot_m); add(scenario.terminal_m)
    add(tuple(scenario.task_positions_m)); add(tuple(scenario.task_data_bits))
    lim = scenario.limits
    add((lim.delta_t_s, lim.v_max_mps, lim.arrive_radius_m, lim.bandwidth_hz,
         lim.departure_step_s, lim.departure_max_s))
    add(tuple(ref_trace))
    return h.hexdigest()


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
            ref_trace.append((src, tgt, c.candidate_id))
        max_flight = max(max_flight, ft)
        total_upload += up
    return ReferenceValues(t_ref_s=float(max_flight), d_ref_s=float(total_upload),
                           source_hash=_reference_hash(scenario, ref_trace))


def default_objective_spec(scenario, library):
    """Frozen reference + scenario weights; the single search entry point."""
    return ObjectiveSpec(reference=compute_reference(scenario, library),
                         w_t=scenario.limits.w_t, w_d=scenario.limits.w_d)
