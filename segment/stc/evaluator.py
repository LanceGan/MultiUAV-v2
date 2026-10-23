"""Slot-synchronous shared-bandwidth state machine (Task 6).

The authoritative evaluator: every formal metric comes from here.  Per slot it
activates due UAVs, advances each along its selected candidate trace, holds at
the arrival boundary until the buffer drains, associates one serving sector per
airborne UAV, splits each sector's bandwidth among *active uploaders* (non-empty
buffers only), updates caches, and triggers service events at the anchors.
"""

import numpy as np

from segment.stc.types import EvaluationResult, UAVTimeline, ViolationVector


def select_serving_sector(sample, has_payload, reliability_limit):
    """Serving sector per §5.5: reliable-then-rate when uploading, min-outage otherwise."""
    if has_payload:
        ok = sample.g2a_outage <= reliability_limit
        if np.any(ok):
            return int(np.argmax(np.where(ok, sample.a2g_efficiency, -np.inf)))
        return int(np.argmin(sample.g2a_outage))
    return int(np.argmin(sample.g2a_outage))


def evaluate_solution(scenario, library, solution, field, keep_timelines=True):
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
    completion = [None] * n_uav
    hold_time = [0.0] * n_uav
    edge_hold = [0.0] * n_uav
    outage_time = [0.0] * n_uav
    air_time = [0.0] * n_uav
    upload_time = [0.0] * n_uav
    timelines = [[] for _ in range(n_uav)]

    unserved = K
    deadline_exceeded = 0
    wait_exceeded = 0

    for n in range(max_slots):
        # 1. activate
        for u in range(n_uav):
            if not airborne[u] and not done[u] and solution.departure_steps[u] <= n:
                airborne[u] = True

        # 2. advance / hold
        for u in range(n_uav):
            if airborne[u] and not done[u]:
                pos = uav_cands[u][edge[u]].positions_m
                if tidx[u] < len(pos) - 2:
                    tidx[u] += 1
                elif tidx[u] == len(pos) - 2:
                    if buf[u] <= 0.0:
                        tidx[u] += 1
                    else:
                        hold_time[u] += dt
                        edge_hold[u] += dt
                        if edge_hold[u] > limits.max_wait_s + 1e-9:
                            wait_exceeded += 1

        # 3. associate sectors + outage
        sectors = [None] * n_uav
        for u in range(n_uav):
            if airborne[u] and not done[u]:
                cand = uav_cands[u][edge[u]]
                p = cand.positions_m[tidx[u]]
                s = field.sample(p)
                sec = select_serving_sector(s, buf[u] > 0.0, limits.eps_out)
                sectors[u] = sec
                outage_time[u] += s.g2a_outage[sec] * dt
                air_time[u] += dt

        # 4. active uploaders + bandwidth split
        active = [airborne[u] and not done[u] and buf[u] > 0.0 for u in range(n_uav)]
        co = {}
        for u in range(n_uav):
            if active[u]:
                co[sectors[u]] = co.get(sectors[u], 0) + 1

        # 5. update cache
        for u in range(n_uav):
            if active[u]:
                cand = uav_cands[u][edge[u]]
                rate = cand.full_rate_bps[tidx[u]][sectors[u]] / co[sectors[u]]
                buf[u] = max(0.0, buf[u] - rate * dt)
                upload_time[u] += dt

        # record per-slot state for conflict attribution: (time, sector, buffer, alloc, full_rate)
        for u in range(n_uav):
            if airborne[u] and not done[u]:
                cand = uav_cands[u][edge[u]]
                fr = cand.full_rate_bps[tidx[u]]
                sec = sectors[u]
                alloc = limits.bandwidth_hz / co.get(sec, 1) if active[u] else 0.0
                timelines[u].append((n * dt, sec, buf[u], alloc, float(fr[sec])))

        # 6. service events
        for u in range(n_uav):
            if airborne[u] and not done[u]:
                cand = uav_cands[u][edge[u]]
                if tidx[u] == len(cand.positions_m) - 1 and buf[u] <= 0.0:
                    _, tgt = uav_edges[u][edge[u]]
                    edge_hold[u] = 0.0
                    if tgt == K:
                        done[u] = True
                        completion[u] = n * dt
                    else:
                        unserved -= 1
                        buf[u] = scenario.task_data_bits[tgt]
                        edge[u] += 1
                        tidx[u] = 0

    for u in range(n_uav):
        if airborne[u] and not done[u]:
            deadline_exceeded += 1

    outage_ratio = tuple(ot / max(at, 1e-9) for ot, at in zip(outage_time, air_time))
    feasible = (unserved == 0 and deadline_exceeded == 0 and wait_exceeded == 0
                and all(c is not None for c in completion)
                and all(r <= limits.eps_out for r in outage_ratio))
    makespan = max((c for c in completion if c is not None), default=0.0)
    total_delay = sum(upload_time)
    violations = ViolationVector(
        unserved_tasks=unserved,
        upload_failures=0,
        outage_excess=max(0.0, max(outage_ratio) - limits.eps_out) if outage_ratio else 0.0,
        deadline_exceeded=deadline_exceeded,
        wait_exceeded=wait_exceeded,
    )

    timelines = tuple(tuple(t) for t in timelines)
    objective = (limits.w_t * makespan + limits.w_d * total_delay) if feasible else float("inf")
    return EvaluationResult(
        feasible=feasible, makespan_s=makespan, total_delay_s=total_delay,
        outage_ratio=outage_ratio, violations=violations, timelines=timelines,
        objective=objective,
    )


def objective(result, t_ref, d_ref, w_t, w_d):
    """Normalized objective ``w_t*T/T_ref + w_d*D/D_ref``."""
    return w_t * result.makespan_s / t_ref + w_d * result.total_delay_s / d_ref


def feasibility_rank(result):
    """Lexicographic feasibility rank (lower is better), §7.2."""
    v = result.violations
    return (v.unserved_tasks, v.upload_failures, v.outage_excess,
            v.deadline_exceeded, v.wait_exceeded, result.objective)


def _direct_candidate(library, src, tgt):
    from segment.stc.slsqp_refine import candidate_cost
    cands = library[(src, tgt)]
    return min(cands, key=lambda c: candidate_cost(c, 0.5))


def compute_reference(scenario, library):
    """Reference (T_ref, D_ref) from a greedy NN route split into n_uav groups."""
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
    for g in np.array_split(order, scenario.n_uav):
        g = list(g)
        if not g:
            continue
        seq = [(-1, g[0])] + [(g[i], g[i + 1]) for i in range(len(g) - 1)] + [(g[-1], scenario.n_tasks)]
        ft = 0.0
        up = 0.0
        for src, tgt in seq:
            c = _direct_candidate(library, src, tgt)
            ft += c.flight_steps * scenario.limits.delta_t_s
            up += c.source_data_bits / max(float(np.mean(c.full_rate_bps.max(axis=1))), 1e-9)
        max_flight = max(max_flight, ft)
        total_upload += up
    return float(max_flight), float(total_upload)
