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
        edges = ([(-1, route[0])]
                 + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
                 + [(route[-1], K)])
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

    timelines = tuple([] for _ in range(n_uav))
    objective = (limits.w_t * makespan + limits.w_d * total_delay) if feasible else float("inf")
    return EvaluationResult(
        feasible=feasible, makespan_s=makespan, total_delay_s=total_delay,
        outage_ratio=outage_ratio, violations=violations, timelines=timelines,
        objective=objective,
    )
