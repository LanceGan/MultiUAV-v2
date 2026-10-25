"""SLSQP continuous waypoint refinement (Task 5).

Refines a candidate's relay waypoints with a local continuous optimizer
(``scipy.optimize.minimize``, SLSQP).  If the refinement fails or worsens the
single-UAV cost, the original candidate is returned unchanged.
"""

import numpy as np

from segment.stc.generators import _build_candidate, _node_data, _node_pos, _trace


def candidate_cost(candidate, delta_t_s):
    """Single-UAV cost = flight time + upload delay (seconds)."""
    flight = candidate.flight_steps * delta_t_s
    mean_rate = float(np.mean(candidate.full_rate_bps.max(axis=1)))
    upload = candidate.source_data_bits / max(mean_rate, 1e-9)
    return flight + upload


def refine_candidate(candidate, scenario, field, max_waypoints=1):
    """Optimise the relay waypoint(s) with SLSQP; return the original on failure."""
    if candidate.generator != "comm_waypoint" or max_waypoints < 1:
        return candidate  # nothing to refine (direct / relaxation / hold)

    src = _node_pos(scenario, candidate.source)
    tgt = _node_pos(scenario, candidate.target)
    data = _node_data(scenario, candidate.source)
    limits = scenario.limits

    # initial guess: the midpoint of the source-target segment
    x0 = np.array([(src[0] + tgt[0]) / 2.0, (src[1] + tgt[1]) / 2.0])
    bounds = [(0.0, scenario.length_m), (0.0, scenario.width_m)]

    def cost(relay):
        positions, _ = _trace(scenario, field, src, tgt, data, via=(relay,))
        mean_rate = float(np.mean([
            np.max(field.sample(p).a2g_efficiency) * limits.bandwidth_hz
            for p in positions]))
        flight = float(np.linalg.norm(relay - src) + np.linalg.norm(tgt - relay)) / limits.v_max_mps
        upload = data / max(mean_rate, 1e-9)
        return flight + upload

    try:
        from scipy.optimize import minimize
        res = minimize(cost, x0, method="SLSQP", bounds=bounds,
                       options={"maxiter": 30, "ftol": 1e-6})
        if not res.success:
            return candidate
        relay_opt = res.x
        positions, hold = _trace(scenario, field, src, tgt, data, via=(relay_opt,))
        refined = _build_candidate(scenario, field, candidate.source, candidate.target,
                                   "comm_waypoint", positions, hold, data, rankable=True,
                                   candidate_id=candidate.candidate_id + "_refined")
        if candidate_cost(refined, limits.delta_t_s) <= candidate_cost(candidate, limits.delta_t_s) + 1e-6:
            return refined
        return candidate
    except Exception:
        return candidate
