"""Grid A*/Dijkstra candidate generation (Task 5).

Searches a discrete communication-cost graph (grid cells, edge cost = flight
time + upload penalty) and returns the shortest path as a candidate.  The result
is optimal *on this grid*, not a continuous-space global optimum.
"""

import heapq

import numpy as np

from segment.stc.generators import _build_candidate, _node_data, _node_pos, _trace


def astar_candidates(scenario, field, source, target, grid=10):
    src = _node_pos(scenario, source)
    tgt = _node_pos(scenario, target)
    data = _node_data(scenario, source)
    limits = scenario.limits

    xs = np.linspace(0.0, scenario.length_m, grid)
    ys = np.linspace(0.0, scenario.width_m, grid)

    def rate_at(i, j):
        s = field.sample(np.array([xs[i], ys[j]]))
        return max(float(np.max(s.a2g_efficiency)) * limits.bandwidth_hz, 1e-9)

    def edge_cost(i, j, ni, nj):
        dist = float(np.hypot(xs[ni] - xs[i], ys[nj] - ys[j]))
        flight = dist / limits.v_max_mps
        return flight + 0.1 * data / rate_at(ni, nj)

    si = int(np.argmin(np.abs(xs - src[0])))
    sj = int(np.argmin(np.abs(ys - src[1])))
    ti = int(np.argmin(np.abs(xs - tgt[0])))
    tj = int(np.argmin(np.abs(ys - tgt[1])))

    dist = {(si, sj): 0.0}
    prev = {}
    pq = [(0.0, si, sj)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist.get((i, j), float("inf")):
            continue
        if (i, j) == (ti, tj):
            break
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < grid and 0 <= nj < grid):
                continue
            nd = d + edge_cost(i, j, ni, nj)
            if nd < dist.get((ni, nj), float("inf")):
                dist[(ni, nj)] = nd
                prev[(ni, nj)] = (i, j)
                heapq.heappush(pq, (nd, ni, nj))

    path = [(ti, tj)]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    path.reverse()
    waypoints = [np.array([xs[i], ys[j]]) for (i, j) in path[1:-1]]

    positions, hold = _trace(scenario, field, src, tgt, data, via=waypoints)
    return _build_candidate(scenario, field, source, target, "astar",
                            positions, hold, data, rankable=True)
