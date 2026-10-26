"""Communication-aware visit sequencing (Task 4).

Four single-UAV sequence solvers over one assigned task cluster, each returning
a :class:`SequenceResult` (``route``, ``selected_candidate_ids``, ``proxy_cost``,
``method``):

- ``distance_nn_2opt`` -- pure Euclidean nearest-neighbour + 2-opt (structural
  baseline, no communication awareness).
- ``jia_time_energy_tsp`` -- deterministic NN + 2-opt over the legacy Jia
  time-energy directed edge weights (``sequence.radio_path.edge_cost``).
- ``ga_eqtsp_sequence`` -- adapter of ``sequence.ga_eqtsp.ga_eqtsp`` (GA solver
  over the same communication-aware directed weights).
- ``library_aware_sequence`` -- NN + 2-opt over a frozen single-UAV proxy
  derived from the *best rankable* candidate per directed edge.

Units: SI throughout (metres / seconds / bits); the legacy 100-m / Mbit units
are converted only inside the Jia / GA-EQTSP adapters (at the scenario/radio
boundary).  Directed source-payload semantics: edge ``i -> j`` carries ``I_i``;
depot edges carry zero; terminal edges carry the last task's payload.

``proxy_cost`` is each method's *own* ordering criterion, not the synchronous
evaluator -- it is a frozen single-UAV proxy used for provenance and initial
ordering only.  None of the baseline solvers here uses proposed multi-UAV
conflict feedback; the authoritative ranking is done later by the synchronous
evaluator.
"""

from dataclasses import dataclass, field

import numpy as np

from segment.stc.slsqp_refine import candidate_cost

# Legacy Jia time-energy edge-weight defaults (mirror ``configs/scenario_uav*.json``).
DEFAULT_JIA_WEIGHTS = {
    "lambda_E": 0.0002,
    "lambda_out": 3.0,
    "lambda_tau": 2.0,
    "tau_max": 12.0,
}


@dataclass
class SequenceResult:
    route: tuple                # tuple of task ids (0..K-1) in visit order
    selected_candidate_ids: tuple  # one candidate id per traversed edge (library-aware only)
    proxy_cost: float           # the method's own ordering criterion value
    method: str                 # method name
    provenance: dict = field(default_factory=dict)  # resolved weights / seeds / edge weights


# --------------------------------------------------------------------------- #
# Library-aware directed edge proxy
# --------------------------------------------------------------------------- #

def _best_rankable(library, src, tgt, delta_t_s):
    """Best (min-cost) rankable candidate for directed edge ``(src, tgt)``.

    Returns ``(cost, candidate_id)``.  Non-rankable relaxation candidates are
    excluded -- they are lower bounds, not selectable trajectories.
    """
    rankable = [c for c in library.candidates.get((src, tgt), ()) if c.rankable]
    if not rankable:
        raise ValueError(
            "no rankable candidate for edge (%r, %r)" % (src, tgt))
    best = min(rankable, key=lambda c: candidate_cost(c, delta_t_s))
    return candidate_cost(best, delta_t_s), best.candidate_id


def edge_proxy(library, src, tgt, delta_t_s):
    """Frozen single-UAV proxy cost (flight time + upload delay) in seconds."""
    cost, _ = _best_rankable(library, src, tgt, delta_t_s)
    return cost


# --------------------------------------------------------------------------- #
# Directed NN + 2-opt over a generic directed cost function
# --------------------------------------------------------------------------- #

def _node_position(scenario, node):
    if node == -1:
        return np.asarray(scenario.depot_m, dtype=float)
    if node == scenario.n_tasks:
        return np.asarray(scenario.terminal_m, dtype=float)
    return np.asarray(scenario.task_positions_m[node], dtype=float)


def _route_edges(route, n_tasks):
    if not route:
        return []
    return (([(-1, route[0])]
             + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
             + [(route[-1], n_tasks)]))


def _nn_2opt_order(task_ids, n_tasks, cost_fn):
    """Nearest-neighbour construction + directed 2-opt over ``cost_fn(src, tgt)``.

    ``cost_fn`` is a directed cost in node encoding (-1 = depot, 0..K-1 = task,
    K = terminal).  The 2-opt reversal accounts for the directed (asymmetric)
    edge costs, so reversing a sub-path re-prices its internal edges.
    """
    remaining = list(task_ids)
    order = []
    cur = -1
    while remaining:
        nxt = min(remaining, key=lambda j: cost_fn(cur, j))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    seq = [-1] + order + [n_tasks]
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        bi = bj = -1
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                a, b = seq[i - 1], seq[i]
                c, d = seq[j], seq[j + 1]
                old = cost_fn(a, b) + cost_fn(c, d)
                new = cost_fn(a, c) + cost_fn(b, d)
                internal_old = sum(cost_fn(seq[k], seq[k + 1]) for k in range(i, j))
                internal_new = sum(cost_fn(seq[k + 1], seq[k]) for k in range(i, j))
                delta = (new - old) + (internal_new - internal_old)
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    bi, bj = i, j
        if bi != -1:
            seq[bi:bj + 1] = reversed(seq[bi:bj + 1])
            improved = True
    return tuple(seq[1:-1])


# --------------------------------------------------------------------------- #
# Distance NN + 2-opt (structural baseline)
# --------------------------------------------------------------------------- #

def distance_nn_2opt(scenario, task_ids, seed=42):
    """Pure Euclidean NN + 2-opt.  No communication awareness, no library."""
    def cost_fn(s, t):
        return float(np.linalg.norm(_node_position(scenario, t)
                                    - _node_position(scenario, s)))

    route = _nn_2opt_order(list(task_ids), scenario.n_tasks, cost_fn)
    edges = _route_edges(route, scenario.n_tasks)
    proxy = float(sum(cost_fn(s, t) for s, t in edges))
    return SequenceResult(
        route=tuple(route), selected_candidate_ids=(),
        proxy_cost=proxy, method="distance_nn_2opt",
        provenance={
            "method": "distance_nn_2opt", "seed": int(seed),
            "n_tasks": scenario.n_tasks, "proxy_units": "metres",
            "edge_lengths_m": tuple(float(cost_fn(s, t)) for s, t in edges),
        })


# --------------------------------------------------------------------------- #
# Jia Time-Energy / GA-EQTSP adapters (wrap legacy sequence/* unchanged)
# --------------------------------------------------------------------------- #

def _legacy_cfg(scenario, weights, seed):
    """Resolve the legacy 100-m/Mbit config dict from the SI scenario + weights."""
    w = dict(DEFAULT_JIA_WEIGHTS, **(weights or {}))
    return {
        "v_max": scenario.limits.v_max_mps / 100.0,
        "tau_max": float(w["tau_max"]),
        "lambda_E": float(w["lambda_E"]),
        "lambda_out": float(w["lambda_out"]),
        "lambda_tau": float(w["lambda_tau"]),
        "uav_h": scenario.limits.uav_height_m / 100.0,
        "seed": int(seed),
    }


def _configure_maps(scenario):
    """Configure the legacy radio maps for the STC region; return the world hash."""
    from segment.stc.radio import configure_radio_world
    return configure_radio_world(max(scenario.length_m, scenario.width_m),
                                 scenario.n_bs, scenario.map_seed,
                                 uplink_power_w=scenario.limits.uplink_power_w)


def _jia_comm_matrix(scenario, task_ids, weights, seed):
    """Directed time-energy edge-weight matrix over [depot] + tasks + [terminal].

    Node labels ``(-1, *task_ids, n_tasks)`` index the matrix rows/columns.  Each
    directed entry is the legacy ``radio_path.edge_cost`` for that edge (with the
    source's payload on task->* edges and zero on depot edges).
    """
    from sequence import radio_path
    cfg = _legacy_cfg(scenario, weights, seed)
    world_hash = _configure_maps(scenario)

    nodes = [-1] + list(task_ids) + [scenario.n_tasks]
    n = len(nodes)
    mat = np.full((n, n), np.inf)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ds = (scenario.task_data_bits[nodes[a]] / 1e6
                  if 0 <= nodes[a] < scenario.n_tasks else 0.0)
            mat[a, b] = radio_path.edge_cost(
                _node_position(scenario, nodes[a]) / 100.0,
                _node_position(scenario, nodes[b]) / 100.0,
                ds, cfg)["cost"]
    return mat, tuple(nodes), world_hash, cfg


def _proxy_from_matrix(mat, idx, route, n_tasks):
    """Sum of directed matrix entries along [depot] + route + [terminal]."""
    seq = [-1] + list(route) + [n_tasks]
    return float(sum(mat[idx[s], idx[t]]
                     for s, t in zip(seq[:-1], seq[1:])))


def jia_time_energy_tsp(scenario, task_ids, weights=None, seed=42):
    """Deterministic NN + 2-opt over the legacy Jia time-energy directed weights."""
    task_ids = list(task_ids)
    mat, nodes, world_hash, cfg = _jia_comm_matrix(scenario, task_ids, weights, seed)
    idx = {node: a for a, node in enumerate(nodes)}
    cost_fn = lambda s, t: mat[idx[s], idx[t]]

    route = _nn_2opt_order(task_ids, scenario.n_tasks, cost_fn)
    proxy = _proxy_from_matrix(mat, idx, route, scenario.n_tasks)
    return SequenceResult(
        route=tuple(route), selected_candidate_ids=(),
        proxy_cost=proxy, method="jia_time_energy_tsp",
        provenance={
            "method": "jia_time_energy_tsp", "seed": int(seed),
            "weights": dict(DEFAULT_JIA_WEIGHTS, **(weights or {})),
            "world_hash": world_hash, "node_labels": tuple(nodes),
            "edge_weights": mat, "proxy_units": "legacy_comm_cost",
        })


def ga_eqtsp_sequence(scenario, task_ids, weights=None, seed=42,
                     num_total=25, iteration=200):
    """Adapter of ``sequence.ga_eqtsp.ga_eqtsp`` (GA over comm-aware directed weights).

    The legacy GA internal criterion is unchanged; the resolved edge weights and
    seeds are recorded for provenance.
    """
    from sequence import ga_eqtsp as _ga_eqtsp
    task_ids = list(task_ids)
    mat, nodes, world_hash, cfg = _jia_comm_matrix(scenario, task_ids, weights, seed)

    cluster_100 = (np.asarray([scenario.task_positions_m[t] for t in task_ids])
                   / 100.0)
    data_mbit = (np.asarray([scenario.task_data_bits[t] for t in task_ids]) / 1e6)
    ini_100 = np.asarray(scenario.depot_m) / 100.0
    end_100 = np.asarray(scenario.terminal_m) / 100.0

    _, order_idx, cost = _ga_eqtsp.ga_eqtsp(
        cluster_100, ini_100, end_100, data_mbit, cfg,
        num_total=num_total, iteration=iteration)

    route = tuple(int(task_ids[i]) for i in order_idx)
    return SequenceResult(
        route=route, selected_candidate_ids=(),
        proxy_cost=float(cost), method="ga_eqtsp",
        provenance={
            "method": "ga_eqtsp", "seed": int(seed),
            "weights": dict(DEFAULT_JIA_WEIGHTS, **(weights or {})),
            "world_hash": world_hash, "node_labels": tuple(nodes),
            "edge_weights": mat, "num_total": int(num_total),
            "iteration": int(iteration), "proxy_units": "legacy_comm_cost",
        })


# --------------------------------------------------------------------------- #
# Library-aware sequence
# --------------------------------------------------------------------------- #

def library_aware_sequence(scenario, library, task_ids, seed=42):
    """NN + 2-opt over a frozen single-UAV proxy from the best rankable candidate.

    For each directed edge the proxy is the best rankable candidate's
    flight-time + upload-delay cost; the selected candidate ids are recorded
    alongside the route.  No multi-UAV conflict feedback is used.
    """
    dt = scenario.limits.delta_t_s
    task_ids = list(task_ids)

    def cost_fn(s, t):
        return edge_proxy(library, s, t, dt)

    route = _nn_2opt_order(task_ids, scenario.n_tasks, cost_fn)
    edges = _route_edges(route, scenario.n_tasks)
    selected = []
    proxy = 0.0
    edge_costs = []
    for s, t in edges:
        c, cid = _best_rankable(library, s, t, dt)
        selected.append(cid)
        edge_costs.append(c)
        proxy += c

    return SequenceResult(
        route=tuple(route), selected_candidate_ids=tuple(selected),
        proxy_cost=float(proxy), method="library_aware",
        provenance={
            "method": "library_aware", "seed": int(seed),
            "n_tasks": scenario.n_tasks, "proxy_units": "seconds",
            "edge_proxies_s": tuple(float(c) for c in edge_costs),
        })
