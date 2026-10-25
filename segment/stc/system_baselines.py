"""Complete-system baseline adapters (Task 8).

Six named end-to-end methods, all evaluated by the SAME synchronous evaluator
(:func:`segment.stc.evaluator.evaluate_solution`) from slot zero, with the same
scenario / library / reference / failure rules.  Only the Proposed STC-ALNS
method uses synchronous conflict-feedback refinement (:class:`STCALNS`); the
five baselines are isolated, single-shot methods.

Method names (comparison order):

- ``region_distance_straight``   Region-Distance-Straight
- ``shortest_balanced_graph``    Shortest-Balanced-Graph
- ``jia_ebtas_style``            Jia-EBTAS-style (style reproduction)
- ``gatsac_style``               GATSAC-style (style reproduction; the SAC
                                 trajectory layer is approximated by a
                                 deterministic communication-aware segment)
- ``library_open_loop``          Library-Open-Loop
- ``proposed_stc_alns``          Proposed STC-ALNS

``library`` is the plain candidate dict ``{(source, target): [SegmentCandidate, ...]}``
consumed by the evaluator / objective / solver.  The library-aware assignment
and sequencing steps wrap it in a :class:`SegmentLibrary` view (only
``.candidates`` is read), so a single library object is shared unchanged across
all six methods.  A caller may also pass a :class:`SegmentLibrary` directly.

Claim integrity: Jia-EBTAS-style and GATSAC-style are labeled *style
reproductions*.  No exact-author reproduction is claimed; in particular the
GATSAC SAC layer is documented as approximated by the closest deterministic
communication-aware segment optimizer, not an SAC reproduction.
"""

from dataclasses import dataclass

from segment.stc.assignment import jia_ebtas, region_partition, spatial_kmeans
from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.operators import build_ebtas_initial_solution
from segment.stc.sequencing import (
    distance_nn_2opt,
    jia_time_energy_tsp,
    library_aware_sequence,
)
from segment.stc.slsqp_refine import candidate_cost
from segment.stc.solver import STCALNS
from segment.stc.types import (
    EvaluationResult,
    ObjectiveSpec,
    PlanSolution,
    SegmentLibrary,
)

# Generator labels used to select the segment (trajectory) layer per method.
STRAIGHT_GENS = frozenset({"straight_hold"})
COMM_AWARE_GENS = frozenset({"comm_direct", "comm_waypoint", "astar", "slsqp_refined"})


@dataclass(frozen=True)
class SystemBaselineResult:
    """Result of one complete-system method, evaluated by the shared evaluator."""
    method: str
    solution: PlanSolution
    result: EvaluationResult
    objective_spec: ObjectiveSpec
    evaluator_calls: dict
    provenance: dict


@dataclass(frozen=True)
class _MethodSpec:
    key: str
    label: str
    proposed: bool
    description: str


_METHOD_SPECS = (
    _MethodSpec("region_distance_straight", "Region-Distance-Straight", False,
                "Geographic partition + distance sequence + Straight-Hold "
                "segments."),
    _MethodSpec("shortest_balanced_graph", "Shortest-Balanced-Graph", False,
                "Balanced spatial assignment + distance/2-opt sequence + "
                "graph/SLSQP communication-aware segment."),
    _MethodSpec("jia_ebtas_style", "Jia-EBTAS-style", False,
                "Jia-balanced assignment + time-energy sequence + closest "
                "deterministic communication-aware segment."),
    _MethodSpec("gatsac_style", "GATSAC-style", False,
                "Spatial k-means + communication-aware TSP ordering + "
                "deterministic communication-aware segment (SAC approximated; "
                "style reproduction)."),
    _MethodSpec("library_open_loop", "Library-Open-Loop", False,
                "Library-aware EBTAS + sequencing + best independent candidate, "
                "with no synchronous conflict feedback."),
    _MethodSpec("proposed_stc_alns", "Proposed STC-ALNS", True,
                "Same initial solution/library as Library-Open-Loop, then "
                "synchronous conflict-feedback refinement."),
)

METHOD_SPECS = {m.key: m for m in _METHOD_SPECS}
SYSTEM_BASELINE_KEYS = tuple(m.key for m in _METHOD_SPECS)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _route_edges(route, n_tasks):
    if not route:
        return ()
    return (([(-1, route[0])]
             + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
             + [(route[-1], n_tasks)]))


def _candidate_ids(library, edges, dt, generator_filter):
    """Best *rankable* candidate per edge by single-UAV cost, filtered by generator.

    Falls back to any rankable candidate when no candidate matches the filter
    (e.g. a reduced library built with Straight-Hold only).
    """
    out = []
    for src, tgt in edges:
        rankable = [c for c in library[(src, tgt)] if c.rankable]
        pool = [c for c in rankable if c.generator in generator_filter]
        if not pool:
            pool = rankable
        out.append(min(pool, key=lambda c: candidate_cost(c, dt)).candidate_id)
    return tuple(out)


def _seg_view(library, scenario, field):
    """Return a ``SegmentLibrary`` view of ``library`` (assignment/sequencing read ``.candidates``)."""
    if isinstance(library, SegmentLibrary):
        return library
    world_hash = getattr(field, "world_hash", None) or "system-baseline"
    return SegmentLibrary(candidates=dict(library), map_seed=scenario.map_seed,
                          task_seed=scenario.task_seed, world_hash=world_hash)


def _plan_from_clusters(scenario, library, clusters, sequencer, generator_filter,
                        seed):
    """Build a PlanSolution from an assignment's clusters + a per-cluster sequencer."""
    dt = scenario.limits.delta_t_s
    routes = []
    cids = []
    for cluster in clusters:
        seq = sequencer(scenario, cluster, seed)
        edges = _route_edges(seq.route, scenario.n_tasks)
        routes.append(tuple(seq.route))
        cids.append(_candidate_ids(library, edges, dt, generator_filter))
    return PlanSolution(routes=tuple(routes), candidate_ids=tuple(cids))


# --------------------------------------------------------------------------- #
# Baseline builders (isolated, single-shot; never call the proposed solver)
# --------------------------------------------------------------------------- #

def _region_distance_straight(scenario, library, field, seed):
    assign = region_partition(scenario, seed=seed)
    return _plan_from_clusters(scenario, library, assign.clusters,
                               distance_nn_2opt, STRAIGHT_GENS, seed)


def _shortest_balanced_graph(scenario, library, field, seed):
    assign = spatial_kmeans(scenario, seed=seed)
    return _plan_from_clusters(scenario, library, assign.clusters,
                               distance_nn_2opt, COMM_AWARE_GENS, seed)


def _jia_ebtas_style(scenario, library, field, seed):
    assign = jia_ebtas(scenario, field=field, seed=seed)

    def sequencer(sc, ids, sd):
        return jia_time_energy_tsp(sc, ids, seed=sd)

    return _plan_from_clusters(scenario, library, assign.clusters,
                               sequencer, COMM_AWARE_GENS, seed)


def _gatsac_style(scenario, library, field, seed):
    seg = _seg_view(library, scenario, field)
    assign = spatial_kmeans(scenario, seed=seed)

    def sequencer(sc, ids, sd):
        return library_aware_sequence(sc, seg, ids, seed=sd)

    return _plan_from_clusters(scenario, library, assign.clusters,
                               sequencer, COMM_AWARE_GENS, seed)


_BUILDERS = {
    "region_distance_straight": _region_distance_straight,
    "shortest_balanced_graph": _shortest_balanced_graph,
    "jia_ebtas_style": _jia_ebtas_style,
    "gatsac_style": _gatsac_style,
}

_COMPONENTS = {
    "region_distance_straight": dict(assignment="region_partition",
                                     sequence="distance_nn_2opt",
                                     trajectory_layer="straight_hold"),
    "shortest_balanced_graph": dict(assignment="spatial_kmeans",
                                    sequence="distance_nn_2opt",
                                    trajectory_layer="graph_slsqp_comm_aware"),
    "jia_ebtas_style": dict(assignment="jia_ebtas",
                            sequence="jia_time_energy_tsp",
                            trajectory_layer="deterministic_comm_aware"),
    "gatsac_style": dict(assignment="spatial_kmeans",
                         sequence="library_aware_sequence",
                         trajectory_layer="deterministic_comm_aware_approx"),
    "library_open_loop": dict(assignment="library_aware_ebtas",
                              sequence="library_aware_sequence",
                              trajectory_layer="best_rankable_candidate"),
}

_STYLE_NOTES = {
    "jia_ebtas_style": ("Jia-EBTAS-style reproduction: preserves the Jia "
                        "time-energy internal criterion; evaluated on the "
                        "common T/D/O metrics."),
    "gatsac_style": ("GATSAC-style reproduction: SAC local-trajectory layer "
                     "approximated by the closest deterministic "
                     "communication-aware segment optimizer; a style "
                     "reproduction, not an SAC reproduction."),
}


def _provenance(spec, seed):
    p = dict(_COMPONENTS[spec.key])
    p.update({"label": spec.label, "proposed": False, "seed": int(seed)})
    if spec.key == "gatsac_style":
        p["trajectory_note"] = _STYLE_NOTES["gatsac_style"]
    if spec.key == "jia_ebtas_style":
        p["style_note"] = _STYLE_NOTES["jia_ebtas_style"]
    return p


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def run_system_baseline(method, scenario, library, field, *, seed=42,
                        method_seed=42, iterations=1000):
    """Run one named complete-system method and return its evaluated result.

    Every method is scored through the same :func:`default_objective_spec`
    reference and the same :func:`evaluate_solution` physics path.  Only
    ``proposed_stc_alns`` invokes :class:`STCALNS`; the five baselines are
    single-shot and never touch conflict feedback.
    """
    spec = METHOD_SPECS.get(method)
    if spec is None:
        raise ValueError("unknown system baseline method %r (expected one of %r)"
                         % (method, SYSTEM_BASELINE_KEYS))

    candidates = library.candidates if isinstance(library, SegmentLibrary) else library
    objective_spec = default_objective_spec(scenario, candidates)

    if spec.key == "proposed_stc_alns":
        seg = _seg_view(library, scenario, field)
        initial = build_ebtas_initial_solution(scenario, seg, seed=seed)
        sr = STCALNS(scenario, candidates, field).solve(
            initial, iterations=iterations, method_seed=method_seed)
        return SystemBaselineResult(
            method=spec.key, solution=sr.best_solution, result=sr.best_result,
            objective_spec=objective_spec, evaluator_calls=sr.evaluator_calls,
            provenance={"label": spec.label, "proposed": True,
                        "initial_solution": initial})

    if spec.key == "library_open_loop":
        seg = _seg_view(library, scenario, field)
        solution = build_ebtas_initial_solution(scenario, seg, seed=seed)
    else:
        solution = _BUILDERS[spec.key](scenario, candidates, field, seed)

    result = evaluate_solution(scenario, candidates, solution, field,
                               objective_spec)
    return SystemBaselineResult(
        method=spec.key, solution=solution, result=result,
        objective_spec=objective_spec, evaluator_calls={"full": 1},
        provenance=_provenance(spec, seed))
