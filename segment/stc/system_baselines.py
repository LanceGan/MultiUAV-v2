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
- ``gatsac_style``               GATSAC-style (SAC candidates + checkpoint;
                                 explicit unavailability when absent)
- ``library_open_loop``          Library-Open-Loop
- ``proposed_stc_alns``          Proposed STC-ALNS

``library`` is the plain candidate dict ``{(source, target): [SegmentCandidate, ...]}``
consumed by the evaluator / objective / solver.  The library-aware assignment
and sequencing steps wrap it in a :class:`SegmentLibrary` view (only
``.candidates`` is read), so a single library object is shared unchanged across
all six methods.  A caller may also pass a :class:`SegmentLibrary` directly.

Jia-EBTAS-style and GATSAC-style are style reproductions, not author code.
Restricted methods never substitute a different trajectory family.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

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
GRAPH_GENS = frozenset({"astar", "slsqp_refined"})
COMM_AWARE_GENS = frozenset({"comm_direct", "comm_waypoint"})


class BaselineUnavailable(RuntimeError):
    """A declared method cannot run with this scenario/library/checkpoint."""

    def __init__(self, method, reason):
        self.method = method
        self.reason = reason
        super().__init__("%s unavailable: %s" % (method, reason))


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
                "checkpoint-bound SAC trajectories (style reproduction)."),
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

    No fallback is allowed: a required family is part of the method identity.
    """
    out = []
    for src, tgt in edges:
        rankable = [c for c in library.get((src, tgt), ()) if c.rankable]
        pool = [c for c in rankable if c.generator in generator_filter]
        if not pool:
            raise ValueError("no required-family candidate for edge %r" % ((src, tgt),))
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
                               distance_nn_2opt,
                               {c.generator for cs in library.values() for c in cs}, seed)


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
                               sequencer,
                               {c.generator for cs in library.values() for c in cs}, seed)


_BUILDERS = {
    "region_distance_straight": _region_distance_straight,
    "shortest_balanced_graph": _shortest_balanced_graph,
    "jia_ebtas_style": _jia_ebtas_style,
    "gatsac_style": _gatsac_style,
}

_COMPONENTS = {
    "region_distance_straight": dict(assignment="region_partition",
                                     sequence="distance_nn_2opt"),
    "shortest_balanced_graph": dict(assignment="spatial_kmeans",
                                    sequence="distance_nn_2opt"),
    "jia_ebtas_style": dict(assignment="jia_ebtas",
                            sequence="jia_time_energy_tsp"),
    "gatsac_style": dict(assignment="spatial_kmeans",
                         sequence="library_aware_sequence"),
    "library_open_loop": dict(assignment="library_aware_ebtas",
                              sequence="library_aware_sequence"),
}

_STYLE_NOTES = {
    "jia_ebtas_style": ("Jia-EBTAS-style reproduction: preserves the Jia "
                        "time-energy internal criterion; evaluated on the "
                        "common T/D/O metrics."),
    "gatsac_style": ("GATSAC-style reproduction using checkpoint-bound SAC "
                     "trajectories; style reproduction, not author-code reproduction."),
}


def _selected_generators(solution, library, n_tasks):
    return sorted({next(c.generator for c in library[edge] if c.candidate_id == cid)
                   for route, ids in zip(solution.routes, solution.candidate_ids)
                   for edge, cid in zip(_route_edges(route, n_tasks), ids)})


def _provenance(spec, seed, solution, library, n_tasks):
    p = dict(_COMPONENTS[spec.key])
    p.update({"label": spec.label, "proposed": False, "seed": int(seed)})
    p["trajectory_generators"] = _selected_generators(solution, library, n_tasks)
    p["trajectory_layer"] = "+".join(p["trajectory_generators"]) or "idle"
    if spec.key == "gatsac_style":
        p["trajectory_note"] = _STYLE_NOTES["gatsac_style"]
    if spec.key == "jia_ebtas_style":
        p["style_note"] = _STYLE_NOTES["jia_ebtas_style"]
    return p


def validate_baseline_availability(method, scenario, library, *,
                                   checkpoint_path=None, graph_generators=None):
    """Predeclare availability before any method-seed runs; return family view.

    SAC generator labels follow ``drl:SAC:<checkpoint SHA-256>`` (also
    ``drl:learned-SAC:<hash>``). The hash must match the supplied checkpoint.
    This consumes existing learned candidates; it never trains or relabels them.
    """
    if method not in METHOD_SPECS:
        raise ValueError("unknown system baseline method %r" % method)
    candidates = library.candidates if isinstance(library, SegmentLibrary) else library
    for edge, entries in candidates.items():
        ids = [c.candidate_id for c in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate_id on edge %r" % (edge,))
    checkpoint = None
    if method == "gatsac_style":
        path = Path(checkpoint_path) if checkpoint_path else None
        if path is None or not path.is_file():
            raise BaselineUnavailable(method, "compatible SAC checkpoint file is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = {"path": str(path.resolve()), "sha256": digest}

        def allowed(c):
            parts = c.generator.split(":")
            return (len(parts) == 3 and parts[0] == "drl"
                    and parts[1].lower().replace("_", "-") in {"sac", "learned-sac"}
                    and parts[2] == digest)
    elif method == "region_distance_straight":
        allowed = lambda c: c.generator in STRAIGHT_GENS
    elif method == "shortest_balanced_graph":
        families = GRAPH_GENS if graph_generators is None else frozenset(graph_generators)
        if not families or not families <= (GRAPH_GENS | COMM_AWARE_GENS):
            raise ValueError("graph_generators must declare graph/SLSQP or labeled communication families")
        allowed = lambda c: c.generator in families
    elif method == "jia_ebtas_style":
        allowed = lambda c: c.generator in COMM_AWARE_GENS
    else:
        allowed = lambda c: True

    view = {edge: tuple(c for c in cs if c.rankable and allowed(c))
            for edge, cs in candidates.items()}
    k = scenario.n_tasks
    edges = ([(-1, t) for t in range(k)]
             + [(s, t) for s in range(k) for t in range(k + 1) if s != t])
    missing = [edge for edge in edges if not view.get(edge)]
    if missing:
        raise BaselineUnavailable(method, "required trajectory family missing on edge %r" % (missing[0],))
    return view, checkpoint


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def run_system_baseline(method, scenario, library, field, *, seed=42,
                        method_seed=42, iterations=1000, checkpoint_path=None,
                        graph_generators=None, solver_config=None):
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
    family_view, checkpoint = validate_baseline_availability(
        method, scenario, library, checkpoint_path=checkpoint_path,
        graph_generators=graph_generators)
    objective_spec = default_objective_spec(scenario, candidates)

    if spec.key == "proposed_stc_alns":
        seg = _seg_view(library, scenario, field)
        initial = build_ebtas_initial_solution(scenario, seg, seed=seed)
        sr = STCALNS(scenario, candidates, field, **(solver_config or {})).solve(
            initial, iterations=iterations, method_seed=method_seed)
        return SystemBaselineResult(
            method=spec.key, solution=sr.best_solution, result=sr.best_result,
            objective_spec=objective_spec, evaluator_calls=sr.evaluator_calls,
            provenance={"label": spec.label, "proposed": True,
                        "iterations_executed": len(sr.search_trace),
                        "trajectory_generators": _selected_generators(
                            sr.best_solution, candidates, scenario.n_tasks),
                        "initial_solution": initial})

    if spec.key == "library_open_loop":
        seg = _seg_view(library, scenario, field)
        solution = build_ebtas_initial_solution(scenario, seg, seed=seed)
    else:
        solution = _BUILDERS[spec.key](scenario, family_view, field, seed)

    result = evaluate_solution(scenario, candidates, solution, field,
                               objective_spec)
    provenance = _provenance(spec, seed, solution, candidates, scenario.n_tasks)
    if checkpoint:
        provenance["checkpoint"] = checkpoint
    return SystemBaselineResult(
        method=spec.key, solution=solution, result=result,
        objective_spec=objective_spec, evaluator_calls={"full": 1},
        provenance=provenance)
