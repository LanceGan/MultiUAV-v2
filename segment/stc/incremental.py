"""Exact affected-window incremental evaluation (Task 6 Step 4).

``EvaluationCache`` caches the committed incumbent's per-slot simulation state
so a local move can resume the authoritative slot machine from the earliest
affected slot instead of slot zero.  The incremental result equals a full
replay exactly (within float tolerance), and a periodic full-replay audit
detects any divergence.  Full and incremental calls are counted honestly:
``full_calls`` counts slot-zero replays only, ``incremental_calls`` counts
resume-from-later-slot replays, and ``recomputed_slots`` counts the slots
actually re-simulated.
"""

import numpy as np

from segment.stc.evaluator import (
    _init_state,
    evaluate_solution,
    evaluate_with_snapshots,
    simulate_from_state,
)


class IncrementalDivergenceError(RuntimeError):
    """Raised when a periodic full replay disagrees with the incremental result."""


def _route_edges(route, n_tasks):
    if not route:
        return []
    return ([(-1, route[0])]
            + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            + [(route[-1], n_tasks)])


def solution_diff(base, new, n_tasks):
    """Return ``(affected_uavs, per_uav_earliest)`` for ``base -> new``.

    ``affected_uavs`` is a frozenset of UAV indices whose route or candidate
    sequence changed; ``per_uav_earliest`` maps a UAV to the smallest route-edge
    index at which it first differs (``None`` for unchanged UAVs).
    """
    affected = set()
    earliest = {}
    for u in range(len(base.routes)):
        br, bc = base.routes[u], base.candidate_ids[u]
        nr, nc = new.routes[u], new.candidate_ids[u]
        if br == nr and bc == nc:
            continue
        affected.add(u)
        eb = _route_edges(br, n_tasks)
        en = _route_edges(nr, n_tasks)
        e = 0
        while e < len(eb) and e < len(en):
            if eb[e] != en[e] or bc[e] != nc[e]:
                break
            e += 1
        earliest[u] = e
    return frozenset(affected), earliest


def _edge_start_slot(timeline, edge_index):
    for slot in timeline.slots:
        if slot.edge_id == edge_index:
            return slot.time_start_s
    return None


def _results_match(a, b, tol=1e-6):
    if a.feasible != b.feasible:
        return False
    if abs(a.makespan_s - b.makespan_s) > tol:
        return False
    if abs(a.total_delay_s - b.total_delay_s) > tol:
        return False
    # per-UAV G2A outage ratio (a divergence here must not go undetected)
    if len(a.outage_ratio) != len(b.outage_ratio):
        return False
    for x, y in zip(a.outage_ratio, b.outage_ratio):
        if abs(x - y) > tol:
            return False
    # full violation vector
    va, vb = a.violations, b.violations
    if va.unserved_tasks != vb.unserved_tasks:
        return False
    if va.upload_failures != vb.upload_failures:
        return False
    if abs(va.outage_excess - vb.outage_excess) > tol:
        return False
    if va.deadline_exceeded != vb.deadline_exceeded:
        return False
    if va.wait_exceeded != vb.wait_exceeded:
        return False
    return True


class EvaluationCache:
    def __init__(self, scenario, library, field, base_solution, base_result,
                 objective_spec, snapshots, audit_interval=None):
        self.scenario = scenario
        self.library = library
        self.field = field
        self.objective_spec = objective_spec
        self.base_solution = base_solution
        self.base_result = base_result
        self.snapshots = snapshots          # slot_index -> SimulationState
        self.audit_interval = audit_interval
        self.dt = scenario.limits.delta_t_s
        self.max_slots = int(scenario.limits.mission_deadline_s / self.dt) + 1
        self.full_calls = 1
        self.incremental_calls = 0
        self.recomputed_slots = 0
        self.audits = 0
        self._pending = None                # (solution, result, snapshots)

    @classmethod
    def from_full(cls, scenario, library, field, solution, objective_spec,
                  audit_interval=None):
        result, snapshots = evaluate_with_snapshots(
            scenario, library, solution, field, objective_spec)
        return cls(scenario, library, field, solution, result, objective_spec,
                   snapshots, audit_interval=audit_interval)

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #

    def _resume_slot(self, per_uav_earliest):
        """Smallest slot index at which any affected UAV starts a changed edge."""
        t0 = None
        for u, e in per_uav_earliest.items():
            if e == 0:
                return 0
            timeline = self.base_result.timelines[u]
            start_s = _edge_start_slot(timeline, e)
            if start_s is None:
                return 0
            slot = int(round(start_s / self.dt))
            t0 = slot if t0 is None else min(t0, slot)
        return 0 if t0 is None else t0

    def _resume_state(self, slot):
        base = self.snapshots[slot].light_snapshot(slot)
        # reconstruct the recorded timeline prefix (< slot) from the base result
        t0 = slot * self.dt
        base.slots_by_uav = [
            [s for s in tl.slots if s.time_start_s < t0 - 1e-9]
            for tl in self.base_result.timelines]
        return base

    def evaluate_after_move(self, solution, affected_uavs=None,
                            earliest_changed_edge=None):
        """Incrementally evaluate ``solution`` against the committed incumbent.

        ``affected_uavs`` / ``earliest_changed_edge`` are accepted for API
        compatibility; the affected set and per-UAV earliest changed edge are
        re-derived from the committed incumbent so the resume point is exact.
        """
        affected, per_uav_earliest = solution_diff(
            self.base_solution, solution, self.scenario.n_tasks)

        if not affected:
            # no change: honest cache hit, zero slots recomputed.  Set the
            # pending slot so a later commit() of this no-op does not resurrect
            # a previously rejected move's state.
            self._pending = (solution, self.base_result, dict(self.snapshots))
            return self.base_result

        slot = self._resume_slot(per_uav_earliest)
        if slot == 0:
            # the earliest change is at the very start: a fresh full replay
            # (first edge changed / task moved onto a previously-idle UAV).
            # This is a full evaluation, not an incremental resume.
            self.full_calls += 1
            state = _init_state(self.scenario, solution)
        else:
            self.incremental_calls += 1
            state = self._resume_state(slot)
        new_snapshots = {}

        def capture(n, st):
            new_snapshots[n] = st.light_snapshot(n)

        result = simulate_from_state(
            self.scenario, self.library, solution, self.field,
            self.objective_spec, state, keep_timelines=True, capture=capture)
        self.recomputed_slots += (self.max_slots - slot)
        # merge the recomputed suffix with the still-valid unchanged prefix
        merged = dict(self.snapshots)
        merged.update(new_snapshots)
        self._pending = (solution, result, merged)

        if self.audit_interval and self.incremental_calls % self.audit_interval == 0:
            self._audit(solution, result)
        return result

    def _audit(self, solution, incremental_result):
        """Full replay of ``solution`` and compare against the incremental result."""
        self.full_calls += 1
        self.audits += 1
        full = evaluate_solution(self.scenario, self.library, solution,
                                 self.field, self.objective_spec)
        if not _results_match(full, incremental_result):
            raise IncrementalDivergenceError(
                "incremental result diverged from full replay: "
                "full=(T=%r, D=%r, feasible=%r) incremental=(T=%r, D=%r, feasible=%r)"
                % (full.makespan_s, full.total_delay_s, full.feasible,
                   incremental_result.makespan_s, incremental_result.total_delay_s,
                   incremental_result.feasible))

    def commit(self, solution=None, result=None):
        """Advance the committed incumbent to the last evaluated (accepted) move.

        Verifies the pending candidate actually matches ``solution`` so a no-op
        commit (candidate == incumbent) can never resurrect a previously
        rejected move's base state.
        """
        if solution is None:
            if self._pending is None:
                return
            solution, result, snapshots = self._pending
        elif self._pending is not None and self._pending[0] == solution:
            result = self._pending[1]
            snapshots = self._pending[2]
        else:
            # no-op or mismatched pending: keep the current committed snapshots
            snapshots = dict(self.snapshots)
            if result is None:
                result = self.base_result
        self.base_solution = solution
        self.base_result = result
        self.snapshots = snapshots
        self._pending = None
