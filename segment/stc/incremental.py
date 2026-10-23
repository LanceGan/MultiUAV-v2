"""Exact incremental evaluation cache (Task 9).

Correctness-first: ``evaluate_after_move`` currently performs a full
re-evaluation, which is exactly equivalent to ``evaluate_solution``.  Affected-
window invalidation (rebuild only the changed UAVs + their overlapping windows)
is a later optimisation; the counters are in place so the optimisation can be
measured once implemented.
"""

from segment.stc.evaluator import evaluate_solution


class EvaluationCache:
    def __init__(self, scenario, library, field, base_solution, base_result):
        self.scenario = scenario
        self.library = library
        self.field = field
        self.base_solution = base_solution
        self.base_result = base_result
        self.full_calls = 1
        self.incremental_calls = 0
        self.recomputed_slots = 0

    @classmethod
    def from_full(cls, scenario, library, field, solution):
        return cls(scenario, library, field, solution,
                   evaluate_solution(scenario, library, solution, field))

    def evaluate_after_move(self, solution, affected_uavs):
        self.incremental_calls += 1
        self.full_calls += 1
        return evaluate_solution(self.scenario, self.library, solution, self.field)
