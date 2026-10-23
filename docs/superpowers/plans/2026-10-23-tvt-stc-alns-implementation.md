# TVT Spatio-Temporal Conflict-Aware Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible TVT-oriented multi-UAV inspection pipeline with a Pareto segment library, exact slot-synchronous shared-bandwidth evaluator, spatio-temporal conflict metrics, and adaptive STC-ALNS search.

**Architecture:** New work lives in a focused `segment/stc/` package rather than extending the legacy monolithic `segment/alns.py` and `segment/closed_loop.py`. A scenario/radio adapter produces deterministic SI-unit inputs; candidate generators build multi-path edge profiles; one synchronous evaluator is the sole source of final metrics; STC-ALNS changes route assignment, order, path choice, and departure shift using incremental evaluation. Legacy DRL models remain read-only baselines.

**Tech Stack:** Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, PyTorch 2.8.0 CPU for legacy learning baselines only, pytest 9.1.1, Matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-23-tvt-spatiotemporal-planning-design.md`

## Global Constraints

- Use SI units inside `segment/stc/`: metre, second, bit, hertz, watt; legacy unit conversion occurs only in `scenario.py` and `radio.py`.
- Fixed-altitude 2-D motion; flight legs use `v_max`, holding uses zero speed; do not add acceleration, turn-radius, or altitude decisions.
- A task generates its payload only at the fixed service anchor; the previous payload must be empty before the next service event.
- Holding for unfinished upload occurs at an explicit point outside the next arrival region and is capped by `max_wait_s`.
- Ground departure delay counts toward makespan but not G2A airborne-outage numerator or denominator.
- Every airborne UAV has exactly one serving sector; only UAVs with non-empty buffers consume uplink bandwidth.
- A solution exceeding `mission_deadline_s`, `max_wait_s`, upload-completion, or outage constraints is infeasible.
- `map_seed`, `task_seed`, `method_seed`, and `train_seed` are distinct fields and must never be silently reused.
- No formal result may read from shared mutable filenames; every run writes to `results/runs/<run_id>/` and records a manifest.
- Do not modify or delete `results/frozen_v0_2026-10-19`, `frozen_v1_2026-10-21`, or `frozen_v2_2026-10-23`.
- Do not tune N=4 TD3 or retrain MARL as part of this plan.
- Straight-MaxSpeed is a relaxation reference and is never ranked as a feasible method.

## File Structure

### New core package

- `segment/stc/types.py`: immutable scenario, candidate, solution, timeline, violation, evaluation, conflict, and search result data models.
- `segment/stc/scenario.py`: legacy-to-SI scenario conversion and validation.
- `segment/stc/radio.py`: deterministic per-sector A2G/G2A field adapter with explicit map seed.
- `segment/stc/generators.py`: Straight-MaxSpeed, Straight-Hold, and Comm-Waypoint trace generation.
- `segment/stc/graph_search.py`: grid A*/Dijkstra candidate generation.
- `segment/stc/slsqp_refine.py`: continuous waypoint refinement and fallback.
- `segment/stc/library.py`: all-edge candidate generation, Pareto/diversity pruning, NPZ/JSON persistence.
- `segment/stc/evaluator.py`: exact slot-synchronous state machine and objective calculation.
- `segment/stc/conflicts.py`: edge/time/sector congestion attribution.
- `segment/stc/incremental.py`: affected-window evaluation cache with exact full-replay equivalence.
- `segment/stc/operators.py`: destroy, repair, route, path-switch, and departure-shift operators.
- `segment/stc/solver.py`: adaptive operator weights, feasibility-first SA, checkpointable search.
- `segment/stc/exact.py`: branch-and-bound tiny-instance benchmark over the discrete candidate space.
- `segment/stc/stats.py`: paired estimates, confidence intervals, permutation/Wilcoxon tests, Holm correction.
- `segment/stc/run_context.py`: run directories, provenance, manifests, atomic result writes.

### New experiment entry points

- `experiment/run_stc_segment_gate.py`: RQ1 segment-method comparison.
- `experiment/run_stc_evaluator_audit.py`: RQ2 static-versus-synchronous audit.
- `experiment/run_stc_exact.py`: tiny exact benchmark.
- `experiment/run_stc_main.py`: base LNS versus STC-ALNS paired main experiment.
- `experiment/run_stc_ablation.py`: RQ4 mechanism ablations.
- `experiment/run_stc_scale.py`: controlled UAV/task/load/map experiments.
- `plots/plot_stc_results.py`: regenerate all formal figures from frozen machine-readable results.

### New configuration and tests

- `configs/stc/base.json`, `configs/stc/seeds.json`, `configs/stc/experiments.json`.
- `tests/stc/`: one focused test module per core file.

## Review Focus

1. **Airborne but empty-buffer UAV:** it must retain one G2A sector and accrue outage, while receiving zero uplink bandwidth; pinned in Task 6 evaluator tests.
2. **Arrival-boundary continuity:** adjacent traces must meet at the same service anchor without teleportation, and upload holding must remain outside the arrival region; pinned in Tasks 4 and 6.
3. **Zero-rate or chronically congested link:** waiting must end deterministically at `max_wait_s` and return a typed failure, never an infinite loop; pinned in Task 6.
4. **Departure shift reliability accounting:** ground delay changes makespan but cannot dilute `outage_time / airborne_time`; pinned in Task 7.
5. **Seed isolation:** changing `method_seed` must not change the map/task, and map construction must not mutate NumPy global RNG; pinned in Tasks 2 and 3.

---

## Phase I — Reproducible Foundation and Authoritative Evaluator

### Task 1: Run-Scoped Provenance and Atomic Artifacts

**Files:**
- Create: `segment/stc/__init__.py`
- Create: `segment/stc/run_context.py`
- Create: `tests/stc/test_run_context.py`
- Create: `configs/stc/base.json`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: repository root and a JSON-serializable resolved configuration.
- Produces: `RunContext.create(root: Path, experiment: str, config: dict, run_id: str | None = None) -> RunContext`; `RunContext.write_json(name: str, payload: dict) -> Path`; `RunContext.finalize(status: str) -> Path`.

- [ ] **Step 1: Write failing tests for unique directories and atomic manifests**

```python
def test_run_context_never_overwrites_existing_run(tmp_path): ...
def test_finalize_records_git_config_versions_and_hashes(tmp_path): ...
def test_failed_run_has_non_success_status(tmp_path): ...
```

Assert that a repeated explicit `run_id` raises `FileExistsError`, result writes use a temporary file plus rename, and the manifest contains Git HEAD/status, command, start/end time, dependency versions, resolved config, hashes, and status.

- [ ] **Step 2: Run tests and verify the intended failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_run_context.py -v`

Expected: FAIL because `segment.stc.run_context` does not exist.

- [ ] **Step 3: Implement the run context and base configuration**

`configs/stc/base.json` must declare SI values for `delta_t_s`, `v_max_mps`, `uav_height_m`, `arrive_radius_m`, `max_wait_s`, `mission_deadline_s`, `departure_step_s`, `departure_max_s`, `bandwidth_hz`, `uplink_power_w`, `eps_out`, `w_t`, and `w_d`. Values inherited from legacy configs are resolved and copied into each run manifest, never looked up later during plotting.

- [ ] **Step 4: Ignore only transient run state**

Add patterns for temporary files under `results/runs/**/.tmp-*`; do not ignore manifests, JSON, NPZ, or frozen results globally.

- [ ] **Step 5: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_run_context.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/__init__.py segment/stc/run_context.py tests/stc/test_run_context.py configs/stc/base.json .gitignore
git commit -m "feat: add run-scoped STC experiment provenance"
```

### Task 2: Typed SI Scenario and Seed Separation

**Files:**
- Create: `segment/stc/types.py`
- Create: `segment/stc/scenario.py`
- Create: `tests/stc/test_types.py`
- Create: `tests/stc/test_scenario.py`
- Create: `configs/stc/seeds.json`

**Interfaces:**
- Consumes: legacy scenario JSON and task data.
- Produces: `STCScenario`, `ScenarioLimits`, `RadioSample`, `SegmentCandidate`, `SegmentLibrary`, `PlanSolution`, `UAVTimeline`, `ViolationVector`, `EvaluationResult`, `ConflictReport`, and `SearchResult`; `load_stc_scenario(uav_num: int, *, task_seed: int, map_seed: int, config_path: Path | None = None) -> STCScenario`.

- [ ] **Step 1: Write failing dataclass validation tests**

```python
def test_segment_candidate_rejects_mismatched_trace_lengths(): ...
def test_plan_solution_requires_every_task_exactly_once(): ...
def test_scenario_requires_positive_finite_limits(): ...
```

Use immutable dataclasses. `SegmentCandidate` carries `candidate_id`, `source`, `target`, `generator`, `positions_m`, `full_rate_bps`, `sector_ids`, `g2a_outage`, `flight_steps`, `hold_position_m`, and `source_data_bits`.

- [ ] **Step 2: Write failing unit-conversion and seed-isolation tests**

```python
def test_legacy_coordinates_convert_from_100m_units_to_metres(): ...
def test_data_mbit_converts_to_bits(): ...
def test_method_seed_does_not_change_scenario(): ...
def test_task_seed_changes_only_tasks_not_map_seed(): ...
```

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_types.py tests/stc/test_scenario.py -v`

Expected: FAIL on missing interfaces.

- [ ] **Step 4: Implement types and scenario adapter**

Use node IDs `-1` for depot, `0..K-1` for tasks, and `K` for terminal. `PlanSolution` stores `routes: tuple[tuple[int, ...], ...]`, `candidate_ids: tuple[tuple[str, ...], ...]`, and `departure_steps: tuple[int, ...]`.

- [ ] **Step 5: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_types.py tests/stc/test_scenario.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/types.py segment/stc/scenario.py tests/stc/test_types.py tests/stc/test_scenario.py configs/stc/seeds.json
git commit -m "feat: define SI scenario and STC data contracts"
```

### Task 3: Explicitly Seeded Per-Sector Radio Field

**Files:**
- Modify: `radio_map/rural_world.py`
- Modify: `radio_map/radio_map_A2G.py`
- Modify: `radio_map/radio_map_G2A.py`
- Create: `segment/stc/radio.py`
- Create: `tests/stc/test_radio.py`
- Modify: `tests/test_radio_map.py`

**Interfaces:**
- Consumes: `STCScenario`, positions in metres, and explicit `map_seed`.
- Produces: `configure_radio_world(length_m: float, n_bs: int, map_seed: int) -> str` returning a world hash; `PerSectorRadioField.from_scenario(scenario: STCScenario, spacing_m: float) -> PerSectorRadioField`; `field.sample(position_m: np.ndarray) -> RadioSample` with per-sector `a2g_efficiency` and `g2a_outage` arrays.

- [ ] **Step 1: Write failing determinism and global-RNG tests**

```python
def test_same_map_seed_reproduces_field_and_world_hash(): ...
def test_different_map_seed_changes_field(): ...
def test_radio_configuration_does_not_mutate_numpy_global_rng(): ...
def test_a2g_and_g2a_share_sector_layout_and_world_hash(): ...
```

- [ ] **Step 2: Write failing per-sector shape and interpolation tests**

Assert `sample()` returns one A2G efficiency and one G2A outage per configured sector, interpolation remains within field bounds, and an out-of-region position raises `ValueError` instead of silently returning zero/one.

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_radio.py tests/test_radio_map.py -v`

Expected: FAIL because map generation currently resets global NumPy seeds and lacks explicit world hashes.

- [ ] **Step 4: Refactor map generation around local RNG instances**

Extend `set_region` to accept explicit terrain/building seeds derived from `map_seed`, preserve existing default behavior for legacy callers, and expose per-sector values without reducing them to a best sector.

- [ ] **Step 5: Implement `PerSectorRadioField`**

Store SI grid coordinates, per-sector spectral efficiency, per-sector G2A outage, BS layout, seed metadata, and world hash. Do not cache only the maximum sector.

- [ ] **Step 6: Run focused and legacy tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_radio.py tests/test_radio_map.py tests/test_radio_path.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add radio_map/rural_world.py radio_map/radio_map_A2G.py radio_map/radio_map_G2A.py segment/stc/radio.py tests/stc/test_radio.py tests/test_radio_map.py
git commit -m "feat: expose deterministic per-sector radio fields"
```

### Task 4: Legal Direct and Communication-Waypoint Trace Generators

**Files:**
- Create: `segment/stc/generators.py`
- Create: `tests/stc/test_generators.py`
- Modify: `segment/baselines.py`

**Interfaces:**
- Consumes: `STCScenario`, `PerSectorRadioField`, directed edge, payload bits.
- Produces: `generate_straight_relaxation(...) -> SegmentCandidate`; `generate_straight_hold(...) -> SegmentCandidate`; `generate_comm_waypoints(..., grid_shape: tuple[int, int], max_waypoints: int) -> list[SegmentCandidate]`.

- [ ] **Step 1: Write failing trace-semantic tests**

```python
def test_straight_hold_waits_outside_arrival_region(): ...
def test_candidate_ends_at_shared_service_anchor_without_jump(): ...
def test_zero_payload_candidate_never_holds(): ...
def test_straight_relaxation_is_marked_non_rankable(): ...
```

Assert successive positions respect `v_max_mps * delta_t_s`, the explicit hold point is at least `arrive_radius_m` from the target anchor, and the final connector terminates at the common anchor.

- [ ] **Step 2: Write failing waypoint-generation tests**

Assert candidates are deterministic, inside the region, deduplicated, and include the direct legal candidate. Use a synthetic field with a known high-rate island to verify at least one communication-aware path visits the island.

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_generators.py -v`

Expected: FAIL on missing trace generators.

- [ ] **Step 4: Implement generators using one shared trace sampler**

Move new logic into `segment/stc/generators.py`; leave compatibility wrappers in `segment/baselines.py` so existing scripts continue to run until Task 15 deprecates them.

- [ ] **Step 5: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_generators.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/generators.py segment/baselines.py tests/stc/test_generators.py
git commit -m "feat: generate legal SI-unit segment traces"
```

### Task 5: Graph Search, SLSQP Refinement, and Pareto Segment Library

**Files:**
- Create: `segment/stc/graph_search.py`
- Create: `segment/stc/slsqp_refine.py`
- Create: `segment/stc/library.py`
- Create: `tests/stc/test_graph_search.py`
- Create: `tests/stc/test_slsqp_refine.py`
- Create: `tests/stc/test_library.py`

**Interfaces:**
- Consumes: Task 4 candidates and Task 3 radio field.
- Produces: `astar_candidates(...) -> list[SegmentCandidate]`; `refine_candidate(candidate, scenario, field, max_waypoints: int) -> SegmentCandidate`; `pareto_prune(candidates, *, max_candidates: int, min_sector_distance: float) -> tuple[SegmentCandidate, ...]`; `build_library(...) -> SegmentLibrary`; `save_library(...)`; `load_library(...)`.

- [ ] **Step 1: Write failing A*/Dijkstra tests**

Use a small synthetic grid where the minimum-cost path is known. Assert path endpoints, obstacle/region compliance, determinism, and that the returned claim is “optimal on this grid,” not continuous-global optimum.

- [ ] **Step 2: Write failing SLSQP fallback tests**

Assert refinement never worsens the evaluated single-UAV objective, preserves anchors and bounds, and returns the original candidate when SciPy reports failure or the refined trace is infeasible.

- [ ] **Step 3: Write failing Pareto/diversity tests**

```python
def test_pareto_prune_removes_strictly_dominated_candidate(): ...
def test_pareto_prune_keeps_sector_diverse_near_ties(): ...
def test_library_round_trip_preserves_arrays_hash_and_metadata(tmp_path): ...
```

- [ ] **Step 4: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_graph_search.py tests/stc/test_slsqp_refine.py tests/stc/test_library.py -v`

Expected: FAIL on missing modules.

- [ ] **Step 5: Implement graph search and refinement**

The graph edge score combines travel time, predicted upload progress, and G2A outage using weights from resolved config. SLSQP optimizes at most `max_waypoints` interior points and is always re-sampled through the same trace evaluator before acceptance.

- [ ] **Step 6: Implement library generation and persistence**

Persist arrays in NPZ and metadata/hashes in adjacent JSON. Refuse to load when scenario hash, map hash, task hash, SI schema version, or generator config differs.

- [ ] **Step 7: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_graph_search.py tests/stc/test_slsqp_refine.py tests/stc/test_library.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/graph_search.py segment/stc/slsqp_refine.py segment/stc/library.py tests/stc/test_graph_search.py tests/stc/test_slsqp_refine.py tests/stc/test_library.py
git commit -m "feat: build Pareto communication-aware segment library"
```

### Task 6: Slot-Synchronous Shared-Bandwidth State Machine

**Files:**
- Create: `segment/stc/evaluator.py`
- Create: `tests/stc/test_evaluator.py`

**Interfaces:**
- Consumes: `STCScenario`, `SegmentLibrary`, `PlanSolution`, `PerSectorRadioField`.
- Produces: `evaluate_solution(scenario, library, solution, field, *, keep_timelines: bool = True) -> EvaluationResult`; `select_serving_sector(sample: RadioSample, has_payload: bool, reliability_limit: float) -> int`.

- [ ] **Step 1: Write failing bandwidth-conservation tests**

```python
def test_two_active_uploaders_same_sector_split_bandwidth_equally(): ...
def test_empty_buffer_uav_has_sector_but_consumes_zero_bandwidth(): ...
def test_different_sectors_do_not_reduce_each_others_rate(): ...
```

- [ ] **Step 2: Write failing cache and task-event tests**

Assert generated bits equal uploaded plus remaining bits, a service event cannot occur while the previous payload remains, and the next payload appears exactly once at the common service anchor.

- [ ] **Step 3: Write failing hold/deadline tests**

```python
def test_upload_finishes_at_boundary_hold_before_service_event(): ...
def test_zero_rate_fails_at_max_wait_without_infinite_loop(): ...
def test_mission_deadline_returns_typed_violation(): ...
```

- [ ] **Step 4: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_evaluator.py -v`

Expected: FAIL because the synchronous evaluator does not exist.

- [ ] **Step 5: Implement one deterministic slot-update loop**

Follow the spec order exactly: activate; move; detect hold boundary; associate all airborne UAVs; identify non-empty buffers; divide bandwidth; update buffers; allow final connector/service; update metrics and violations.

- [ ] **Step 6: Return complete typed timelines**

Each `UAVTimeline` records per slot: time, position, route edge, candidate ID, airborne flag, buffer, service sector, allocated bandwidth, rate, uploaded bits, G2A outage, holding flag, and service event.

- [ ] **Step 7: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_evaluator.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/evaluator.py tests/stc/test_evaluator.py
git commit -m "feat: add authoritative synchronous multi-UAV evaluator"
```

### Task 7: Reliability, Normalized Objective, and Reference Baseline

**Files:**
- Modify: `segment/stc/evaluator.py`
- Create: `tests/stc/test_objective.py`
- Create: `tests/stc/test_reliability.py`

**Interfaces:**
- Consumes: evaluator timelines.
- Produces: `compute_reference(scenario, library) -> tuple[float, float]`; `objective(result: EvaluationResult, t_ref: float, d_ref: float, w_t: float, w_d: float) -> float`; `feasibility_rank(result) -> tuple[int, ...]`.

- [ ] **Step 1: Write failing reference/objective tests**

Assert `w_t*T/T_ref + w_d*D/D_ref` exactly matches hand calculation, reference values are positive and fixed for an experiment family, and an infeasible Straight-MaxSpeed result cannot outrank a feasible solution.

- [ ] **Step 2: Write failing airborne-outage tests**

```python
def test_empty_buffer_flight_still_accumulates_g2a_outage(): ...
def test_ground_departure_delay_changes_makespan_not_outage_ratio(): ...
def test_route_outage_constraint_uses_airborne_time_only(): ...
```

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_objective.py tests/stc/test_reliability.py -v`

Expected: FAIL on missing objective/reliability behavior.

- [ ] **Step 4: Implement objective and lexicographic feasibility rank**

Rank by unserved tasks, upload violations, outage excess, deadline/wait violations, then normalized objective. Maintain `best_feasible` separately in later search code.

- [ ] **Step 5: Run evaluator and objective suites**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_evaluator.py tests/stc/test_objective.py tests/stc/test_reliability.py -v`

Expected: PASS.

- [ ] **Step 6: Run the full pre-STC regression suite**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests -v`

Expected: all existing and new tests PASS.

- [ ] **Step 7: Commit**

```bash
git add segment/stc/evaluator.py tests/stc/test_objective.py tests/stc/test_reliability.py
git commit -m "feat: enforce STC reliability and normalized objective"
```

## Phase II — Conflict Attribution and STC-ALNS

### Task 8: Edge-Time-Sector Conflict Attribution

**Files:**
- Create: `segment/stc/conflicts.py`
- Create: `tests/stc/test_conflicts.py`

**Interfaces:**
- Consumes: full `EvaluationResult` timelines.
- Produces: `build_conflict_report(result: EvaluationResult) -> ConflictReport`; report fields include total overlap time, exclusive-versus-shared lost bits, per-UAV loss, per-edge hotspot score, and pairwise overlap windows.

- [ ] **Step 1: Write failing synthetic conflict tests**

Use hand-built timelines for no overlap, same-sector overlap with empty buffers, and two active same-sector uploaders. Assert conflict occurs only in the last case and lost-bit attribution sums to the system total.

- [ ] **Step 2: Write failing edge attribution tests**

Assert each conflicting slot is attributed to the route edge active in that slot, and the edge hotspot order matches known synthetic losses rather than raw co-sector headcount.

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_conflicts.py -v`

Expected: FAIL on missing conflict report.

- [ ] **Step 4: Implement conflict attribution**

Calculate both overlap time and rate/bit loss. Do not use the legacy per-UAV average factor as the proposed metric.

- [ ] **Step 5: Run tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_conflicts.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/conflicts.py tests/stc/test_conflicts.py
git commit -m "feat: attribute spatio-temporal sector conflicts"
```

### Task 9: Exact Incremental Evaluation Cache

**Files:**
- Create: `segment/stc/incremental.py`
- Create: `tests/stc/test_incremental.py`

**Interfaces:**
- Consumes: base solution/result and a candidate solution plus affected UAV IDs.
- Produces: `EvaluationCache.from_full(...) -> EvaluationCache`; `cache.evaluate_after_move(solution: PlanSolution, affected_uavs: frozenset[int]) -> EvaluationResult`; cache counters for full calls, incremental calls, recomputed slots, and hits.

- [ ] **Step 1: Write failing move-equivalence tests**

Cover relocate, swap, path switch, and departure shift. For each move, compare every aggregate metric and every affected timeline value against a fresh `evaluate_solution` call within `1e-9` for scalar values and exact equality for discrete events.

- [ ] **Step 2: Write failing contention-propagation test**

Change one UAV so it begins sharing a sector with an otherwise unchanged UAV. Assert the unchanged UAV is included in the recomputed overlap window and its rate/delay changes correctly.

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_incremental.py -v`

Expected: FAIL on missing cache.

- [ ] **Step 4: Implement conservative affected-window invalidation**

Start with correctness-first invalidation: rebuild each changed UAV from its earliest changed edge and all other UAV windows sharing any candidate sector/time range. Optimize further only after exact equivalence tests pass.

- [ ] **Step 5: Run tests and benchmark fixture**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_incremental.py -v`

Expected: PASS and the benchmark fixture reports fewer recomputed slots than full replay for a local path switch.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/incremental.py tests/stc/test_incremental.py
git commit -m "feat: add exact incremental synchronous evaluation"
```

### Task 10: Initial Solutions and Resource-Unaware Baselines

**Files:**
- Create: `segment/stc/operators.py`
- Create: `tests/stc/test_initial_solution.py`
- Create: `tests/stc/test_baseline_search.py`

**Interfaces:**
- Consumes: scenario, library, evaluator/cache.
- Produces: `build_initial_solution(...) -> PlanSolution`; `run_nearest_neighbor_2opt(...) -> SearchResult`; `run_cheapest_insertion(...) -> SearchResult`; `run_lns_sa(...) -> SearchResult`.

- [ ] **Step 1: Write failing coverage and deterministic-seed tests**

Assert all tasks appear exactly once, every route has one candidate per traversed edge including depot/terminal edges, departure shifts belong to the configured set, and identical method seeds reproduce identical initial solutions.

- [ ] **Step 2: Write failing no-feasible-insertion behavior**

Assert construction returns a typed infeasible result with unserved tasks instead of crashing, duplicating a task, or silently accepting an invalid edge.

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_initial_solution.py tests/stc/test_baseline_search.py -v`

Expected: FAIL on missing operators/baselines.

- [ ] **Step 4: Implement initial construction and three baseline searches**

The resource-unaware baselines may use independent segment costs to guide moves, but all reported final metrics must come from the authoritative synchronous evaluator.

- [ ] **Step 5: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_initial_solution.py tests/stc/test_baseline_search.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/operators.py tests/stc/test_initial_solution.py tests/stc/test_baseline_search.py
git commit -m "feat: add reproducible STC search baselines"
```

### Task 11: Conflict-Aware Destroy, Repair, and Resource Moves

**Files:**
- Modify: `segment/stc/operators.py`
- Create: `tests/stc/test_operators.py`

**Interfaces:**
- Consumes: `PlanSolution`, `EvaluationResult`, `ConflictReport`, `EvaluationCache`, RNG.
- Produces operator callables with common signature `operator(solution, context, rng) -> PlanSolution` and named registries `destroy_operators`, `repair_operators`, `local_operators`.

- [ ] **Step 1: Write failing destroy tests**

Test random removal, worst-cost removal, critical-route removal, conflict-hotspot removal, sector-related removal, and infeasibility removal. On synthetic reports, each targeted operator must remove the known target task/edge more often than a uniform operator under a fixed seed.

- [ ] **Step 2: Write failing repair tests**

Test cheapest feasible, regret-2, makespan-balancing, conflict-aware, and sector-diversity insertion. Assert repaired solutions contain every original task exactly once and use synchronous marginal costs for conflict-aware insertion.

- [ ] **Step 3: Write failing local/resource move tests**

Cover relocate, swap, 2-opt, route-tail exchange, path-switch, departure-shift, and conflict-window shift. Assert bounds, candidate-edge compatibility, and no task loss/duplication.

- [ ] **Step 4: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_operators.py -v`

Expected: FAIL because the complete operator set is missing.

- [ ] **Step 5: Implement operators in small named functions**

Each operator changes one concern only and returns a new immutable `PlanSolution`; it never mutates the incumbent in place.

- [ ] **Step 6: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_operators.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add segment/stc/operators.py tests/stc/test_operators.py
git commit -m "feat: add conflict-aware STC neighborhood operators"
```

### Task 12: Adaptive STC-ALNS Solver

**Files:**
- Create: `segment/stc/solver.py`
- Create: `tests/stc/test_solver.py`
- Create: `configs/stc/experiments.json`

**Interfaces:**
- Consumes: scenario, library, radio field, operator registries, solver configuration.
- Produces: `STCALNS.solve(initial: PlanSolution, *, iterations: int, method_seed: int) -> SearchResult`; serializable `SearchTrace` containing accepted moves, operator choices, weights, incumbent ranks, best-feasible history, and evaluator counters.

- [ ] **Step 1: Write failing adaptive-weight tests**

Assert an operator repeatedly receiving the “new global best” score gains selection probability after a segment update, unused operators do not divide by zero, and all probabilities remain finite and sum to one.

- [ ] **Step 2: Write failing feasibility-first and best-feasible tests**

Assert a feasible result beats a lower-objective infeasible result, SA cannot erase the separately stored best feasible solution, and a run with no feasible solution returns `best_feasible=None` plus the least-violating result.

- [ ] **Step 3: Write failing reproducibility/checkpoint tests**

Assert identical inputs and method seed reproduce the trace, and resume from a serialized solver checkpoint yields the same final result as an uninterrupted run.

- [ ] **Step 4: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_solver.py -v`

Expected: FAIL on missing solver.

- [ ] **Step 5: Implement adaptive roulette selection and SA**

Use the spec update `w <- (1-rho)w + rho*score/max(1,uses)` at fixed update segments. Validation-set configuration fixes reaction factor, score constants, starting temperature, cooling, removal range, and stopping rules.

- [ ] **Step 6: Integrate periodic full-replay audits**

At configured intervals, compare incremental and full evaluation; raise and save a diagnostic artifact on any mismatch rather than continuing with corrupted search state.

- [ ] **Step 7: Run solver and all Phase II tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_conflicts.py tests/stc/test_incremental.py tests/stc/test_initial_solution.py tests/stc/test_baseline_search.py tests/stc/test_operators.py tests/stc/test_solver.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/solver.py tests/stc/test_solver.py configs/stc/experiments.json
git commit -m "feat: implement adaptive conflict-aware STC-ALNS"
```

## Phase III — Exact Benchmark and Formal Experiment Pipeline

### Task 13: Tiny-Instance Exact Discrete Benchmark

**Files:**
- Create: `segment/stc/exact.py`
- Create: `tests/stc/test_exact.py`
- Create: `experiment/run_stc_exact.py`
- Create: `docs/stc-method-analysis.md`

**Interfaces:**
- Consumes: tiny scenario, capped candidate library, exact evaluator, departure set.
- Produces: `solve_exact_tiny(..., max_tasks: int = 8, time_limit_s: float = 3600.0) -> ExactResult` with incumbent, admissible lower bound, proven-optimal flag, explored/pruned nodes, gap, and runtime.

- [ ] **Step 1: Write failing exactness tests against exhaustive enumeration**

For one- and two-UAV fixtures with up to four tasks, enumerate every assignment/order/candidate/departure combination in the test and assert `solve_exact_tiny` returns the same best feasible objective.

- [ ] **Step 2: Write failing timeout/lower-bound tests**

Assert a timed-out run returns a valid incumbent, an admissible lower bound not exceeding the incumbent, `proven_optimal=False`, and a nonnegative gap.

- [ ] **Step 3: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_exact.py -v`

Expected: FAIL on missing exact solver.

- [ ] **Step 4: Implement branch-and-bound over the discrete candidate space**

Use independent no-contention edge lower costs as the admissible bound. Branch on task assignment/order first, then path candidate and departure shift. Always confirm leaves with the synchronous evaluator.

- [ ] **Step 5: Implement run entry point with run context**

Write `summary.json`, `solutions.json`, `search_trace.json`, and final manifest under one run directory.

- [ ] **Step 6: Document complexity and the NP-hard restricted case**

In `docs/stc-method-analysis.md`, give the restricted-case reduction from multiple-TSP/vehicle routing, define the discrete candidate-space exact benchmark, state the admissible lower bound, and distinguish “proven optimal in the discrete candidate space” from continuous global optimality.

- [ ] **Step 7: Run tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_exact.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/exact.py tests/stc/test_exact.py experiment/run_stc_exact.py docs/stc-method-analysis.md
git commit -m "feat: add exact tiny-instance STC benchmark"
```

### Task 14: Statistical Protocol and RQ1/RQ2 Entry Points

**Files:**
- Create: `segment/stc/stats.py`
- Create: `tests/stc/test_stats.py`
- Create: `experiment/run_stc_segment_gate.py`
- Create: `experiment/run_stc_evaluator_audit.py`
- Create: `configs/stc/seeds.json`

**Interfaces:**
- Consumes: paired machine-readable records.
- Produces: `paired_summary(a, b, *, confidence: float = 0.95, permutations: int = 10000, seed: int = 0) -> PairedSummary`; `hierarchical_paired_summary(records, *, scenario_keys: tuple[str, ...] = ("map_seed", "task_seed"), replicate_key: str = "method_seed") -> PairedSummary`; `holm_adjust(p_values: dict[str, float]) -> dict[str, float]`.

- [ ] **Step 1: Write failing statistical tests**

Assert exact mean/median/std, deterministic bootstrap/permutation intervals, zero-difference p-value behavior, mismatched-pair rejection, NaN rejection, and monotone Holm-adjusted p-values. Add a fixture proving repeated method seeds are first aggregated within a map/task scenario and are not incorrectly treated as independent scenario samples.

- [ ] **Step 2: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_stats.py -v`

Expected: FAIL on missing stats module.

- [ ] **Step 3: Implement RQ1 segment gate**

Evaluate Straight-MaxSpeed, Straight-Hold, Comm-Waypoint, Grid A*/Dijkstra, SLSQP, TD3, DDPG, and SAC on the same edge IDs. Save full-edge feasibility, common-feasible paired metrics, failure reasons, generation/inference time, normalized objective, and provenance. Never average each method over a different success set without labeling it separately.

- [ ] **Step 4: Implement RQ2 evaluator audit**

For identical solutions, compare exclusive bandwidth, legacy static average factor, and synchronous evaluation. Save absolute/relative T/D errors, feasibility reversals, and at least one ranking-reversal candidate if found; absence of a reversal must be reported honestly.

- [ ] **Step 5: Run tests and CLI smoke checks**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_stats.py -v`

Run: `F:/anaconda3/envs/pytorch/python.exe experiment/run_stc_segment_gate.py --help`

Run: `F:/anaconda3/envs/pytorch/python.exe experiment/run_stc_evaluator_audit.py --help`

Expected: tests PASS and both commands exit 0.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/stats.py tests/stc/test_stats.py experiment/run_stc_segment_gate.py experiment/run_stc_evaluator_audit.py configs/stc/seeds.json
git commit -m "feat: add paired STC statistics and gate experiments"
```

### Task 15: Main, Ablation, Scale, Plotting, and Final Freeze Pipeline

**Files:**
- Create: `experiment/run_stc_main.py`
- Create: `experiment/run_stc_ablation.py`
- Create: `experiment/run_stc_scale.py`
- Create: `plots/plot_stc_results.py`
- Create: `tests/stc/test_experiment_cli.py`
- Create: `tests/stc/test_plot_stc_results.py`
- Create: `docs/stc-experiment-protocol.md`
- Modify: `README.md`
- Modify: `仿真实验.md`
- Modify: `segment/closed_loop.py`
- Modify: `segment/run_multiseed.py`

**Interfaces:**
- Consumes: all completed STC modules and frozen run directories.
- Produces: run-scoped raw JSON/NPZ, summary tables, figures, gate decisions, and a frozen manifest; legacy scripts emit a deprecation warning and never write formal STC results.

- [ ] **Step 1: Write failing CLI/config tests**

Assert every runner supports `--config`, `--map-seeds`, `--task-seeds`, `--method-seeds`, `--dry-run`, and `--run-id`; rejects overlapping train/validation/test seeds; and prints the resolved run matrix under `--dry-run` without generating maps or results.

- [ ] **Step 2: Write failing controlled-scale tests**

Assert the “vary UAV count” matrix keeps area, BSs, tasks, payloads, and map fixed; the “vary task scale” matrix changes area/task/BS density according to explicit config; and no result aggregator mixes these two experiment families.

- [ ] **Step 3: Write failing plotting provenance tests**

Assert plotting reads only finalized run manifests and raw records, rejects mixed scenario/map/code hashes, contains no hard-coded result values, and writes source-run IDs into figure metadata or adjacent JSON.

- [ ] **Step 4: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_experiment_cli.py tests/stc/test_plot_stc_results.py -v`

Expected: FAIL because formal experiment runners do not exist.

- [ ] **Step 5: Implement `run_stc_main.py`**

Run paired nearest-neighbor+2-opt, cheapest insertion, LNS-SA, static-factor iterative LNS, and STC-ALNS on identical scenarios. Save feasibility, T, D, J, outage, conflict loss, runtime, exact/full/incremental evaluator counts, and paired statistics.

- [ ] **Step 6: Implement mechanism ablations**

Run fixed-weight, no-hotspot-destroy, no-conflict-repair, no-path-switch, no-departure-shift, single-candidate-library, and full-replay-only variants. Change exactly one mechanism per ablation.

- [ ] **Step 7: Implement controlled scale and robustness families**

Separate UAV-count, task-density, communication-load, map-seed, and fast-fading replay experiments. Power sweeps regenerate the library at each power; they never reuse a fixed-policy result as if it were retrained or re-optimized.

- [ ] **Step 8: Implement formal plotting and tables**

Generate RQ1–RQ5 figures, paired-difference confidence plots, feasibility/failure charts, small-instance gap table, runtime/scaling table, and ablation plot directly from finalized raw results.

- [ ] **Step 9: Replace stale narrative claims and deprecate legacy formal paths**

Rewrite `仿真实验.md` to point to the approved STC design and protocol; remove claims of TD3 27–32% improvement, universal 90% generalization, natural contention avoidance, and successful N=2/3/4 scaling. Add warnings in `segment/closed_loop.py` and `segment/run_multiseed.py` that they are legacy prototypes not accepted by the formal pipeline.

- [ ] **Step 10: Document exact commands and Gate 0–4 interpretation**

`docs/stc-experiment-protocol.md` must list smoke, pilot, validation, formal, freeze, and plot commands; expected artifact names; seed matrices; gate rules; and which claims are allowed after each gate.

The formal runner must emit explicit `gate_0` through `gate_4` decisions. Gate 4 passes only when every reported table/figure source belongs to finalized runs with matching code, config, map, task, library, and command hashes.

- [ ] **Step 11: Run focused tests and CLI dry runs**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_experiment_cli.py tests/stc/test_plot_stc_results.py -v`

Run: `F:/anaconda3/envs/pytorch/python.exe experiment/run_stc_main.py --config configs/stc/experiments.json --dry-run`

Run: `F:/anaconda3/envs/pytorch/python.exe experiment/run_stc_ablation.py --config configs/stc/experiments.json --dry-run`

Run: `F:/anaconda3/envs/pytorch/python.exe experiment/run_stc_scale.py --config configs/stc/experiments.json --dry-run`

Expected: tests PASS; dry runs exit 0 and print non-overlapping, controlled matrices.

- [ ] **Step 12: Run the complete regression suite**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 13: Run a tiny end-to-end smoke experiment**

Run: `F:/anaconda3/envs/pytorch/python.exe experiment/run_stc_main.py --config configs/stc/experiments.json --map-seeds 42 --task-seeds 42 --method-seeds 42 --smoke`

Expected: one finalized run directory containing resolved config, segment-library hash, raw records, summary, search trace, and manifest; no shared result file is overwritten.

- [ ] **Step 14: Regenerate smoke figures from the finalized run**

Run: `F:/anaconda3/envs/pytorch/python.exe plots/plot_stc_results.py --runs results/runs/<smoke-run-id>`

Expected: figures and adjacent provenance JSON are created using only that run.

- [ ] **Step 15: Commit**

```bash
git add experiment/run_stc_main.py experiment/run_stc_ablation.py experiment/run_stc_scale.py plots/plot_stc_results.py tests/stc/test_experiment_cli.py tests/stc/test_plot_stc_results.py docs/stc-experiment-protocol.md README.md 仿真实验.md segment/closed_loop.py segment/run_multiseed.py
git commit -m "feat: complete reproducible STC experiment pipeline"
```

## Milestone Reviews

### After Phase I

- Gate 0 evaluator invariants all pass.
- A hand-calculated two-UAV/two-sector fixture matches the evaluator exactly.
- Segment library round-trip hashes match.
- Existing repository tests remain green.
- Do not proceed if the evaluator still depends on legacy TD3 rollout semantics.

### After Phase II

- Incremental and full evaluations are exactly equivalent within declared floating tolerance.
- STC operators preserve task coverage and candidate compatibility.
- Adaptive weights visibly respond on a synthetic operator-quality fixture.
- A congestion stress fixture shows conflict-aware moves can reduce actual lost bits.
- This milestone proves mechanism correctness, not paper superiority.

### After Phase III

- Tiny cases have exact or bounded reference results.
- All formal methods share the same evaluator and paired instances.
- Gate decisions are generated from machine-readable results.
- Figures can be deleted and regenerated from frozen runs.
- Only after pilot Gate 1–3 results are reviewed should the expensive formal matrix be launched.

## Recommended Execution Order and Review Policy

Execute Tasks 1–15 strictly in order because the data contracts and evaluator are dependencies of every later component. Use a fresh implementation subagent for each task and a fresh review subagent before starting the next task. Require a whole-branch review after Task 7, Task 12, and Task 15; evaluator or provenance defects found at those gates block later experiment runs.
