# STC-ALNS Recovery and Experiment-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the current STC-ALNS prototype into a scientifically correct, reproducible implementation that passes Gate 0–4 and can generate the formal RQ1–RQ5 evidence required for a TVT submission.

**Architecture:** Keep the existing `segment/stc/` package and seeded radio-field work, but replace placeholder behavior in the evaluator, conflict pipeline, solver, exact benchmark, persistence, statistics, and experiment runners. The authoritative flow becomes `resolved scenario + persisted Pareto library -> synchronous evaluator -> conflict report -> STC-ALNS/full baselines -> run-scoped raw records -> paired statistics/figures/gates`. Graph optimization is strengthened as a segment-layer baseline and candidate generator; Partial-CKM is explicitly excluded.

**Tech Stack:** Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, PyTorch 2.8.0 CPU for legacy learned baselines only, pytest 9.1.1, Matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-23-tvt-spatiotemporal-planning-design.md`

**Supersedes for future execution:** `docs/superpowers/plans/2026-10-23-tvt-stc-alns-implementation.md`. That plan remains historical evidence of the first implementation round; unfinished tasks are re-specified here from the current code state.

## Verified Starting State

- Code baseline: `90cb8ba3835ceab2fb3bd9696e3ca9aa9f17f323`; this recovery plan is committed immediately after it.
- Untracked file: `experiment/run_stc_gates.py`; it is exploratory, Gate 3 is deferred, and it is not accepted as formal evidence.
- The audit at `d1b435d` observed `tests/stc` 62/62 and full `tests` 86 passed/3 failed with approximately 1.06 million warnings. Commit `f7691c5` updated those three stale tests and scalar extraction; Task 1 requires fresh full-suite verification and stronger regression coverage rather than assuming success.
- Commit `929d28b` added optional `t_ref/d_ref`, but the evaluator still retains an unnormalized fallback and no frozen `ReferenceValues` contract.
- Commit `056d7cb` injects a per-UAV conflict report into the solver, but edge hotspots/pairwise windows and synchronous marginal repair remain absent.
- Commit `2ec4a37` adds candidate enumeration and a size guard, but still eagerly materializes permutations/candidate products and has no useful lower bound/gap.
- Commit `90cb8ba` persists candidate arrays and takes an explicit statistics method pair, but lacks compatibility/content hashes, replicate validation, and failure policy.
- `results/runs/smoke-test-1` contains only seed lists and a manifest; it is not a scientific smoke result.
- Current formal runners, plotting entry point, library persistence, edge hotspot attribution, incremental evaluation, and parts of exact/statistical inference are placeholders.

## Global Constraints

- Do not add Partial-CKM, Kriging, online CKM completion, or CKM measurement objectives to this implementation.
- Use complete deterministic CKM/radio fields with explicit `map_seed`; map uncertainty may only be future work.
- Keep SI units inside `segment/stc/`; legacy conversion remains at `scenario.py` and `radio.py` boundaries.
- The previous payload must be empty before the next service-anchor event; waiting occurs outside the arrival region and is bounded.
- Every airborne UAV accrues G2A outage; only UAVs with pre-slot non-empty buffers consume uplink bandwidth.
- All optimization uses `w_t*T/T_ref + w_d*D/D_ref`; no unnormalized objective may enter acceptance, ranking, plots, or tables.
- A conflict is defined from pre-slot active-upload state, common sector, bandwidth loss, edge identity, and overlap window.
- Formal baselines and STC-ALNS use the same persisted library, synchronous evaluator, reference values, failure policy, scenarios, and wall-clock accounting.
- Straight-MaxSpeed is a non-rankable relaxation reference.
- Do not report exactness unless every assignment, order, permitted segment candidate, and permitted departure shift in the declared discrete domain is covered or safely pruned by an admissible bound.
- Do not create a success manifest without raw records, resolved inputs, artifact hashes, and completed expected outputs.
- Do not modify or delete `results/frozen_v0_2026-10-19`, `frozen_v1_2026-10-21`, or `frozen_v2_2026-10-23`.
- Do not launch the full formal matrix until the pilot produces machine-readable Gate 0–3 decisions and receives review approval.

## File Responsibility Map

- `segment/stc/types.py`: add objective/reference, typed timeline slots, exact/statistical result contracts.
- `segment/stc/evaluator.py`: authoritative slot semantics, normalized scoring, Gate 0 invariants.
- `segment/stc/library.py`: full candidate persistence, hashes, compatibility checks.
- `segment/stc/run_context.py`: immutable provenance, expected-artifact contract, sealed atomic finalization.
- `segment/stc/conflicts.py`: edge hotspots, pairwise windows, lost-bit attribution.
- `segment/stc/operators.py`: scientifically valid resource-unaware search baselines.
- `segment/stc/stc_operators.py`: full conflict-aware neighborhood family and ablation switches.
- `segment/stc/solver.py`: conflict feedback, adaptive weights, checkpoint/resume, full-replay audits.
- `segment/stc/incremental.py`: exact affected-window recomputation.
- `segment/stc/graph_search.py`: state-expanded communication-aware graph baseline and Pareto weight sweeps.
- `segment/stc/slsqp_refine.py`: exact-replay refinement acceptance and explicit failure reporting.
- `segment/stc/exact.py`: guarded lazy discrete exact search with candidate branching and admissible bounds.
- `segment/stc/stats.py`: explicit method contrasts and valid hierarchical pairing.
- `experiment/run_stc_*.py`: real RQ and formal experiment execution.
- `plots/plot_stc_results.py`: manifest-validated figure/table regeneration.
- `experiment/run_stc_gates.py`: replace the untracked exploratory script with a committed gate orchestrator only after underlying runners are real.

## Review Focus

1. **Objective consistency:** the evaluator, baselines, STC-ALNS, exact solver, raw records, statistics, and plots must use the same frozen `ReferenceValues`; pinned in Tasks 2, 7, 9, and 10.
2. **Last upload slot:** a buffer that becomes empty during a slot still consumed bandwidth and caused conflict in that slot; pinned in Tasks 2 and 4.
3. **Conflict feedback authenticity:** hotspot operators and conflict-aware repair must use edge/time synchronous marginal effects, not static per-UAV penalties or random fallbacks; pinned in Task 5.
4. **Exactness safety:** oversized tasks must be rejected before permutation allocation, and multi-candidate branches cannot be silently omitted; pinned in Task 8.
5. **Scientific run integrity:** a runner must not accept missing config/input files, silently skip method/scenario pairs, or finalize success with missing artifacts; pinned in Tasks 3, 9, 10, and 11.

---

## Phase A — Restore a Trustworthy Gate 0

### Task 1: Reconcile the Full Test Suite and Radio Warnings

**Files:**
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_radio_map.py`
- Modify: `tests/test_radio_path.py`
- Modify: `radio_map/radio_map_A2G.py`
- Modify: `radio_map/radio_map_G2A.py`
- Test: `tests/test_metrics.py`, `tests/test_radio_map.py`, `tests/test_radio_path.py`

**Interfaces:**
- Consumes: current scenario definitions and radio-map APIs.
- Produces: a full green baseline suite with no NumPy array-to-scalar deprecation warning from antenna-gain assignment.

- [ ] **Step 1: Verify the merged model updates and add missing regression tests**

```python
def test_get_data_sizes_shape_and_range():
    sizes = config.get_data_sizes(20, seed=42)
    assert np.all((10.0 <= sizes) & (sizes <= 30.0))

def test_users_files_fit_their_scenario_bounds(): ...
def test_radio_path_cost_is_deterministic_and_components_are_finite(): ...
def test_antenna_gain_is_python_scalar_without_warning(): ...
```

The merged Users test now uses 4/5/6 km bounds and data sizes 10–30 Mbit. Extend it so bounds are loaded from scenario JSON rather than duplicated constants. Replace the invalid assumption “shorter geometry always has smaller communication-weighted cost” with separate assertions that geometric flight time is monotone under a constant synthetic radio field and real-map cost components are finite/deterministic.

- [ ] **Step 2: Run the repaired tests as a fresh baseline**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/test_metrics.py tests/test_radio_map.py tests/test_radio_path.py -v`

Expected: the three previously failing assertions now PASS; newly added determinism/scalar regressions expose any remaining issue before later refactors.

- [ ] **Step 3: Harden scalar extraction without changing radio values**

Keep the merged `.item()` extraction in both radio-map modules. Add fixed-position output regression and warning-as-error coverage so later changes cannot reintroduce the NumPy failure or alter radio values unnoticed.

- [ ] **Step 4: Run the full suite with warnings treated as errors for the repaired category**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests -q -W error::DeprecationWarning`

Expected: all tests PASS; no array-to-scalar deprecation failures.

- [ ] **Step 5: Commit**

```bash
git add tests/test_metrics.py tests/test_radio_map.py tests/test_radio_path.py radio_map/radio_map_A2G.py radio_map/radio_map_G2A.py
git commit -m "test: reconcile scenarios and radio-map regressions"
```

### Task 2: Make the Synchronous Evaluator Scientifically Authoritative

**Files:**
- Modify: `segment/stc/types.py`
- Modify: `segment/stc/evaluator.py`
- Modify: `tests/stc/test_evaluator.py`
- Modify: `tests/stc/test_objective.py`
- Modify: `tests/stc/test_reliability.py`
- Create: `tests/stc/test_gate0_handcases.py`

**Interfaces:**
- Consumes: `STCScenario`, `SegmentLibrary`, `PlanSolution`, `PerSectorRadioField`, `ObjectiveSpec`.
- Produces: `ReferenceValues(t_ref_s: float, d_ref_s: float, source_hash: str)`; `ObjectiveSpec(reference: ReferenceValues, w_t: float, w_d: float)`; typed `TimelineSlot`; `evaluate_solution(..., objective_spec: ObjectiveSpec, keep_timelines: bool = True) -> EvaluationResult`.

- [ ] **Step 1: Write failing normalized-objective integration tests**

```python
def test_evaluation_result_uses_frozen_normalized_objective(): ...
def test_solver_inputs_cannot_omit_reference_values(): ...
def test_changing_reference_changes_score_not_raw_metrics(): ...
```

Delete the unnormalized fallback. Missing/nonpositive references must raise before search starts.

- [ ] **Step 2: Write failing slot-semantics tests**

```python
def test_completion_at_first_slot_is_delta_t_not_zero(): ...
def test_final_upload_slot_records_prebuffer_active_and_conflicting(): ...
def test_keep_timelines_false_omits_slots_but_preserves_metrics(): ...
def test_deadline_with_residual_buffer_records_upload_failure(): ...
```

`TimelineSlot` records `time_start_s`, `time_end_s`, `edge_id`, `candidate_id`, `position_m`, `airborne`, `pre_buffer_bits`, `post_buffer_bits`, `active_uploader`, `serving_sector`, `allocated_bandwidth_hz`, `rate_bps`, `uploaded_bits`, `g2a_outage`, `holding`, and `service_event`.

- [ ] **Step 3: Write hand-calculated Gate 0 fixtures**

Include: one UAV/one task/no payload; one UAV with exactly two upload slots; two UAVs sharing one sector for one slot; two UAVs on different sectors; one UAV waiting outside the arrival radius; departure delay with identical airborne outage ratio; zero-rate max-wait failure.

- [ ] **Step 4: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_evaluator.py tests/stc/test_objective.py tests/stc/test_reliability.py tests/stc/test_gate0_handcases.py -v`

Expected: new tests FAIL on objective, timing, typed timeline, and final-slot semantics.

- [ ] **Step 5: Implement one authoritative state transition and score path**

Compute raw metrics once, then score only through `ObjectiveSpec`. Count activity from pre-slot buffer, apply service events after upload/motion, and use `(n+1)*delta_t_s` for slot completion. Do not duplicate objective formulas in solver or scripts.

- [ ] **Step 6: Run focused and full tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_evaluator.py tests/stc/test_objective.py tests/stc/test_reliability.py tests/stc/test_gate0_handcases.py -v`

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests -q`

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add segment/stc/types.py segment/stc/evaluator.py tests/stc/test_evaluator.py tests/stc/test_objective.py tests/stc/test_reliability.py tests/stc/test_gate0_handcases.py
git commit -m "fix: make synchronous evaluator authoritative"
```

### Task 3: Persist the Complete Segment Library and Seal Run Provenance

**Files:**
- Modify: `segment/stc/library.py`
- Modify: `segment/stc/run_context.py`
- Modify: `tests/stc/test_library.py`
- Modify: `tests/stc/test_run_context.py`
- Create: `tests/stc/test_provenance.py`

**Interfaces:**
- Consumes: complete `SegmentLibrary`, resolved scenario/config, radio-world hash, optional checkpoint hashes.
- Produces: `save_library(library, metadata, path) -> LibraryManifest`; `load_library(path, *, expected: LibraryCompatibility) -> SegmentLibrary`; sealed `RunContext` with declared expected artifacts.

- [ ] **Step 1: Write failing complete round-trip tests**

Assert every candidate scalar and array survives NPZ+JSON round-trip, candidate order and IDs are preserved, and content hashes change when any trace value changes.

- [ ] **Step 2: Write failing compatibility tests**

Reject mismatched schema version, scenario hash, task-content hash, map/world hash, radio parameters, generator configuration, and SI-unit version. Checking only seed equality is insufficient.

- [ ] **Step 3: Write failing provenance/sealing tests**

```python
def test_run_records_resolved_config_and_input_contents_not_only_paths(): ...
def test_dirty_diff_and_untracked_source_hashes_are_frozen(): ...
def test_finalize_fails_when_expected_artifact_missing(): ...
def test_finalized_context_rejects_further_writes_or_refinalize(): ...
def test_manifest_write_is_atomic(): ...
```

- [ ] **Step 4: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_library.py tests/stc/test_run_context.py tests/stc/test_provenance.py -v`

Expected: FAIL on compatibility/content-hash and sealed-context assertions; basic candidate-array round-trip should already pass after `90cb8ba`.

- [ ] **Step 5: Implement NPZ+JSON persistence and immutable finalization**

Manifest hashes must include source diff/untracked source bundle, resolved configs, task arrays, radio-world inputs, library NPZ/JSON, checkpoints used, raw records, search traces, summaries, and figures. `status="success"` requires all declared artifacts.

- [ ] **Step 6: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_library.py tests/stc/test_run_context.py tests/stc/test_provenance.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add segment/stc/library.py segment/stc/run_context.py tests/stc/test_library.py tests/stc/test_run_context.py tests/stc/test_provenance.py
git commit -m "fix: persist libraries and seal experiment provenance"
```

### Task 4: Complete Edge-Time-Sector Conflict Attribution

**Files:**
- Modify: `segment/stc/conflicts.py`
- Modify: `segment/stc/types.py`
- Modify: `tests/stc/test_conflicts.py`
- Create: `tests/stc/test_conflict_windows.py`

**Interfaces:**
- Consumes: typed timelines from Task 2 and `ObjectiveSpec`.
- Produces: `build_conflict_report(result, scenario) -> ConflictReport` with total overlap, lost bits, per-UAV loss, `edge_hotspots: dict[EdgeOccurrence, float]`, and merged `PairwiseConflictWindow` records.

- [ ] **Step 1: Write failing last-slot and edge-attribution tests**

Assert a UAV whose buffer becomes zero in a slot is still an active competitor for that slot; every lost bit is assigned to one UAV-edge occurrence; system loss equals the sum of per-UAV and per-edge losses within tolerance.

- [ ] **Step 2: Write failing pairwise-window tests**

Adjacent conflict slots for the same UAV pair/sector merge into one interval; sector or edge changes split windows; empty-buffer overlap creates no window.

- [ ] **Step 3: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_conflicts.py tests/stc/test_conflict_windows.py -v`

Expected: FAIL because current `edge_hotspots` and `pairwise_windows` are empty placeholders.

- [ ] **Step 4: Implement attribution from typed timeline slots**

Use exclusive rate minus allocated shared rate for lost-bit accounting. Key hotspots by `(uav_id, route_edge_index, source, target, candidate_id)` so destroy/path-switch operators can target an exact occurrence.

- [ ] **Step 5: Run Gate 0 evaluator/conflict tests together**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_gate0_handcases.py tests/stc/test_conflicts.py tests/stc/test_conflict_windows.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/conflicts.py segment/stc/types.py tests/stc/test_conflicts.py tests/stc/test_conflict_windows.py
git commit -m "feat: complete edge-time-sector conflict attribution"
```

## Phase B — Make the Core Algorithm Match Its Name

### Task 5: Replace Prototype Baselines and Operators With Real Search Methods

**Files:**
- Modify: `segment/stc/operators.py`
- Modify: `segment/stc/stc_operators.py`
- Modify: `tests/stc/test_initial_solution.py`
- Modify: `tests/stc/test_baseline_search.py`
- Modify: `tests/stc/test_operators.py`
- Create: `tests/stc/test_operator_marginals.py`

**Interfaces:**
- Consumes: Task 2 evaluator, Task 4 conflict report, full evaluator callback, bounded departure set.
- Produces: real NN+2-opt, cheapest-insertion+local-search, LNS-SA baselines; complete destroy/repair/local operator registries; `AblationConfig` controlling each mechanism.

- [ ] **Step 1: Write failing baseline-behavior tests**

Assert cheapest insertion is constructed point-by-point from an empty solution rather than returning NN split; LNS-SA performs destroy+repair and probabilistic acceptance; fixed seed reproduces its trace; all final metrics come from the synchronous evaluator.

- [ ] **Step 2: Write failing operator-completeness tests**

Require destroy operators: random, worst cost, critical route, conflict hotspot, sector related, infeasibility. Require repair operators: cheapest feasible, regret-2, makespan balancing, conflict-aware marginal, sector diversity. Require local operators: relocate, swap, 2-opt, route-tail exchange, path switch, bounded departure shift, conflict-window shift.

- [ ] **Step 3: Write failing synchronous-marginal tests**

On a synthetic case where independent edge cost prefers insertion A but shared-bandwidth evaluation prefers B, assert conflict-aware repair chooses B by evaluating candidate deltas. A static `1+0.1*per_uav_loss` multiplier must fail this test.

- [ ] **Step 4: Write failing departure/path bounds tests**

Departure moves must use `0..departure_max_s/departure_step_s`; no hard-coded modulo 10. Path switching may select only rankable candidates compatible with that exact directed edge.

- [ ] **Step 5: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_initial_solution.py tests/stc/test_baseline_search.py tests/stc/test_operators.py tests/stc/test_operator_marginals.py -v`

Expected: FAIL because current baselines and registries implement only a small subset and conflict repair uses a static penalty.

- [ ] **Step 6: Implement immutable complete operators and ablation switches**

Each operator returns a new valid `PlanSolution`; each ablation disables exactly one registry/mechanism without altering other solver parameters.

- [ ] **Step 7: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_initial_solution.py tests/stc/test_baseline_search.py tests/stc/test_operators.py tests/stc/test_operator_marginals.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/operators.py segment/stc/stc_operators.py tests/stc/test_initial_solution.py tests/stc/test_baseline_search.py tests/stc/test_operators.py tests/stc/test_operator_marginals.py
git commit -m "feat: implement complete STC baselines and operators"
```

### Task 6: Connect Conflict Feedback, Adaptive Weights, and Checkpointable Search

**Files:**
- Modify: `segment/stc/solver.py`
- Modify: `segment/stc/types.py`
- Modify: `tests/stc/test_solver.py`
- Create: `tests/stc/test_solver_conflict_feedback.py`
- Create: `tests/stc/test_solver_resume.py`

**Interfaces:**
- Consumes: completed operators, `build_conflict_report`, `ObjectiveSpec`, evaluation backend.
- Produces: `STCALNS.solve(initial, objective_spec, iterations, method_seed, ablation, checkpoint=None) -> SearchResult`; serializable `SolverCheckpoint`; trace records including operator weights, feasibility rank, raw/normalized metrics, accepted move, conflict loss, and evaluator counts.

- [ ] **Step 1: Write failing conflict-context tests**

Assert solver constructs a conflict report for the incumbent, refreshes it after accepted moves, passes it to hotspot destroy/repair, and targets a known synthetic hotspot rather than random route tail removal.

- [ ] **Step 2: Write failing normalized acceptance tests**

Create two feasible solutions whose ordering differs between unnormalized and normalized objectives. Assert solver follows `ObjectiveSpec` and preserves best feasible separately from SA incumbent.

- [ ] **Step 3: Write failing weight/ablation tests**

Assert productive operators gain probability, disabled mechanisms are never selected, probabilities remain normalized, and each trace segment stores the applied weights.

- [ ] **Step 4: Write failing checkpoint/resume tests**

An interrupted run resumed from `SolverCheckpoint` must exactly match uninterrupted final solution, weights, RNG state, trace suffix, and evaluator counters.

- [ ] **Step 5: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_solver.py tests/stc/test_solver_conflict_feedback.py tests/stc/test_solver_resume.py -v`

Expected: FAIL because current context keeps `conflict=None`, uses evaluator’s unnormalized objective, and has no checkpoints.

- [ ] **Step 6: Implement conflict-fed adaptive search**

Update conflict context only from authoritative results. Candidate marginal evaluations may use full replay until Task 7; do not fake incremental counts.

- [ ] **Step 7: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_solver.py tests/stc/test_solver_conflict_feedback.py tests/stc/test_solver_resume.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/solver.py segment/stc/types.py tests/stc/test_solver.py tests/stc/test_solver_conflict_feedback.py tests/stc/test_solver_resume.py
git commit -m "feat: connect conflict feedback to adaptive STC-ALNS"
```

### Task 7: Implement Exact Affected-Window Evaluation

**Files:**
- Modify: `segment/stc/incremental.py`
- Modify: `segment/stc/evaluator.py`
- Modify: `tests/stc/test_incremental.py`
- Create: `tests/stc/test_incremental_propagation.py`

**Interfaces:**
- Consumes: base solution/result/timelines, moved solution, exact affected UAV/edge hints.
- Produces: `EvaluationCache.evaluate_after_move(solution, affected_uavs, earliest_changed_edge) -> EvaluationResult`; counters `full_calls`, `incremental_calls`, `recomputed_slots`, `cache_hits`, `audit_calls`.

- [ ] **Step 1: Write failing no-full-fallback test**

Monkeypatch `evaluate_solution` after cache construction to raise. A local path switch with a bounded affected window must still return the exact result without calling full replay.

- [ ] **Step 2: Write failing propagation tests**

Cover a changed UAV entering/leaving another UAV’s sector, changed completion time shifting later conflicts, and departure shift moving the whole route. Recompute every interacting window until a fixed point; compare with independent full replay at `1e-9` scalar tolerance and exact discrete-event equality.

- [ ] **Step 3: Write periodic-audit failure test**

Inject a corrupted cached slot and assert the configured full-replay audit raises, saves a diagnostic, and prevents solver continuation.

- [ ] **Step 4: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_incremental.py tests/stc/test_incremental_propagation.py -v`

Expected: FAIL because current implementation always invokes full evaluation and never counts recomputed slots.

- [ ] **Step 5: Implement correctness-first affected-window recomputation**

Reuse the evaluator’s slot transition function; do not maintain a second physics implementation. Update solver to pass precise affected hints from operators.

- [ ] **Step 6: Run tests and a deterministic micro-benchmark**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_incremental.py tests/stc/test_incremental_propagation.py -v`

Expected: PASS; local path switch recomputes fewer slots than full replay.

- [ ] **Step 7: Commit**

```bash
git add segment/stc/incremental.py segment/stc/evaluator.py tests/stc/test_incremental.py tests/stc/test_incremental_propagation.py
git commit -m "feat: implement exact affected-window evaluation"
```

### Task 8: Strengthen the Graph/SLSQP Segment Baselines Without Partial CKM

**Files:**
- Modify: `segment/stc/graph_search.py`
- Modify: `segment/stc/slsqp_refine.py`
- Modify: `segment/stc/library.py`
- Modify: `tests/stc/test_graph_search.py`
- Modify: `tests/stc/test_slsqp_refine.py`
- Create: `tests/stc/test_segment_pareto.py`

**Interfaces:**
- Consumes: complete radio field, source payload, `ObjectiveSpec` segment weights, grid/buffer discretization config.
- Produces: `search_state_expanded_paths(...) -> list[SegmentCandidate]`; `generate_pareto_segment_candidates(weight_grid, ...) -> tuple[SegmentCandidate, ...]`; exact-replayed SLSQP refinements with explicit status.

- [ ] **Step 1: Write failing buffer-state graph tests**

Use nodes `(grid_x, grid_y, remaining_payload_bin)`. Edge transition decrements payload using the selected sector’s full-bandwidth rate and adds flight/outage cost. Assert the returned path reaches the target service anchor only at zero payload and is optimal on a small hand-enumerated graph.

- [ ] **Step 2: Write failing nonnegative/admissible A* tests**

All graph costs must be nonnegative; heuristic cannot exceed remaining straight-flight lower bound. Dijkstra and A* must return identical cost on the same finite graph.

- [ ] **Step 3: Write failing Pareto-sweep tests**

Sweep predeclared time/delay/outage weights and retain nondominated, sector-diverse candidates. Assert one edge can retain a fast/high-conflict path and a slower/low-conflict path; `min_sector_distance` must affect pruning instead of being ignored.

- [ ] **Step 4: Write failing SLSQP validation tests**

Refinement accepts only candidates whose exact single-UAV replay improves the declared segment objective and remains feasible. Replace broad `except Exception` swallowing with typed failure metadata.

- [ ] **Step 5: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_graph_search.py tests/stc/test_slsqp_refine.py tests/stc/test_segment_pareto.py -v`

Expected: FAIL because current graph has only spatial nodes, hard-coded `0.1` weighting, one result, and no buffer feasibility.

- [ ] **Step 6: Implement full-CKM graph and Pareto candidate generation**

This task uses the complete known map only. Do not add interpolation uncertainty, Kriging, measurement rewards, or CKM-update rounds.

- [ ] **Step 7: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_graph_search.py tests/stc/test_slsqp_refine.py tests/stc/test_segment_pareto.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/graph_search.py segment/stc/slsqp_refine.py segment/stc/library.py tests/stc/test_graph_search.py tests/stc/test_slsqp_refine.py tests/stc/test_segment_pareto.py
git commit -m "feat: add full-CKM Pareto graph segment optimizer"
```

### Task 9: Replace the False Exact Solver With a Guarded Discrete Benchmark

**Files:**
- Modify: `segment/stc/exact.py`
- Modify: `experiment/run_stc_exact.py`
- Modify: `tests/stc/test_exact.py`
- Create: `tests/stc/test_exact_multicandidate.py`
- Modify: `docs/stc-method-analysis.md`

**Interfaces:**
- Consumes: bounded tiny scenario, persisted candidate library, departure set, `ObjectiveSpec`, time limit.
- Produces: `solve_exact_tiny(...) -> ExactResult` with finite admissible bound when available, correct gap, proven flag, explored/pruned branches, timeout status, and run artifacts.

- [ ] **Step 1: Write a pre-allocation size-guard test**

`scenario.n_tasks > max_tasks` must raise `ValueError` before any call to permutations/product or evaluator. The default CLI must either load an explicitly tiny scenario or refuse the full 20-task scene.

- [ ] **Step 2: Write independent multi-candidate/multi-UAV/multi-departure oracle tests**

The test’s brute-force oracle must not call `_pick_candidate` or production branch helpers. Construct a case where a non-cheapest independent candidate is globally best under contention and assert the exact solver selects it.

- [ ] **Step 3: Write timeout/bound/gap tests**

Use lazy iterators and frequent deadline checks. A timeout returns an incumbent, a valid lower bound no greater than it, finite gap when both are finite, and `proven_optimal=False`. A completed search reports zero gap only after covering/pruning the declared domain.

- [ ] **Step 4: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_exact.py tests/stc/test_exact_multicandidate.py -v`

Expected: FAIL on lazy-search, finite-bound/gap, normalized-objective, and independent-oracle assertions. The merged size guard and basic candidate enumeration should already pass.

- [ ] **Step 5: Implement lazy branch-and-bound and exact run artifacts**

Branch over assignment/order, candidate per traversed edge, and departure shifts. Use independent no-contention lower costs as an admissible relaxation. Persist resolved domain, incumbent, bound history, solution, search trace, and manifest.

- [ ] **Step 6: Correct method-analysis claims**

Document exactly which discrete space is certified, the bound derivation, and the largest instance actually completed. Remove unsupported tractability statements.

- [ ] **Step 7: Run focused tests and a two-task CLI smoke**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_exact.py tests/stc/test_exact_multicandidate.py -v`

Run: `F:/anaconda3/envs/pytorch/python.exe experiment/run_stc_exact.py --max-tasks 2 --time-limit-s 60 --run-id exact-smoke-<unique>`

Expected: tests PASS; run contains solution, trace, summary, and sealed manifest.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/exact.py experiment/run_stc_exact.py tests/stc/test_exact.py tests/stc/test_exact_multicandidate.py docs/stc-method-analysis.md
git commit -m "fix: implement honest discrete exact benchmark"
```

## Phase C — Produce Real Scientific Evidence

### Task 10: Repair Hierarchical Statistics and Failure Pairing

**Files:**
- Modify: `segment/stc/stats.py`
- Modify: `tests/stc/test_stats.py`
- Create: `tests/stc/test_stats_pairing.py`

**Interfaces:**
- Consumes: records keyed by scenario, method, replicate, metric, and feasibility.
- Produces: `hierarchical_paired_summary(records, method_a, method_b, metric, scenario_keys, replicate_key, failure_policy) -> PairedSummary`; `prespecified_contrasts(...)`; Holm-adjusted contrast table.

- [ ] **Step 1: Write failing explicit-contrast tests**

Require `method_a` and `method_b`; do not infer sorted method names independently per scenario. A three-method dataset must produce each requested contrast rather than silently disappear.

- [ ] **Step 2: Write failing replicate-completeness tests**

Reject duplicate scenario/method/replicate records and disjoint replicate sets unless an explicit aggregation policy is supplied. Aggregate method seeds within scenario before treating scenarios as independent samples.

- [ ] **Step 3: Write failing nonfinite/failure-policy tests**

Reject NaN/Inf metric input. Support predeclared policies: success-rate comparison on all paired instances; cost comparison on common-feasible instances; penalized objective on all instances. Never silently drop failures.

- [ ] **Step 4: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_stats.py tests/stc/test_stats_pairing.py -v`

Expected: FAIL because current code still ignores `replicate_key`, accepts infinity/duplicates/disjoint replicates, and lacks an explicit failure policy. Explicit method-pair selection should already pass after `90cb8ba`.

- [ ] **Step 5: Implement deterministic hierarchical contrasts and Holm correction**

Store contrast definitions and failure policy in the returned object so plots/tables cannot reinterpret the sample later.

- [ ] **Step 6: Run focused tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_stats.py tests/stc/test_stats_pairing.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add segment/stc/stats.py tests/stc/test_stats.py tests/stc/test_stats_pairing.py
git commit -m "fix: enforce valid hierarchical paired inference"
```

### Task 11: Implement Real RQ1 and RQ2 Runners

**Files:**
- Rewrite: `experiment/run_stc_segment_gate.py`
- Rewrite: `experiment/run_stc_evaluator_audit.py`
- Modify: `tests/stc/test_experiment_cli.py`
- Create: `tests/stc/test_rq1_pipeline.py`
- Create: `tests/stc/test_rq2_pipeline.py`

**Interfaces:**
- Consumes: resolved config, scenario seeds, persisted library, learned baseline checkpoints when requested, evaluator/reference values.
- Produces: finalized run directories with per-edge/per-solution raw records, paired summaries, failure reasons, timings, gate decisions, and manifests.

- [ ] **Step 1: Write failing config/input rejection tests**

Nonexistent config, library, checkpoint, or incompatible hashes must exit nonzero and must not finalize success. `--dry-run` resolves and prints the full matrix without generating a map.

- [ ] **Step 2: Write failing RQ1 artifact tests**

Require records for Straight-MaxSpeed, Straight-Hold, Comm-Waypoint, state-expanded graph, SLSQP, TD3, DDPG, and SAC on identical edge IDs. Save full-set feasibility, common-feasible paired raw metrics, failure categories, normalized score, and compute time.

- [ ] **Step 3: Write failing RQ2 artifact tests**

For identical solutions, require exclusive, static-factor, and synchronous outputs; approximation errors; feasibility/ranking reversals; and explicit “none found” when no reversal exists.

- [ ] **Step 4: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_experiment_cli.py tests/stc/test_rq1_pipeline.py tests/stc/test_rq2_pipeline.py -v`

Expected: FAIL because current runners only parse/print arguments.

- [ ] **Step 5: Implement both runners using RunContext expected artifacts**

The runners call library load/build, objective reference, evaluator, statistics, and manifest APIs; they contain no duplicate physics or objective formulas.

- [ ] **Step 6: Run deterministic tiny pipeline tests and one real smoke per RQ**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_experiment_cli.py tests/stc/test_rq1_pipeline.py tests/stc/test_rq2_pipeline.py -v`

Expected: PASS; real smoke directories contain raw records and nonempty metrics.

- [ ] **Step 7: Commit**

```bash
git add experiment/run_stc_segment_gate.py experiment/run_stc_evaluator_audit.py tests/stc/test_experiment_cli.py tests/stc/test_rq1_pipeline.py tests/stc/test_rq2_pipeline.py
git commit -m "feat: implement RQ1 and RQ2 experiment runners"
```

### Task 12: Implement Main, Ablation, Scale, Plotting, and Gate Orchestration

**Files:**
- Rewrite: `experiment/run_stc_main.py`
- Rewrite: `experiment/run_stc_ablation.py`
- Rewrite: `experiment/run_stc_scale.py`
- Replace/commit: `experiment/run_stc_gates.py`
- Rewrite: `plots/plot_stc_results.py`
- Modify: `tests/stc/test_experiment_cli.py`
- Modify: `tests/stc/test_plot_stc_results.py`
- Create: `tests/stc/test_main_pipeline.py`
- Create: `tests/stc/test_ablation_pipeline.py`
- Create: `tests/stc/test_scale_matrix.py`
- Create: `tests/stc/test_gate_orchestrator.py`
- Modify: `docs/stc-experiment-protocol.md`
- Modify: `仿真实验.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed core APIs and finalized RQ/exact runs.
- Produces: real RQ3–RQ5 runs, ablation contrasts, controlled scale families, figures/tables with provenance, machine-readable Gate 0–4 decisions.

- [ ] **Step 1: Write failing main-pipeline artifact tests**

Require NN+2-opt, cheapest insertion, LNS-SA, static-factor iterative LNS, and STC-ALNS on identical paired scenarios. Every record contains feasibility, raw T/D/outage/conflict loss, normalized J, runtime, evaluator counters, library/reference hashes, and failure reason.

- [ ] **Step 2: Write failing ablation-isolation tests**

Each run disables exactly one of: adaptive weights, hotspot destroy, conflict-aware repair, path switch, departure shift, multi-candidate library, or incremental evaluation. Resolved configurations must differ only at the intended switch.

- [ ] **Step 3: Write failing controlled-scale tests**

“Vary UAV count” holds map, area, BSs, tasks, payloads, and library family fixed. “Vary task scale” changes task/area/BS density through a separate declared matrix. Communication-load and power sweeps rebuild their libraries. No Partial-CKM family is accepted.

- [ ] **Step 4: Write failing plotting/freeze tests**

Plotting rejects missing/unfinalized/mixed-hash runs; uses no hard-coded numbers; writes adjacent provenance JSON. Deleting generated figures and rerunning must reproduce identical data hashes.

- [ ] **Step 5: Write failing gate-orchestrator tests**

Gate 0 reads test/evaluator evidence; Gate 1 reads RQ1 paired results; Gate 2 reads main contrasts; Gate 3 requires at least two prespecified ablation contributions; Gate 4 validates complete manifests. A deferred/missing input is `NOT_RUN`, never PASS.

- [ ] **Step 6: Run tests and verify current failures**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_main_pipeline.py tests/stc/test_ablation_pipeline.py tests/stc/test_scale_matrix.py tests/stc/test_plot_stc_results.py tests/stc/test_gate_orchestrator.py -v`

Expected: FAIL because current files are placeholders and the untracked gate script defers Gate 3.

- [ ] **Step 7: Implement real runners with progress, timeout, and checkpoint logging**

Emit scenario/method/replicate progress immediately (`flush=True`), estimated completed/total matrix count, solver checkpoints, and nonzero exit on incomplete expected output. No runner may print success solely because argument parsing succeeded.

- [ ] **Step 8: Implement figure/table regeneration and gate reports**

Generate Pareto T–D, T–conflict, D–outage, paired-effect CI, feasibility/failure, exact-gap, runtime/scaling, and ablation figures from raw finalized records.

- [ ] **Step 9: Rewrite active documentation to the implemented method only**

Remove stale TD3 improvement/generalization/natural-avoidance claims. Document full-known CKM, actual Gate state, exact baseline domain, seed hierarchy, failure policy, and commands. Explicitly state Partial-CKM is outside this paper.

- [ ] **Step 10: Run the full suite**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests -q`

Expected: all tests PASS.

- [ ] **Step 11: Run one bounded end-to-end pilot**

Run RQ1, RQ2, exact tiny, main, ablation, plot, and gate orchestration for one small predeclared scenario/method seed set. Every command must have a configured time limit and write progress/checkpoint artifacts.

Expected: nonempty raw metrics, figures, summaries, and `gate_report.json`; no shared mutable results; Gate 3 may fail scientifically but cannot be deferred technically.

- [ ] **Step 12: Run independent whole-branch review before formal matrix**

Review must compare this plan and design to the branch, inspect pilot artifacts, reproduce one result from raw inputs, and verify no placeholder/TODO/deferred behavior remains in formal paths.

- [ ] **Step 13: Commit**

```bash
git add experiment/run_stc_main.py experiment/run_stc_ablation.py experiment/run_stc_scale.py experiment/run_stc_gates.py plots/plot_stc_results.py tests/stc/test_experiment_cli.py tests/stc/test_plot_stc_results.py tests/stc/test_main_pipeline.py tests/stc/test_ablation_pipeline.py tests/stc/test_scale_matrix.py tests/stc/test_gate_orchestrator.py docs/stc-experiment-protocol.md 仿真实验.md README.md
git commit -m "feat: complete formal STC experiment and gate pipeline"
```

## Phase Gates and Stop Rules

### Gate A — Before Core Algorithm Work

- Full repository tests pass.
- Gate 0 hand cases pass.
- Normalized objective is the only optimization score.
- Library persistence and run sealing pass adversarial tests.
- Edge hotspots and conflict windows are nonempty on a synthetic conflict case.

If Gate A fails, do not modify STC-ALNS operators or run performance experiments.

### Gate B — Before Experiment Runner Work

- Conflict report is actually supplied to operators.
- Synchronous marginal repair passes the reversal fixture.
- Incremental evaluation equals full replay and recomputes fewer slots locally.
- State-expanded graph produces upload-feasible Pareto candidates.
- Exact solver passes independent multi-candidate enumeration and size guards.

If Gate B fails, do not implement or run the formal experiment matrix.

### Gate C — Before Formal Multi-Seed Matrix

- RQ1/RQ2 real smoke artifacts exist and are reproducible.
- Main/ablation pilot produces raw records for every expected method.
- Statistics reject broken pairing and nonfinite metrics.
- Plotting reproduces figures from finalized raw data.
- Gate orchestrator reports explicit PASS/FAIL/NOT_RUN with source run IDs.
- Independent whole-branch review has no Critical or Important findings.

Only after Gate C approval may the expensive formal matrix start.

## Recommended Execution and Review Policy

Use subagent-driven development exactly one task at a time. After each task, run its focused tests and dispatch a fresh read-only reviewer against that task’s commit. Stop on any Critical/Important finding. Require whole-branch reviews after Task 4 (Gate A), Task 9 (Gate B), and Task 12 (Gate C). Do not parallelize tasks that modify `types.py`, `evaluator.py`, `solver.py`, or experiment schemas; their interfaces are sequential dependencies.
