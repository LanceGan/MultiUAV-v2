# Final Multi-UAV Communication-Aware Inspection Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute the final full-CKM, simultaneous-takeoff multi-UAV inspection study: all-pairs communication-aware segment generation, EBTAS-inspired balanced assignment, communication-aware sequencing, synchronous shared-bandwidth evaluation, and conflict-feedback STC-ALNS.

**Architecture:** Precompute a persisted Pareto candidate library for every legal directed edge before assignment. Build an EBTAS-inspired balanced initial allocation and a communication-aware visiting order, concatenate the selected segments, then replay every UAV from `t=0` on one synchronous clock. Attribute edge-time-sector conflicts and let STC-ALNS refine task assignment, visit order, and segment choice; no departure-time decision exists.

**Tech Stack:** Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, PyTorch 2.8.0 CPU for learned baselines, pytest 9.1.1, Matplotlib.

**Spec:** This plan is the authoritative final experiment contract and supersedes the departure-shift portions of `docs/superpowers/specs/2026-10-23-tvt-spatiotemporal-planning-design.md` and all earlier implementation plans.

## 1. Final Research Contract

### 1.1 Fixed scenario assumptions

- The complete A2G uplink and G2A downlink CKMs are known before planning.
- Partial-CKM, Kriging, diffusion reconstruction, and online CKM completion are outside this paper.
- Energy consumption `E` is outside the objective, constraints, and formal result tables for this paper.
- Every UAV starts at `t=0`; takeoff staggering is prohibited for the proposed method and all baselines.
- Each UAV starts from the configured depot, visits its assigned inspection points, uploads the last payload, and reaches the configured terminal.
- Inspection at point `k` immediately generates `I_k` bits.
- The payload generated at point `i` must be completely uploaded before the next service-anchor event.
- All UAVs use the same slot duration and are evaluated on one global clock.
- Every airborne UAV accrues G2A outage; only UAVs with pre-slot nonempty buffers consume uplink bandwidth.

### 1.2 Legal directed edges

For `K` inspection points, build exactly `K(K+1)` legal directed edges:

- depot to every task: `K` edges, zero source payload;
- task `i` to task `j`, `i != j`: `K(K-1)` edges, payload `I_i`;
- every task to terminal: `K` edges, payload `I_i`.

No self loops and no direct depot-to-terminal edge are part of the task edge library; an empty UAV route uses a separately declared idle trace, not a task edge.

### 1.3 Decision variables

The final solution contains only:

- task assignment to UAVs;
- visit order within each UAV route;
- candidate segment choice for each traversed directed edge.

Remove all departure-shift variables, configuration values, exact-solver branches, operators, and ablations.

### 1.4 Objective and constraints

Optimize

```text
J = w_T * T/T_ref + w_D * D/D_ref,
w_T + w_D = 1.
```

Where:

- `T = max_u T_u` is system makespan;
- `D = sum_k D_k` is total payload completion delay;
- `T_ref`, `D_ref` are frozen reference values shared by all methods in an experiment family.

Hard feasibility includes task coverage, no duplicate task, complete upload before the next service event, last-payload completion, route-level G2A outage threshold, maximum pre-arrival waiting time, and mission deadline.

### 1.5 Reported secondary metrics

- system success rate;
- per-UAV completion times `T_u`;
- average completion time `T_avg`;
- completion-time standard deviation `sigma_T`;
- completion-time range `max(T_u)-min(T_u)`;
- total and average upload delay;
- per-UAV G2A outage ratio and maximum consecutive outage;
- shared-bandwidth lost bits and conflict duration;
- waiting time;
- planning/runtime cost and evaluator calls.

## 2. Method Lineage and Positioning

### 2.1 Jia-2025 lineage retained

Retain these ideas as first-class components and baselines:

- balanced task assignment using geometry, payload volume, and communication conditions;
- simultaneous start and balanced completion times;
- communication-aware directed TSP edge weights;
- decomposition into task assignment, visit sequence, and adjacent-point trajectory subproblems;
- data generated at inspection points and completed before the next point.

The proposed method improves this lineage by using precomputed optimized segment profiles before assignment/sequencing, complete complex-environment CKMs, multiple Pareto segments per edge, explicit shared bandwidth, and closed-loop conflict feedback.

### 2.2 GATSAC lineage retained as a baseline

Retain the structural baseline:

```text
K-means assignment -> communication-aware TSP/3-opt -> SAC local trajectories.
```

Use complete CKM for fairness and call it `GATSAC-style` unless the authors' exact code/models are available. Do not reproduce its Partial-CKM/diffusion component.

### 2.3 Proposed method

```text
all-pairs Pareto segment library
-> EBTAS-inspired balanced initial assignment
-> communication-aware initial sequence
-> simultaneous synchronous evaluation
-> edge-time-sector conflict attribution
-> STC-ALNS assignment/order/segment refinement.
```

## 3. Global Implementation Constraints

- New formal code remains under `segment/stc/`; legacy `assignment/` and `sequence/` are adapters/baseline sources, not the authoritative evaluator.
- SI units inside `segment/stc/`; conversions only at scenario/radio boundaries.
- One authoritative objective path via `ObjectiveSpec`.
- One authoritative physics path via `evaluate_solution`.
- Every formal runner writes only to `results/runs/<run_id>/` and cannot overwrite an existing run.
- Formal success requires expected artifacts, resolved inputs, source/input hashes, raw records, summaries, and a sealed manifest.
- Formal plots read only finalized compatible runs and contain no hard-coded result values.
- Frozen v0/v1/v2 directories are read-only historical artifacts.
- Do not launch the formal matrix until the pilot and independent branch review pass.

## 4. File Responsibility Map

- `segment/stc/types.py`: final no-departure data contracts.
- `segment/stc/scenario.py`: SI scenario and full-known map/task seeds.
- `segment/stc/radio.py`: per-sector full CKM.
- `segment/stc/generators.py`, `graph_search.py`, `slsqp_refine.py`: segment candidates.
- `segment/stc/library.py`: complete directed Pareto library persistence.
- `segment/stc/assignment.py`: region, spatial, Jia-EBTAS, and library-aware balanced assignment.
- `segment/stc/sequencing.py`: distance 3-opt, Jia time-energy order, GA-EQTSP adapter, and library-aware sequence.
- `segment/stc/evaluator.py`: simultaneous slot-synchronous evaluation.
- `segment/stc/conflicts.py`: edge-time-sector conflict report.
- `segment/stc/incremental.py`: exact affected-window evaluation.
- `segment/stc/operators.py`: resource-unaware search baselines.
- `segment/stc/stc_operators.py`, `solver.py`: STC-ALNS.
- `segment/stc/exact.py`: tiny discrete optimum/bound.
- `segment/stc/stats.py`: hierarchical paired inference.
- `segment/stc/run_context.py`: provenance and freezing.
- `experiment/run_stc_*.py`: RQ1-RQ5 and gate execution.
- `plots/plot_stc_results.py`: formal figures/tables.

## 5. Review Focus

1. **Simultaneous start:** every method must have first airborne slot at `t=0`; no hidden waiting, departure field, or staggered baseline.
2. **Directed source-payload semantics:** `i->j` carries `I_i`; depot edges carry zero; terminal edges carry the last task payload.
3. **Shared-bandwidth final slot:** a payload draining in the current slot still consumed bandwidth and caused conflict in that slot.
4. **Baseline isolation:** Jia-style and GATSAC-style adapters must not silently use proposed conflict feedback or multi-candidate refinement.
5. **Claim integrity:** no PASS, optimality, significance, or figure is accepted without finalized raw evidence and compatible hashes.

---

## Phase I - Finalize the Model and Reusable Planning Layers

### Task 1: Remove Departure-Time Decisions and Enforce Simultaneous Takeoff

**Files:**
- Modify: `configs/stc/base.json`
- Modify: `segment/stc/types.py`
- Modify: `segment/stc/scenario.py`
- Modify: `segment/stc/evaluator.py`
- Modify: `segment/stc/operators.py`
- Modify: `segment/stc/stc_operators.py`
- Modify: `segment/stc/solver.py`
- Modify: `segment/stc/exact.py`
- Modify: all affected `tests/stc/test_*.py`
- Create: `tests/stc/test_simultaneous_start.py`

**Interfaces:**
- Consumes: current scenario/solution/evaluator interfaces.
- Produces: `PlanSolution(routes, candidate_ids)` with no departure field; every UAV active from global slot zero.

- [ ] **Step 1: Write failing simultaneous-start contract tests**

```python
def test_plan_solution_has_no_departure_field(): ...
def test_every_nonempty_uav_starts_at_global_slot_zero(): ...
def test_idle_uav_does_not_consume_bandwidth(): ...
def test_exact_domain_contains_no_departure_dimension(): ...
def test_operator_registry_contains_no_departure_or_time_shift(): ...
```

- [ ] **Step 2: Run tests and verify failure**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_simultaneous_start.py -v`

Expected: FAIL because current contracts still contain `departure_steps` and departure configuration.

- [ ] **Step 3: Remove departure state throughout the call graph**

Delete `departure_step_s`, `departure_max_s`, `departure_steps`, `departure_set`, `departure_shift`, and corresponding trace/ablation fields. Evaluator initializes every non-idle UAV as airborne at slot zero.

- [ ] **Step 4: Update Gate 0 hand cases**

Remove delayed-takeoff tests. Retain one/two UAV upload, different/same sector, bounded waiting, zero-rate failure, and final-upload-slot conflict cases.

- [ ] **Step 5: Run full tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests -q`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/stc/base.json segment/stc tests/stc
git commit -m "refactor: enforce simultaneous multi-UAV takeoff"
```

### Task 2: Complete and Freeze the All-Pairs Pareto Segment Library

**Files:**
- Modify: `segment/stc/generators.py`
- Modify: `segment/stc/graph_search.py`
- Modify: `segment/stc/slsqp_refine.py`
- Modify: `segment/stc/library.py`
- Create: `tests/stc/test_edge_inventory.py`
- Create: `tests/stc/test_segment_payload_semantics.py`
- Create: `tests/stc/test_segment_pareto.py`
- Modify: `tests/stc/test_library.py`

**Interfaces:**
- Consumes: `STCScenario`, full `PerSectorRadioField`, generator config.
- Produces: `build_all_pairs_library(...) -> SegmentLibrary` containing exactly `K(K+1)` task edges and multiple rankable candidates where available.

- [ ] **Step 1: Write failing edge inventory tests**

Assert legal count `K(K+1)`, no self edge, no depot-terminal task edge, correct depot/terminal payload semantics, and distinct directed `i->j`/`j->i` payloads.

- [ ] **Step 2: Write failing candidate feasibility tests**

Require Straight-Hold, Comm-Waypoint, state-expanded A*/Dijkstra, and SLSQP-refined candidates. DRL candidates are optional baseline entries and must be labeled by model/checkpoint hash.

- [ ] **Step 3: Write failing Pareto/diversity tests**

Prune on flight time, upload delay, outage, and sector signature. Retain a slower candidate when it provides materially different sector occupancy; reject dominated duplicates.

- [ ] **Step 4: Write persistence compatibility tests**

Round trip all arrays/metadata and reject mismatched scenario, task content, map/world, radio parameters, SI schema, generator config, or checkpoint hash.

- [ ] **Step 5: Run tests and implement missing behavior**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_edge_inventory.py tests/stc/test_segment_payload_semantics.py tests/stc/test_segment_pareto.py tests/stc/test_library.py -v`

Expected after implementation: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/generators.py segment/stc/graph_search.py segment/stc/slsqp_refine.py segment/stc/library.py tests/stc
git commit -m "feat: finalize directed Pareto segment library"
```

### Task 3: Implement EBTAS-Inspired Balanced Assignment

**Files:**
- Create: `segment/stc/assignment.py`
- Create: `tests/stc/test_assignment.py`
- Create: `tests/stc/test_ebtas_adapter.py`
- Reference: `assignment/clustering.py`
- Reference: `assignment/baselines.py`

**Interfaces:**
- Consumes: task positions/data, point radio features, segment-library summaries, UAV count, seed.
- Produces: `AssignmentResult(labels, clusters, feature_summary, balance_metrics)`; methods `region_partition`, `spatial_kmeans`, `jia_ebtas`, and `library_aware_ebtas`.

- [ ] **Step 1: Write failing assignment invariants**

Assert each task appears exactly once, cluster count equals UAV count, deterministic seed behavior, maximum cluster-size enforcement, and support for empty clusters only when `K < U`.

- [ ] **Step 2: Reproduce Jia-style feature assignment**

Use geometry, payload volume, and point communication rate features with explicit normalization and configured weights. Match the existing `jia_balanced` result on a fixed fixture.

- [ ] **Step 3: Add library-aware assignment features**

Replace local proxy-only features with precomputed incident-edge summaries: median feasible flight time, upload delay, outage, and sector-diversity score. Keep the Jia-style method unchanged as a baseline.

- [ ] **Step 4: Test balance metrics**

Report payload sum, task count, proxy route cost, and their inter-UAV standard deviations. A synthetic high-payload case must assign more evenly than spatial-only K-means.

- [ ] **Step 5: Run tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_assignment.py tests/stc/test_ebtas_adapter.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add segment/stc/assignment.py tests/stc/test_assignment.py tests/stc/test_ebtas_adapter.py
git commit -m "feat: add EBTAS-inspired balanced task assignment"
```

### Task 4: Implement Communication-Aware Sequence Solvers

**Files:**
- Create: `segment/stc/sequencing.py`
- Create: `tests/stc/test_sequencing.py`
- Create: `tests/stc/test_jia_sequence_adapter.py`
- Reference: `sequence/radio_path.py`
- Reference: `sequence/ga_eqtsp.py`
- Reference: `sequence/baselines.py`

**Interfaces:**
- Consumes: one assigned task cluster, directed segment library, start/terminal.
- Produces: `SequenceResult(route, selected_candidate_ids, proxy_cost, method)` for distance NN/3-opt, Jia Time-Energy TSP adapter, GA-EQTSP adapter, and library-aware sequence.

- [ ] **Step 1: Write route validity and directed-cost tests**

Every assigned point appears exactly once; depot/terminal are fixed; reversing a route uses reversed directed costs and source payloads.

- [ ] **Step 2: Preserve Jia/GA-EQTSP baselines**

Wrap current implementations without changing their internal optimization criteria. Record their resolved edge weights and seeds for provenance.

- [ ] **Step 3: Implement library-aware ordering**

For each directed edge, use a frozen single-UAV proxy derived from rankable candidates; final ranking still uses the synchronous evaluator. Do not use proposed multi-UAV conflict feedback in baseline sequence solvers.

- [ ] **Step 4: Run tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_sequencing.py tests/stc/test_jia_sequence_adapter.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add segment/stc/sequencing.py tests/stc/test_sequencing.py tests/stc/test_jia_sequence_adapter.py
git commit -m "feat: add communication-aware visit sequencing"
```

## Phase II - Complete the Shared-Bandwidth Core Method

### Task 5: Finalize Synchronous Evaluation and Conflict Attribution

**Files:**
- Modify: `segment/stc/evaluator.py`
- Modify: `segment/stc/conflicts.py`
- Modify: `segment/stc/types.py`
- Modify: `tests/stc/test_gate0_handcases.py`
- Create: `tests/stc/test_conflict_windows.py`
- Create: `tests/stc/test_balance_metrics.py`

**Interfaces:**
- Consumes: final `PlanSolution`, library, field, `ObjectiveSpec`.
- Produces: raw/normalized metrics, typed timelines, balance metrics, edge hotspots, pairwise conflict windows, lost bits.

- [ ] **Step 1: Verify the authoritative slot sequence**

All UAVs move on the same global slot; association precedes bandwidth allocation; active upload is determined from pre-slot buffer; service occurs after motion/upload; completion time is slot end.

- [ ] **Step 2: Complete edge/window conflict attribution**

Key hotspots by UAV, route-edge occurrence, source, target, and candidate ID. Merge adjacent pair/sector conflict slots. Ensure system lost bits equals sums by UAV and edge.

- [ ] **Step 3: Add task-balance outputs**

Return `T_u`, `T_avg`, `sigma_T`, and completion-time range without adding them to the main objective.

- [ ] **Step 4: Run Gate 0 tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_gate0_handcases.py tests/stc/test_conflicts.py tests/stc/test_conflict_windows.py tests/stc/test_balance_metrics.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add segment/stc/evaluator.py segment/stc/conflicts.py segment/stc/types.py tests/stc
git commit -m "feat: finalize simultaneous shared-bandwidth evaluation"
```

### Task 6: Implement STC-ALNS Over Assignment, Order, and Segment Choice

**Files:**
- Modify: `segment/stc/operators.py`
- Modify: `segment/stc/stc_operators.py`
- Modify: `segment/stc/solver.py`
- Modify: `segment/stc/incremental.py`
- Create: `tests/stc/test_ebtas_initial_solution.py`
- Create: `tests/stc/test_operator_marginals.py`
- Create: `tests/stc/test_solver_conflict_feedback.py`
- Create: `tests/stc/test_incremental_propagation.py`

**Interfaces:**
- Consumes: EBTAS assignment, initial sequences, full candidate library, evaluator/conflicts.
- Produces: STC-ALNS solution and trace using task reassignment, order moves, and path switches only.

- [ ] **Step 1: Build proposed initial solution**

`build_ebtas_initial_solution` runs library-aware EBTAS and library-aware sequencing, then chooses one initial rankable candidate per edge.

- [ ] **Step 2: Complete operator registries**

Destroy: random, worst cost, critical route, conflict hotspot, sector related, infeasibility. Repair: cheapest feasible, regret-2, makespan balancing, synchronous conflict-aware, sector diversity. Local: relocate, swap, 2-opt, route-tail exchange, path switch. No departure/time-shift operator.

- [ ] **Step 3: Use real synchronous marginal evaluation**

On a fixture where independent edge cost prefers A but shared evaluation prefers B, conflict-aware repair must choose B.

- [ ] **Step 4: Complete exact incremental evaluation**

Recompute affected UAV timelines and interacting sector/time windows; periodic full replay must match within declared tolerance. Do not report full replay as incremental.

- [ ] **Step 5: Add adaptive weights and checkpoint/resume**

Trace operator weights, acceptance, feasibility rank, objective, conflict loss, and evaluator counters. Resumed and uninterrupted runs must match exactly.

- [ ] **Step 6: Run tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_ebtas_initial_solution.py tests/stc/test_operator_marginals.py tests/stc/test_solver.py tests/stc/test_solver_conflict_feedback.py tests/stc/test_incremental.py tests/stc/test_incremental_propagation.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add segment/stc/operators.py segment/stc/stc_operators.py segment/stc/solver.py segment/stc/incremental.py tests/stc
git commit -m "feat: implement simultaneous-start STC-ALNS"
```

### Task 7: Finalize Exact Benchmark, Statistics, and Provenance

**Files:**
- Modify: `segment/stc/exact.py`
- Modify: `segment/stc/stats.py`
- Modify: `segment/stc/run_context.py`
- Modify: `tests/stc/test_exact.py`
- Create: `tests/stc/test_exact_multicandidate.py`
- Create: `tests/stc/test_stats_pairing.py`
- Create: `tests/stc/test_provenance.py`

**Interfaces:**
- Produces: guarded tiny exact result without departure dimension; explicit hierarchical paired contrasts; sealed run manifests.

- [ ] **Step 1: Make exact search honest and lazy**

Enumerate assignment, order, and candidate choices only. Enforce `max_tasks` before allocations, use lazy branches/deadline checks, admissible no-contention bound, correct gap, and independent brute-force oracle tests.

- [ ] **Step 2: Enforce valid paired statistics**

Require explicit method pairs, unique scenario/method/replicate keys, complete replicate pairing, finite metrics, and declared failure policy. Aggregate method seeds within scenario before scenario-level inference; apply Holm correction.

- [ ] **Step 3: Seal provenance**

Freeze resolved config, dirty/untracked source content, map/task arrays, library/checkpoint hashes, raw records, traces, summaries, and figures. Missing expected artifact prevents success finalization.

- [ ] **Step 4: Run tests**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/stc/test_exact.py tests/stc/test_exact_multicandidate.py tests/stc/test_stats.py tests/stc/test_stats_pairing.py tests/stc/test_run_context.py tests/stc/test_provenance.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add segment/stc/exact.py segment/stc/stats.py segment/stc/run_context.py tests/stc
git commit -m "fix: finalize exact statistics and provenance"
```

## Phase III - Baselines and Formal Experiments

### Task 8: Implement Complete-System Baseline Adapters

**Files:**
- Create: `segment/stc/system_baselines.py`
- Create: `tests/stc/test_system_baselines.py`
- Modify: `experiment/run_stc_main.py`

**Interfaces:**
- Produces six named methods evaluated by the same synchronous evaluator.

- [ ] **Step 1: Region-Distance-Straight**

Geographic partition, distance sequence, Straight-Hold segments.

- [ ] **Step 2: Shortest-Balanced-Graph**

Balanced spatial assignment, distance/3-opt sequence, graph/SLSQP segment.

- [ ] **Step 3: Jia-EBTAS-style**

Existing Jia-balanced assignment, existing Time-Energy/GA-EQTSP sequence, Jia/SCA-style or closest deterministic communication-aware segment optimizer. Preserve its internal criterion, but evaluate on common T/D/O metrics.

- [ ] **Step 4: GATSAC-style**

Spatial K-means assignment, communication-aware TSP/3-opt ordering, SAC local trajectories. Use full CKM; label as style reproduction unless exact author artifacts exist.

- [ ] **Step 5: Library-Open-Loop**

Library-aware EBTAS + sequencing + best independent candidates, with no synchronous conflict feedback.

- [ ] **Step 6: Proposed STC-ALNS**

Same initial solution/library as Library-Open-Loop, then synchronous conflict-feedback refinement.

- [ ] **Step 7: Test isolation and simultaneous start**

Assert all baselines start at `t=0`, use identical scenario/library/reference/failure rules, and no baseline silently calls proposed feedback operators.

- [ ] **Step 8: Commit**

```bash
git add segment/stc/system_baselines.py tests/stc/test_system_baselines.py experiment/run_stc_main.py
git commit -m "feat: add Jia and GATSAC system baselines"
```

### Task 9: Implement RQ1-RQ5 Experiment Runners

**Files:**
- Rewrite: `experiment/run_stc_segment_gate.py`
- Rewrite: `experiment/run_stc_evaluator_audit.py`
- Rewrite: `experiment/run_stc_main.py`
- Rewrite: `experiment/run_stc_ablation.py`
- Rewrite: `experiment/run_stc_scale.py`
- Create: `tests/stc/test_rq_pipelines.py`

**Interfaces:**
- Produces finalized raw records and summaries for RQ1-RQ5.

- [ ] **Step 1: RQ1 - pairwise segment quality**

Compare Straight-Hold, Comm-Waypoint, graph, SLSQP, TD3, DDPG, and SAC on identical directed edges. Report full-set feasibility, common-feasible T/D/O, sector diversity, and compute time.

- [ ] **Step 2: RQ2 - assignment and sequence value**

Compare region, spatial K-means, shortest-balanced, Jia-EBTAS, and library-aware EBTAS under fixed segment/evaluator settings; then compare distance, Jia/GA-EQTSP, and library-aware sequences.

- [ ] **Step 3: RQ3 - synchronous shared-bandwidth necessity**

Compare independent full bandwidth, legacy static congestion factor, and slot-synchronous evaluation on identical complete plans. Report metric error, feasibility reversal, and ranking reversal.

- [ ] **Step 4: RQ4 - full method comparison**

Run the six system baselines from Task 8. Primary contrasts: Proposed vs Jia-EBTAS-style; Proposed vs GATSAC-style; Proposed vs Library-Open-Loop.

- [ ] **Step 5: RQ5 - mechanisms and applicability**

Ablate EBTAS initialization, multi-candidate library, hotspot destroy, conflict-aware repair, path switch, adaptive weights, and incremental evaluation. Separately vary UAV count, task scale/density, payload, bandwidth, uplink power, and full-known map seed.

- [ ] **Step 6: Run deterministic pipeline tests**

Require real nonempty records, no success on missing inputs, no silent missing method/scenario pair, and progress/checkpoint output for long runs.

- [ ] **Step 7: Commit**

```bash
git add experiment/run_stc_segment_gate.py experiment/run_stc_evaluator_audit.py experiment/run_stc_main.py experiment/run_stc_ablation.py experiment/run_stc_scale.py tests/stc/test_rq_pipelines.py
git commit -m "feat: implement final RQ1-RQ5 pipelines"
```

### Task 10: Implement Plotting, Gates, and Formal Freeze

**Files:**
- Rewrite/commit: `experiment/run_stc_gates.py`
- Rewrite: `plots/plot_stc_results.py`
- Create: `tests/stc/test_gate_orchestrator.py`
- Modify: `tests/stc/test_plot_stc_results.py`
- Rewrite: `docs/stc-experiment-protocol.md`
- Rewrite: `仿真实验.md`
- Modify: `README.md`

**Interfaces:**
- Produces: machine-readable Gate 0-4 report, formal tables/figures, frozen run bundle.

- [ ] **Step 1: Implement Gate 0-4 orchestration**

Gate 0: evaluator/full tests. Gate 1: segment evidence. Gate 2: assignment/sequence and synchronous-evaluator evidence. Gate 3: proposed-method contrast plus at least two contributing ablations. Gate 4: provenance/reproduction. Missing evidence is `NOT_RUN`, never PASS.

- [ ] **Step 2: Implement formal plots**

Generate T-D Pareto front, success/feasibility, completion-time balance, outage, conflict lost bits, waiting, paired-effect CI, exact gap, runtime/scaling, and ablation figures.

- [ ] **Step 3: Update active documentation**

State full-known CKM, no energy, simultaneous start, source-data semantics, six baselines, exact RQs, seed hierarchy, failure policy, and allowed claims. Remove departure-shift and stale TD3/scaling claims.

- [ ] **Step 4: Run full suite and one bounded pilot**

Run all tests, then one small RQ1-RQ5 pilot with time limits. Require nonempty raw data, figures, summaries, gate report, and sealed manifest.

- [ ] **Step 5: Independent whole-branch review**

No Critical/Important findings; reproduce one raw plan/result/figure from frozen inputs before formal runs.

- [ ] **Step 6: Commit**

```bash
git add experiment/run_stc_gates.py plots/plot_stc_results.py tests/stc docs/stc-experiment-protocol.md 仿真实验.md README.md
git commit -m "feat: finalize STC experiment gates and reporting"
```

## 6. Experiment Matrix

### 6.1 Development and pilot

- unit/Gate 0 fixtures: synthetic 1-2 UAV cases;
- exact smoke: `U=2`, `K=2-4`;
- pilot: `U=2`, `K=6`, one map/task/method seed;
- no formal claim from pilot.

### 6.2 Formal fixed-task UAV-count experiment

Hold constant:

- area: 4 km x 4 km;
- BS layout: 4 BSs;
- tasks: same 20 points and payloads;
- CKM and radio parameters.

Vary only `U in {2,3,4}`. This is the only experiment interpreted as the effect of UAV count.

### 6.3 Formal task/area scale experiment

Use separate density-oriented families:

- `(U=2, K=20, area=4 km, BS=4)`;
- `(U=3, K=30, area=5 km, BS=5)`;
- `(U=4, K=40, area=6 km, BS=6)`.

Interpret as system-scale performance, not pure UAV-count scalability.

### 6.4 Scenario and method seeds

Minimum formal set:

- five paired scenario instances with distinct `map_seed` and `task_seed`;
- five `method_seed` replicates for stochastic assignment/search methods;
- learned baselines use at least five `train_seed` checkpoints or are explicitly labeled as limited single-checkpoint baselines.

Aggregate method seeds within a scenario before scenario-level inference.

### 6.5 Objective and sensitivity

- primary: `w_T=w_D=0.5`;
- Pareto: `w_T in {0.2,0.4,0.5,0.6,0.8}`, `w_D=1-w_T`;
- payload multiplier: `{0.5,1.0,1.5,2.0}`;
- bandwidth: `{0.5,1.0,2.0}` MHz;
- uplink power: `{0.1,0.2,0.5}` W;
- every physical-parameter sweep rebuilds the full segment library.

## 7. Statistics and Failure Policy

- Success rate uses all paired instances.
- Cost comparisons use the common-feasible paired set and report its size.
- Also report a predeclared penalized all-instance objective; never silently drop failures.
- Report mean, median, standard deviation, 95% confidence interval, paired difference CI, effect size, and paired permutation/Wilcoxon result.
- Apply Holm correction to prespecified multiple contrasts.
- Primary comparisons: Proposed vs Jia-style, Proposed vs GATSAC-style, Proposed vs Library-Open-Loop.

## 8. Required Formal Tables and Figures

1. System/model parameter table.
2. Segment-method feasibility and paired T/D/O table.
3. Assignment/sequence comparison table.
4. Independent/static/synchronous evaluator error table.
5. Complete-system baseline table.
6. Exact tiny-instance gap/runtime table.
7. Ablation table.
8. UAV-count and scale table.
9. Representative trajectories over A2G/G2A maps.
10. Time-aligned UAV route/sector/buffer/conflict timeline.
11. T-D Pareto front.
12. Paired-effect confidence plot.
13. Success/failure-reason plot.
14. Conflict-loss and waiting-time plot.
15. Runtime/scalability plot.

## 9. Stop Rules

- Stop before algorithm work if simultaneous-start Gate 0 fails.
- Stop before formal runners if segment library payload semantics or persistence fails.
- Stop before formal matrix if Jia/GATSAC baselines are not isolated and reproducible.
- Stop if exact solver claims optimality without a valid covered domain and bound.
- Stop if STC operators do not receive real edge/time conflict information.
- Stop if formal smoke creates success manifests without raw records.
- If Proposed does not improve objective but improves success rate, frame the paper around reliable feasibility.
- If Proposed improves only under high load, report that applicability boundary explicitly.

## 10. Execution and Review Policy

Execute Tasks 1-10 strictly in order using subagent-driven development. Each task gets one implementation subagent and one independent read-only reviewer before the next task starts. Require whole-branch reviews after Task 5, Task 7, and Task 10. The formal experiment matrix requires explicit user approval after the bounded pilot report.
