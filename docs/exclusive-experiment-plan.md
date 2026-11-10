# Exclusive-Sector Hierarchical Multi-UAV Experiment Plan

## 1. Scope

The final experiment pipeline is:

```text
4D communication-aware task assignment
-> GA-EQTSP inspection sequencing
-> QoS-constrained full-trajectory planning
-> one common exclusive-sector physical environment
```

All deterministic and learned methods use the same:

- fixed U=2/3/4 s42 task data;
- 10 m task and terminal acceptance radius;
- one UAV per sector per slot;
- upload-before-next-service and final-upload constraints;
- per-UAV maximum contiguous G2A outage of 20 s;
- ten-second-or-better trajectory representation and full mission horizon;
- feasibility-first metrics.

No new task seeds, map seeds, method seeds, s43/s44 scenes, or exact K
enumerations are part of the new formal matrix.

## 2. Paper Claims and Required Evidence

The final wording must follow measured results rather than pre-assuming that
SharedTD3-BC wins every metric.

| Claim | Required evidence |
|---|---|
| 4D communication-aware assignment is useful | RQ2 with GA-EQTSP + MCO-A* fixed |
| GA-EQTSP improves visit ordering | RQ3 with 4D + MCO-A* fixed |
| Full-trajectory planning beats segment concatenation | RQ1: Straight, MCO-A*, segmented TD3, full SharedTD3-BC |
| Learned full-trajectory methods can satisfy QoS | SharedTD3-BC, MAPPO, and MA-D3QN deterministic feasibility |
| The complete hierarchy is strongest overall | RQ4 system comparison |
| Results persist at larger fleet sizes | U=3/U=4 s42 single-seed probes |

If the final objective or individual metrics do not place SharedTD3-BC first,
the paper must report the actual tradeoff. For example, MA-D3QN may have lower
makespan while SharedTD3-BC has lower upload delay.

## 3. Experiment Matrix

### RQ1: Full-trajectory planning

Fixed:

- U=2/s42;
- 4D assignment;
- GA-EQTSP sequence.

Compare:

1. Straight-Hold;
2. MCO-A*;
3. segmented TD3 concatenation;
4. full SharedTD3-BC;
5. MAPPO;
6. MA-D3QN-style.

The four system-level arms are Straight, MCO-A*, segmented TD3, and full
SharedTD3-BC. MAPPO and MA-D3QN are algorithm-level full-trajectory controls.

### RQ2: Assignment

Fixed:

- GA-EQTSP;
- MCO-A*.

Compare:

1. Region;
2. Spatial K-means;
3. Balanced Spatial K-means;
4. Jia-EBTAS;
5. 4D communication-aware clustering.

### RQ3: Sequencing

Fixed:

- 4D assignment;
- MCO-A*.

Compare:

1. NN + 2-opt;
2. Euclidean GA;
3. Jia Time-Energy;
4. GA-EQTSP.

### RQ4: System-level comparison

Main systems:

1. Region-Distance-Straight;
2. Jia-style;
3. GATSAC-style;
4. MCO-A*-based hierarchical;
5. MAPPO;
6. MA-D3QN-style;
7. Proposed SharedTD3-BC.

Supplementary negative results:

- original MADDPG;
- original MATD3;
- GG-MATD3-style limited-budget reproduction.

## 4. Formal Training Configuration

All learned formal methods use:

```text
episodes: 10,000 additional episodes
horizon: 2,000 steps
map-guide probability: 0.5 constant
safety outage threshold: 0.2
outage constraint: 20 s per UAV
evaluation: deterministic, teacher episodes disabled
checkpoints: best_constraint.pth and best_completion.pth
```

SharedTD3-BC:

```text
BC weight: 1.0 -> 0.25 over 600 episodes
demo fraction: 0.25
actor learning rate: 1e-5
critic learning rate: 1e-3
```

MAPPO:

```text
actor learning rate: 1e-4
critic learning rate: 1e-3
```

MA-D3QN:

```text
n-step: 10
action repeat: 2
DQfD margin: 0.8
demo fraction: 0.25
learning rate: 1e-4
pretraining: already completed in the accepted warm-start run
```

## 5. Shared Scenario and Routes

The U=2/s42 fixed route is:

```text
configs/exclusive/u2_s42_4d_ga_routes.json
```

U=3/U=4 routes are generated once on Mac and frozen before Windows
scale-probe training.

## 6. Accepted Short-Run Warm Starts

These short runs have passed the 20/20 task, 20/20 upload, upload-order,
collision, boundary, and per-UAV contiguous-outage checks.

| Method | Accepted run | Warm-start checkpoint |
|---|---|---|
| SharedTD3-BC | `short-v15-u2-s42-sharedtd3bc-safe020-e200-s2000` | `best_constraint.pth` |
| MAPPO | `short-v18-mappo-safe-e200-s2000` | `best_constraint.pth` |
| MA-D3QN | `diag-v17b-ma-d3qn-repaired-e200-s2000` | `best_constraint.pth` |

The short runs are acceptance artifacts and warm starts, not final paper
results.

## 7. Mac Responsibilities

Mac is responsible for:

- environment, route, and code freeze;
- U=3/U=4 fixed route generation;
- Straight-Hold and MCO-A* experiments;
- segmented TD3 concatenation experiment;
- RQ2 assignment experiments;
- RQ3 sequencing experiments;
- deterministic Jia/GATSAC/MCO-A* system arms;
- artifact hash audit and final aggregation;
- regenerating all manuscript figures from sealed outputs;
- final manuscript text and result consistency.

Mac must not retune the formal learned methods after Windows training starts.

## 8. Windows Responsibilities

Windows is responsible for all learned formal training:

- SharedTD3-BC;
- MAPPO;
- MA-D3QN;
- GATSAC-style training/replay where a learned checkpoint is required;
- GG-MATD3-style supplemental negative reproduction.

Windows must:

- use one training seed;
- use the exact frozen config and route file;
- avoid post-result hyperparameter changes;
- retain complete manifests;
- return `best_constraint.pth`, `best_completion.pth`, training curves, and
  deterministic rollout metrics;
- report infeasible results rather than rerunning until a favorable result is
  found.

## 9. Windows Commands

From PowerShell after syncing the repository:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest tests/test_td3_bc.py tests/test_ma_d3qn.py tests/test_exclusive_marl_pilot.py tests/stc/test_exclusive_e2e_env.py -q
```

SharedTD3-BC:

```powershell
.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm SharedTD3 --episodes 10000 --horizon-steps 2000 `
  --eval-interval 50 --guide-prob 0.5 --guide-min-prob 0.5 `
  --bc-weight 1.0 --bc-min-weight 0.25 --bc-decay-episodes 600 `
  --demo-fraction 0.25 --actor-lr 1e-5 --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --init-checkpoint results\runs_short_v15\short-v15-u2-s42-sharedtd3bc-safe020-e200-s2000\best_constraint.pth `
  --routes-json configs\exclusive\u2_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u2-s42-sharedtd3bc
```

MAPPO:

```powershell
.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm MAPPO --episodes 10000 --horizon-steps 2000 `
  --eval-interval 50 --guide-prob 0.5 --guide-min-prob 0.5 `
  --actor-lr 1e-4 --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --init-checkpoint results\runs_short_v18\short-v18-mappo-safe-e200-s2000\best_constraint.pth `
  --routes-json configs\exclusive\u2_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u2-s42-mappo
```

MA-D3QN:

```powershell
.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm MA-D3QN --episodes 10000 --horizon-steps 2000 `
  --eval-interval 50 --guide-prob 0.5 --guide-min-prob 0.5 `
  --demo-fraction 0.25 --demo-margin 0.8 --actor-lr 1e-4 `
  --safety-outage-threshold 0.2 --n-step 10 --action-repeat 2 `
  --pretrain-demo-episodes 0 --pretrain-updates 0 `
  --init-checkpoint results\runs_short_v17\diag-v17b-ma-d3qn-repaired-e200-s2000\best_constraint.pth `
  --routes-json configs\exclusive\u2_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u2-s42-ma-d3qn
```

## 10. Handoff and Audit

1. Mac defines the commit and frozen configs.
2. Windows checks out exactly that commit.
3. Windows runs only the predefined commands.
4. Windows commits/pushes sealed `results/formal_win/` runs.
5. Mac verifies manifests and hashes.
6. Mac generates the final tables and figures.
7. Only audited sealed runs enter the manuscript.
