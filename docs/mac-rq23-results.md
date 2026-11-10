# Mac RQ2/RQ3 First-Pass Results

## Protocol

- Scenario: U=2/s42.
- Common full-route executor: MapGuide + one-step safety projection.
- RQ2 fixes GA-EQTSP and varies assignment.
- RQ3 fixes 4D assignment and varies sequence.
- One deterministic run per method.
- Reference values: `T_ref=436.5 s`, `D_ref=507.0 s`.

## RQ2 Assignment

| Assignment | Feasible | T (s) | D (s) | J |
|---|---:|---:|---:|---:|
| Balanced spatial | yes | 316.0 | 275.5 | 0.6337 |
| Spatial K-means | yes | 384.0 | 305.5 | 0.7411 |
| 4D communication-aware | yes | 372.5 | 322.0 | 0.7442 |
| Region | yes | 513.0 | 506.0 | 1.0866 |
| Jia-style | no | 352.5 | 360.0 | inf |

The first-pass result does **not** support a claim that 4D assignment is best.
Balanced spatial assignment is best under the current MapGuide executor.

## RQ3 Sequencing

| Sequence | Feasible | T (s) | D (s) | J |
|---|---:|---:|---:|---:|
| Jia Time-Energy | yes | 361.5 | 311.5 | 0.7213 |
| GA-EQTSP | yes | 372.5 | 322.0 | 0.7442 |
| Euclidean GA | no | 363.0 | 432.0 | inf |
| NN + 2-opt | no | 390.5 | 486.5 | inf |

The first-pass result does **not** support a claim that GA-EQTSP is best.
Jia Time-Energy is best under the current executor and one-scene protocol.

## Interpretation

These runs isolate assignment and sequencing because the same MapGuide
executor and physical constraints are used. They are not interchangeable with
the final MCO-A*-based component experiments.

Before writing RQ2/RQ3 conclusions, the team must choose one of:

1. improve the 4D and GA-EQTSP methods;
2. re-evaluate them under the final MCO-A* executor;
3. change the contribution claim to match the measured result.

The first-pass results must not be hidden or replaced by old STC-ALNS data.
