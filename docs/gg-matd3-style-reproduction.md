# GG-MATD3-style End-to-End Baseline

Reference:

> J. Pan, Y. Li, R. Chai, S. Xia, and L. Zuo, "Distributed Multi-UAV
> 3D Trajectory Planning Based on Multi-Agent Deep Reinforcement
> Learning," IEEE Transactions on Cognitive Communications and
> Networking, vol. 12, 2026.

## Reproduction Level

This is a **style reproduction**, not an author-code reproduction. The
reference paper does not provide code or a complete hyperparameter dump for
all baseline components.

Implemented here:

- centralised-training/decentralised-execution MATD3;
- twin critics, target-policy smoothing, and delayed actor updates;
- continuous per-UAV velocity and heading actions;
- dense per-step reward with energy, distance, outage, upload, and
  constraint penalties;
- distance-aware speed taper corresponding to the reference Greedy stage;
- local repulsion correction for close neighboring UAVs.

Not transferable or intentionally adapted:

- The UAV count is fixed by each STC scenario, so the reference GA UAV-count
  stage is not applicable.
- The reference one-UAV-per-device access rule is not used. Training and
  evaluation retain simultaneous shared-sector upload under the STC model.
- The reference AoI is replaced by the STC payload-generation,
  upload-before-next-service, outage, delay, and makespan semantics.
- The fixed-altitude STC mission uses `[v, theta]`; the reference 3D pitch
  action is unnecessary.
- The first pilot uses 2,000 episodes rather than the reference maximum of
  50,000 episodes. All results must therefore be labelled
  **limited-budget GG-MATD3-style**.

## Environment Adaptation

Implemented in `segment/stc/e2e.py`:

- each UAV observes its position, last action, buffer, remaining time,
  local radio quality, all task relative positions and visit masks, and
  neighboring UAV positions;
- a visited task generates its payload immediately;
- a UAV cannot start another task until its current payload is uploaded;
- concurrent uploaders in the same sector split the configured bandwidth;
- feasibility requires all tasks visited, all payloads uploaded, route-level
  outage within `eps_out`, and zero UAV-UAV collision events.

## Pilot

- Scenario: U=2, s42, K=20
- Train/evaluation seeds: 42
- Horizon: 1,000 steps
- Episode budget: 2,000
- Network update: every four environment steps after replay warm-up
- Device: MPS when available
- Run ID: `formal-gg-matd3-u2-s42`

The run is only promoted to a main-text baseline if the deterministic
checkpoint satisfies all hard constraints. A failed or non-completing pilot
is reported as a bounded negative result, not as a general claim that MADRL
cannot solve multi-UAV planning.
