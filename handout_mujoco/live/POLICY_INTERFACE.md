# Policy interface — what is established, and what is UNKNOWN

**Status: PARTIAL — blocked on checkpoints.** Nothing below is guessed. Every row is
either sourced from an artifact in this repo (cited) or marked **UNKNOWN**.

## Summary table

| policy | checkpoint | obs dim | obs fields/order | normalization | action dim | action mapping | inference API |
|---|---|---|---|---|---|---|---|
| cheap MLP | **NOT FOUND** | UNKNOWN | UNKNOWN | UNKNOWN | **3** | **a∈[-1,1]³ → EE displacement a×0.05 m** | UNKNOWN |
| Reach TQC | **NOT FOUND** | UNKNOWN | UNKNOWN | UNKNOWN | **3** | same | UNKNOWN (SB3/sb3-contrib `TQC.predict`, unverified) |
| Push TQC | **NOT FOUND** | UNKNOWN | UNKNOWN | UNKNOWN | **3** | same | UNKNOWN (as above) |

## What IS established

| item | value | source |
|---|---|---|
| cheap parameter count | 5,891 | `results/efficiency_summary.json: params.cheap` |
| Reach actor parameter count | 5,382 | `params.expert_reach_actor` |
| Push actor parameter count | 541,190 | `params.expert_push_actor` |
| action dimension | 3 | `data/meta.json: arrays.actions`, shape (T,3) |
| action semantics | EE displacement command in [−1,1]³, ×0.05 m per step | `meta.json: arrays.actions` |
| control period | 0.04 s (25 Hz) | `meta.json: control_dt_s / control_hz` |
| inference determinism | deterministic actions | `results/REPORT_reach_push.md` §1 ("deterministic actions") |
| Push checkpoint lineage | fine-tuned from `enaitzb/TQC-PandaPush-v3`, run 2 @350k, 0.90 on 200 fresh seeds | `REPORT_reach_push.md` §1 |
| Reach success | 1.00 | `CLAIMS.md` |
| cheap policy origin | "a 5.9k-parameter MLP distilled from the experts" | `CLAIMS.md` |
| measured CPU inference | cheap 0.0845 ms, reach 0.3153 ms, push 0.4136 ms | `efficiency_summary.json` |
| phase definition | reach → push; phase A must succeed within 50 steps or the episode fails; phase B has a 50-step budget | `REPORT_reach_push.md` §2 |
| phase-A success | ‖ee − reach_goal‖ < 0.05 | `meta.json: goals.reach_goal` |
| phase-B success | ‖cube − task_goal‖ < 0.05 | `meta.json: goals.task_goal` |
| phase label array | `labels`: 0 = reach, 1 = push | `meta.json: arrays.labels` |

## What is UNKNOWN and cannot be inferred

**Observation vector.** Dimension, field order, and content are all unrecorded. The
recorded episodes store `qpos/obj_pos/obj_quat/ee/reach_goal/task_goal` — enough to
*rebuild* a plausible panda-gym observation, but panda-gym's `PandaPush-v3` returns a
Dict space (`observation` / `achieved_goal` / `desired_goal`) whose exact flattening,
whether velocities are included, and whether the composed wrapper appended phase
information are **not determinable from any artifact here**. Guessing wrong produces a
policy that runs and is silently not the benchmarked one.

**Normalization.** Whether `VecNormalize` was used, and if so its running mean/var, is
unknown. SB3 stores these in a separate `vecnormalize.pkl` — absent.

**Architecture beyond parameter count.** Layer widths, activations, and (for TQC) the
actor head structure are unknown. 5,891 parameters constrains but does not determine
the shape.

**Stochastic vs deterministic at collection.** The report says deterministic for
*evaluation*; whether the recorded episodes used `deterministic=True` is not stated
per-condition.

**Recurrence.** No evidence either way; the parameter counts are consistent with
feed-forward MLPs, but that is an inference, not a fact.

## Consequence

`live/mj_env.py: observation()` deliberately raises `NotImplementedError`. It will be
implemented by reading the layout off the recovered checkpoint and its env wrapper —
not before. See `results/path_b_checkpoint_audit.md` for the recovery list.
