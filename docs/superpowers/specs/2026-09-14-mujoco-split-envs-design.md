# Design — splitting the MuJoCo Reach→Push environment into two single-phase envs

**Date:** 2026-09-14 · branch `mujoco-arbitration` · Phase 1 of the arbitration build.
**Supersedes nothing.** `live/mujoco_reach_push_env.py` (the composed env) stays on disk
as the record of the 4%-success attempt; nothing new imports it except `quat_to_euler`.

## 1. Why split at all

A single TQC on the composed task reached 4% after 400k × 3 seeds. The measured cause
was not exploration noise: median closest approach to the cube 10.7 cm, cube moved in
3% of push phases. The agent banks the +10 reach bonus and hovers, because the dense
push reward has a hovering optimum. The reference system never trained one end-to-end
agent either — it trained two experts and composed them by phase switch.

So: two envs, two experts, composed by a state machine. Both wrap the **existing**
`MJLive`, so training and the live demo share one simulator, one tuned controller
(`kp×1, kv/kp=0.035, forcerange×10`) and one 25 Hz control period. There is exactly one
simulator in this project and Phase 1 does not add a second.

## 2. Reach goal semantics — Option A, decided

`reach_goal` is a **pre-push waypoint**, not an arbitrary point in space:

```
u          = unit(cube_pos − task_goal)          # points away from the goal
reach_goal = cube_pos + 0.08 · u                 # at cube height, z = 0.02
```

**Rationale.** Option B (an arbitrary waypoint sampled on the ±0.15 × [0.02, 0.30] box)
spends up to 50 steps travelling somewhere irrelevant and hands the push phase an
arbitrary arm pose. Option A makes reach *set up* the push, and it makes the two phases
genuinely different regimes — free-space servoing versus contact-rich manipulation.
That difference is the whole reason a phase-conditioned q̂ is meaningful rather than
decorative: if both phases were free-space motion, one threshold would do.

**Consequence, stated explicitly:** the Phase-1 brief also specifies
`reach_goal ~ U(±0.15, ±0.15, [0.02, 0.30])`. That is the Option-B reset and is
**superseded** — under Option A the waypoint is derived, and its z is pinned to cube
height. Reach becomes a table-plane approach task.

**Measured consequences** (`analysis/phase1_measurements.py`, N=400):

| | |
|---|---|
| waypoint outside the ±0.15 sample box | **62.3%** (max \|x\| 0.229, \|y\| 0.229) |
| kinematically reachable (DLS IK residual) | 100%, median 7e-5 m, max 1e-4 m |

The spec's earlier "reachability verified at all 8 corners" covered the ±0.15 box, which
no longer contains 58% of the waypoints. Reachability is re-verified above for the
waypoint set that Option A actually produces.

## 3. Reset distribution

Shared by both envs, so the two experts see the same task geometry:

```
cube      : x, y ~ U(−0.15, 0.15),  z = 0.02
task_goal : x, y ~ U(−0.15, 0.15),  z = 0.02
resample while ‖cube − task_goal‖ < 0.05
```

**The minimum separation is a decision, not a detail.** 7.7% of 20,000 raw uniform pairs
land inside the 0.05 success radius, i.e. the episode is won at t=0 before any action.
Those are not task instances; leaving them in would put an ~8% floor under every success
number this project reports, at every gate, and the floor would be invisible in the
JSON. Rejecting also conditions the waypoint: ‖cube − task_goal‖ ≥ 0.05 keeps
`unit(cube − task_goal)` stable, where the raw minimum was 0.0015 m.

This departs from the reference PyBullet task (which had no rejection). The two systems
are reported side by side and never pooled, so the departure costs nothing and is
recorded here.

## 4. `ReachOnly`

| | |
|---|---|
| obs | flat `Box(−inf, inf, (24,))` — the 26-dim layout minus the phase one-hot |
| action | `Box(−1, 1, (3,))`, EE displacement `a × 0.05 m` per 0.04 s step |
| reset | arm at `NEUTRAL`, cube/goal per §3, `reach_goal` per §2 |
| reward | `−‖ee − reach_goal‖ − 0.01‖a‖²` |
| terminate | `‖ee − reach_goal‖ < 0.05` → success · 50 steps → failure |

The cube fields stay in the observation even though reach does not touch the cube: the
Phase-5 cheap MLP runs one flat 26-dim observation across both phases, so the two
single-phase layouts must be slices of that one vector, not bespoke per-phase vectors.

**Risk check** (`analysis/phase1_measurements.py`, N=150): a plain P-servo solves this
100% in a median of 4 steps (p95 6, max 7) against a 50-step budget. Gate 3 (≥0.95) is
not where this project is at risk.

## 5. `PushOnly`

| | |
|---|---|
| obs | `Dict{observation: Box(18), achieved_goal: Box(3), desired_goal: Box(3)}` |
| `observation` | ee_pos 3, ee_vel 3, cube_pos 3, cube_rot 3, cube_lin_vel 3, cube_ang_vel 3 |
| `achieved_goal` | `cube_pos` · `desired_goal`: `task_goal` |
| reward | sparse: `0.0` when `‖ag − dg‖ < 0.05`, else `−1.0` |
| `compute_reward` | vectorized over a leading batch axis, as HER requires |
| terminate | success · 50 steps · `cube_z < −0.05` |

### 5.1 The reset is the hard part

The push expert must start where reach leaves it, or Phase 4 composes two policies that
never met. The arm is therefore placed at `cube + offset` by **kinematic** IK — DLS on
the `tcp` site with `mj_forward` only, no physics, no cube disturbance. Measured
residual: median 7e-5 m, max 1e-4 m over 400 targets.

**The offset distribution is fitted in the approach frame, not per-axis.** Decomposing
the hand-off pose onto `u` (approach ray), `n = u⊥` and `z` (N=150):

| component | mean | sd | p5 | p95 |
|---|---|---|---|---|
| along `u` | +0.0557 | 0.0234 | +0.0158 | +0.0880 |
| lateral `n` | −0.0000 | 0.0141 | −0.0215 | +0.0248 |
| vertical `z` | +0.0132 | 0.0127 | −0.0119 | +0.0311 |

The offset lives on a **shell along the approach ray** — the `along` component is four
times the lateral spread — so a factorized per-axis fit in world xyz reproduces every
marginal while destroying the structure. The frame is justified by fidelity to the
geometry Phase 4 will actually present, **not** by physical validity: see §5.2, where
the measurement shows the fit barely moves penetration at all.

### 5.2 Reset validity is enforced by the contact solver, not by the fit

A kinematic placement can bury the closed gripper in the cube, and the episode then
opens by ejecting it. Measured over 200 resets each:

| sampler | overlap < −1 mm | max cube jump | jump > 5 mm | attempts |
|---|---|---|---|---|
| factorized xyz, no rejection | 30.5% | 18.3 mm | 15.5% | 1.00 |
| approach frame, no rejection | 28.5% | 25.4 mm | 14.0% | 1.00 |
| **approach frame + rejection** | **3.0%** | **1.98 mm** | **0%** | 1.32 (max 5) |

**The first two rows are the point.** Choosing the right frame moves penetration from
30.5% to 28.5% — nothing. What makes the reset valid is resampling while any cube↔robot
contact is deeper than −2 mm. The closed Panda gripper is not convex and its clearance
around a 4 cm cube is not a number worth deriving; the contact solver already knows it.

−2 mm is not arbitrary: poses the P-servo reaches *through physics* — valid by
construction — carry cube↔gripper overlap up to 6 mm, and finger-pad self-contact of
1.8 mm is present in every pose because the gripper is blocked closed.

### 5.3 This offset distribution is PROVISIONAL

It is fitted to a **P-servo**, not to `ReachExpert`, which does not exist until Phase 3.
The brief is explicit that a hardcoded guess creates a train/deploy mismatch at
composition, so the sampler's parameters are module-level constants with one re-fit
entry point, and Phase 4's first diagnostic on a composed-success miss is to compare the
offset `ReachExpert` actually delivers against these numbers. A P-servo fit measured in
the real simulator is a defensible prior; it is not a measurement of the trained policy.

**Known, deliberately deferred:** the P-servo already displaces the cube > 1 cm in
**24%** of reach episodes (max 6.5 cm) — the waypoint is only 8 cm behind a 4 cm cube.
`PushOnly`'s reset does not model that: it places the cube exactly where it was sampled.
Closing this now would mean guessing a disturbance distribution from a policy that does
not exist. Phase 4 measures it against the real `ReachExpert` and it is fixed there.

## 6. Files

| File | Change |
|---|---|
| `live/envs.py` | **new** — `ReachOnly`, `PushOnly`, and the four shared reset helpers |
| `live/test_envs.py` | **new** — both envs, incl. `check_env` and `compute_reward` batching |
| `live/MUJOCO_TRAINING_SPEC.md` | updated: §§1, 3, 4, 5, 7, 9, 10 rewritten; §§11–12 added |
| `analysis/phase1_measurements.py` | **new** — regenerates every number quoted above |
| `results/mujoco_train/phase1_measurements.json` | **new** — its output |

One module, not a package: the two envs share every reset helper and are ~200 lines
together. `MJLive` is reused, not wrapped twice.

## 7. Gate 1

Both envs pass `gymnasium.utils.env_checker.check_env`, and the smoke checks cover:
reset/step shapes and dtypes · bit-identical seeding · action causality · each
termination branch at its exact budget step · `Dict` space validity ·
`compute_reward` agreeing with `step`'s reward and vectorizing over a batch ·
push-reset validity (no born-successful episode, no cube ejection).
