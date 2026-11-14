# MuJoCo Panda controller tuning — what was selected and why

**Date:** 2026-09-13 · branch `mujoco-arbitration` · no policies loaded, none retrained.
Nothing in `mujoco/`, `data/` or the PyBullet benchmark was modified.

## The problem

A 0.05 m end-effector command completed only **0.208** of its displacement inside the
0.04 s control period, with actuators 2, 4 and 6 pinned at their force limits on the
first substep. Scaling the action did nothing — the servo was already at its torque
ceiling, so commanding harder only saturated harder.

## The acceptance target is 0.655, not 1.0

Measured on the recorded episodes, **PyBullet itself delivers a median
achieved/commanded ratio of 0.655** (expert) / 0.597 (cheap). Its controller does not
complete the full displacement either. Tuning MuJoCo to 1.0 would make it track
*better* than the system the policies were trained in — a different mismatch, not a
fix. The target is to match the reference.

## Sweep

60 configurations over MuJoCo's affine-bias position servo: stiffness `kp` ×{1,2,5,10,20},
damping ratio `kv/kp` ∈{0.10,0.05,0.02}, force limits ×{1,3,10,30}. Metrics per
candidate: ratio at 20 substeps, final ratio, time to 90%, peak force fraction,
saturated-actuator count, overshoot, residual oscillation.

**The fix is damping, not stiffness.** The best configurations all use `kp × 1`.
panda.xml's default `kv/kp = 0.10` over-damps the servo: time-to-90% is **109 substeps
(0.22 s) against a 20-substep control period**, so most of the command is still
unexecuted when the next one arrives.

A second finding changed the selection: the probe sweep used unit-norm direction
vectors, but real actions have median norm 1.335. Selecting on the probe gave
`kv/kp = 0.02`, which overshoots to 0.855 under real sequences. Re-selecting against
**recorded action sequences** gave the final choice.

## Selected: `kp × 1`, `kv/kp = 0.035`, `forcerange × 10`

| | baseline | selected | PyBullet |
|---|---|---|---|
| achieved/commanded | 0.300 | **0.645** | 0.655 |
| EE error @ step 2 | 0.0836 m | **0.0341 m** | — |
| EE error @ step 10 | 0.1044 m | **0.0641 m** | — |
| EE error @ end | 0.1207 m | **0.0906 m** | — |
| cube error @ end | 0.2503 m | 0.2386 m | — |
| saturated actuators | 3 | **0** | — |
| time to 90% | 109 substeps | 26.5 | — |

Tracking now matches the reference to within 0.010. EE error improved 25–59%.

## What this does NOT fix

**Cube error barely moved (0.2503 → 0.2386 m) and success is 0/8 either way**, against
6/8 for the same seeds in PyBullet. The controller was a real defect and is fixed, but
it was not the dominant source of trajectory divergence — **contact physics is**. Once
the gripper touches the cube, the two engines' contact models part company and the cube
ends ~0.24 m from where PyBullet left it, 4.8× the 0.05 m success radius.

**This is not evidence that policies will fail in MuJoCo.** Open-loop action replay is
the worst case: it has no feedback, so any divergence compounds without bound. A
closed-loop policy observes the actual state and corrects. The transfer question can
only be settled by running the policies closed-loop, which requires the checkpoints
that are not in this repo.

## Caveat on physical realism

`forcerange × 10` puts joint torques well above a real Panda's 87 Nm. That is
deliberate and matches the reference: panda-gym drives PyBullet's `POSITION_CONTROL`,
which is effectively a kinematic constraint with very high force authority. Matching
PyBullet's *response* therefore requires non-physical limits in MuJoCo. If the goal
were sim-to-real rather than sim-to-sim, this is the wrong trade.

## Reproduce

```bash
python live/build_live_scene.py      # scene with a collidable cube and a valid keyframe
python live/smoke_test.py            # 4/4: the loop is real and physics-driven
python live/tune_controller.py       # the 60-config sweep -> controller_tuning.{json,png}
python live/transfer_gap_tuned.py    # expert-action replay under each controller
```

Artifacts: `results/controller_tuning.json`, `results/controller_tuning.png`,
`results/transfer_gap_tuned.json`.
