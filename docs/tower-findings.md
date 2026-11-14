# Closed-loop Tower Creation — what was built, and what it showed

**Date:** 2026-09-11 · every number measured in this repo

## The three layers, all compatible and verified

```
  policy  ──►  7 absolute joint targets + grasp
              ──►  rate-limited delta  (action = clip((target−q)/RATE, −1, 1))
                  ──►  robosuite JointPositionController, Panda
                      ──►  MuJoCo, 20 Hz control / 500 Hz physics
                          ──►  3 cubes, contacts, stacking success predicate
                              ──►  new observation  ↺
```

| layer | file | status |
|---|---|---|
| TowerCreation env, 3 cubes, success predicate | `scripts/tower_env.py` | works |
| joint-target → controller bridge + DLS IK | `scripts/tower_control.py` | works |
| scripted expert | `scripts/tower_control.py` | **8/8 towers** |
| demo collection with DART | `scripts/tower_collect.py` | 344/400 successful |
| BC-RNN + ACT training on Route-B obs | `scripts/tower_train.py` | both converge |
| closed-loop adapters + evaluator | `scripts/tower_policies.py` | works |
| viewer | `scripts/tower_run.py` | `./run.sh tower` |

**The closed loop is real and verified.** `./run.sh tower` → height 3/3, success True, 221 steps.

## Two controller facts that cost time

**`control_delta` is never consulted.** robosuite's `JointPositionController.set_goal`
always does `goal = current + scale_action(action)`. Absolute joint targets must be
converted to rate-limited deltas. `RATE = 0.2 rad/step` at 20 Hz = 4 rad/s.

**Sawyer's joint 0 does not actuate in robosuite 1.4.1.** 79 Nm applied directly to the
actuator for 2 simulated seconds moves it 0.0007 rad; joints 1–6 track a 0.4 rad step to
within 0.03. Panda tracks all seven to 0.000. Hence Panda is the default.

## The learned policies do not complete the task

| variant | closed-loop success |
|---|---|
| Route-B obs (absolute cube positions) | 0/15 |
| + ACT chunk execution, exec_len 1/4/8/16 | 0/8 each |
| + DART noise-injected demos | 0/12 |
| + eef-relative cube vectors | 0/12 |

**It is not a pipeline bug.** Replaying the *recorded* expert actions open-loop
reproduces a 3-cube tower (`success True`).

**It is not underfitting.** Teacher-forced error is 0.016 rad (BC-RNN) / 0.033 rad (ACT),
and predicted gripper timing matches the demo exactly (0.49 vs 0.49 closed).

### Diagnosis 1 — causal confusion

Hold the robot state fixed, move the cubes ±5 cm, and see how much the predicted joint
target changes; compare against moving the arm ±5 cm.

| | cube ±5 cm | arm ±5 cm | ratio |
|---|---|---|---|
| BC-RNN, plain demos | 0.0053 rad | 0.0434 rad | **0.12×** |
| ACT, plain demos | 0.0085 rad | 0.0977 rad | **0.09×** |
| ACT, DART demos | 0.0187 rad | 0.0360 rad | **0.52×** |

The scripted expert is so smooth that the next joint target is nearly determined by the
current arm pose, so the cube channels are redundant and the network learns to ignore
them. DART made ACT **5.7× more object-sensitive** — the fix works on the metric it
targets — but did not produce towers.

### Diagnosis 2 — grasp precision

Distance from the gripper to the nearest cube at the moment it closes:

| | grasp attempts | median distance |
|---|---|---|
| expert | 2 | **0.011 m** |
| ACT (DART) | 7 | **0.084 m** |

A grasp needs roughly < 0.03 m. The policy learned the *rhythm* of pick-and-place —
descend, close, lift — but closes on a schedule rather than on arrival, at arm's length
from anything. That is why no cube ever leaves the table.

## What would plausibly fix it (not attempted)

- **DAgger.** The principled remedy for compounding error: run the policy, have the
  expert label the states it actually reaches, retrain. DART perturbs blindly; DAgger
  corrects where the policy actually goes wrong.
- **More data.** 344 demos is small for a three-stage task.
- **Object-centric architecture.** Per-cube tokens with attention, rather than nine
  concatenated coordinates.
- **Explicit task phase.** The expert has hidden state (`which cube next`) that the
  observation does not carry; the policy must infer it from cube heights.
- **Start with 2 cubes** to separate "task too long" from "not enough capacity/data".

## Honest status

Infrastructure: **complete and working.** Learned policies: **do not perform the task.**
That is a real result with a clean diagnosis, not a blocked build — every component is
verified independently, and a policy that can target precisely will drop into the same
loop unchanged.
