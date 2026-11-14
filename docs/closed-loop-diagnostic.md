# Closed-loop diagnostic — why the tower policies fail

**Date:** 2026-09-12 · every number measured in this repo, in closed loop, 20 episodes
per policy, seed 0, identical environment and procedure.

No teacher forcing anywhere. `scripts/tower_diagnose.py` drives
`observation(env) -> policy.act() -> env.step()`; demonstration data is never read
during evaluation, and expert actions are never used at runtime.

## The result

| Policy | Input dim | Success | Mean height | Grasp success | Median dist at close | <1cm | <3cm | <5cm | Closes/ep | First failure |
|---|---|---|---|---|---|---|---|---|---|---|
| BC-RNN/dart | 32 | 0% | 1.00 | 0% | 24.7 cm | 0% | 0% | 0% | 1.0 | B×20 |
| BC-RNN/rel | 41 | 0% | 1.00 | 4% | 17.0 cm | 0% | 4% | 4% | 1.3 | B×18, H×1, C×1 |
| ACT/dart | 32 | 0% | 1.00 | 8% | 12.5 cm | 0% | 2% | 10% | 14.0 | B×18, H×1, D×1 |
| ACT/rel | 41 | 0% | 1.00 | 5% | 15.8 cm | 0% | 2% | 6% | 12.2 | B×15, D×4, H×1 |
| **expert** (n=50) | — | **90%** | **2.80** | **99%** | **1.3 cm** | 14% | **100%** | 100% | 2.0 | — |

Failure codes: **A** wrong cube · **B** wrong spatial targeting · **C** never closed ·
**D** premature/late closure · **E** failed grasp despite alignment · **F** failed lift ·
**G** failed carry · **H** failed placement.

**B dominates: 71 of 80 episodes.** Not gripper timing, not grasp mechanics, not lift or
carry — the arm never arrives at the cube. BC-RNN/dart never once reached within 5 cm
across 20 episodes.

## Did the 41D eef-relative cube vectors fix targeting? No.

BC-RNN improved (24.7 → 17.0 cm) but **ACT got worse** (12.5 → 15.8 cm). A representation
that fixed targeting would help both. Neither crosses 3 cm; all four stay at 0% success
and mean height 1.00. The representation is not the bottleneck, so retraining on it is
not the next step.

## Why: the supervision is one-to-many

`TowerExpert` carries hidden state the observation does not — `idx` (which cube is next),
`stage`, and `timer`. Measured over 81,190 demo states, 3,000 queries,
12 nearest neighbours drawn only from *other* episodes:

| | |
|---|---|
| action spread among nearest neighbours | 0.3711 |
| action spread across the whole dataset | 1.0000 |
| **ratio** | **0.371** |
| neighbours wanting a *different* target cube | **44.0%** |
| queries where >20% of neighbours disagree | 68.4% |
| median separation of the two candidate cubes | 0.165 m |

**37% of the dataset's total action variation survives among states the
network cannot tell apart.** Behaviour cloning is fitting a one-to-many function, and a
least-squares regressor answers it with the mean of the branches.

That prediction is quantitative: two candidates 0.165 m apart,
averaged, put the gripper **~8 cm from either**.
Measured close distances are 12.5–24.7 cm. Prediction and measurement agree.

It also explains the behavioural split: BC-RNN closes **once** per episode at 24.7 cm and
gives up; ACT closes **14 times** at 12.5 cm, repeatedly grabbing at the gap.

## Is it an inference bug? No — the modes collapsed in training.

BC-RNN has a 5-mode GMM head, and `tower_policies.py` decodes it with `gmm_mean()`. If the
GMM had learned "cube A" and "cube B" as branches, the mean would land between them and the
fix would be one line. Same weights, three decoding rules, 20 closed-loop episodes each:

| Decode | Success | Median dist | <3cm | Grasp | First failure |
|---|---|---|---|---|---|
| `mean` | 0% | 17.0 cm | 4% | 4% | B×18, H×1, C×1 |
| `argmax` | 0% | 12.4 cm | 5% | 10% | B×19, H×1 |
| `sample` | 0% | 19.3 cm | 0% | 6% | B×20 |

| | |
|---|---|
| GMM mode spread (median) | 0.1259 |
| **top mixture weight (median)** | **0.967** |

The gating puts 96.7% of its mass on one component,
so `gmm_mean()` is nearly a no-op, and the components sit 0.126
apart against a global action scale of 1.0 — far too close together to encode two cubes
16.5 cm apart. `argmax` looks better (12.4 cm) but a 3.3% weight difference cannot cause a
27% targeting change; that is closed-loop divergence, and the outcome is unchanged at 0%,
B×19.

**The multimodality was destroyed by the objective, not discarded by the decoder.** ACT is
in the same position for a simpler reason: no mixture head at all, L1 loss against a
one-to-many mapping, so its 12.5 cm *is* the conditional mean. No decoding rule recovers a
branch that was never represented.

## Recommendation — one change

**Make the target cube explicit in the observation.** Append the eef→intended-cube vector,
where "intended" comes from the expert's own selection rule applied to cube geometry
(`scripts/tower_diagnose.py:intended_cube`): stack onto the cube nearest the base, take the
remaining cubes in order, skip any already above the stack threshold.

That rule is a pure function of the observation, so it needs no expert at runtime and is
not an oracle. It converts the one-to-many mapping into a function, which is the single
measured cause of the 71/80 code-B failures.

Not recommended, with reasons: retraining on 41D (measured — not the bottleneck); more
epochs (teacher-forced error is already 0.033 rad — not underfitting); DAgger (addresses
distribution shift, but the policies fail on the *first* reach from a nominal start, which
is function ambiguity, not drift); arbitration (two 0% arms arbitrate to 0%).

## Reproduce

```bash
.venv-lerobot/bin/python -m scripts.tower_baseline  --n 50                 # expert baseline
.venv-lerobot/bin/python -m scripts.tower_diagnose  --n 20 --variants dart rel
.venv-lerobot/bin/python -m scripts.tower_ambiguity --queries 3000         # supervision ambiguity
.venv-lerobot/bin/python -m scripts.tower_modes     --n 20                 # GMM decode test
```

Artifacts: `data/tower/baseline_expert.json`, `closed_loop_diagnostic.json`,
`closed_loop_traces.npz` (per-timestep tape), `ambiguity.json`, `mode_collapse.json`.
No checkpoints were written or overwritten; nothing was retrained.
