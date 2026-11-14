# Global-threshold ablation — result

**Date:** 2026-09-08 · **Data:** 450 rollouts, tasks 0/1/8, no new collection
**Script:** `scripts/ablation_global.py` · **Raw:** `data/results/ablation_global.json`

## Question

`docs/technical-report.md:441` states the claim: split CP on a trajectory score is
standard; calibrating a **separate threshold per stage** is the novel part. Does
that separation buy detection power over one well-calibrated global threshold?

## Verdict: yes, at the operating points that matter

At matched false-alarm rate, the stage-aware monitor (K=4, Bonferroni α/4)
detects more failures than a single global threshold (K=1, α) on **all three
tasks at every FAR ≥ 0.10**.

| achieved FAR | task 0 | task 1 | task 8 | mean ΔTPR |
|---|---|---|---|---|
| 0.05 | +0.073 | +0.022 | −0.021 | +0.025 |
| 0.10 | +0.132 | +0.034 | +0.012 | +0.059 |
| 0.15 | +0.086 | +0.060 | +0.067 | +0.071 |
| 0.20 | +0.075 | +0.128 | +0.138 | +0.114 |
| 0.25 | +0.070 | +0.230 | +0.153 | +0.151 |
| 0.30 | +0.077 | +0.215 | +0.125 | +0.139 |

Positive = stage-aware detects more failures **at the same false-alarm rate**.
20 seeds per point, paired within seed. At FAR 0.25, 93% of the 60 paired draws
favour stage-aware; at FAR 0.20, 87%.

At the project's stated operating point (α=0.10, achieved FAR ≈ 0.15) the
advantage is **+0.071 TPR**, positive on all three tasks.

## Why the naive comparison says the opposite

Comparing the two arms at the same *nominal* α reverses the conclusion — it makes
stage-aware look **worse** by up to −0.27 TPR. That comparison is invalid.
Bonferroni makes the stage-aware arm conservative, so at a given α the two arms
sit at different false-alarm rates:

| α | stage FAR / TPR | global FAR / TPR | naive ΔTPR |
|---|---|---|---|
| 0.05 | 0.000 / 0.000 | 0.123 / 0.223 | −0.223 |
| 0.10 | 0.152 / 0.449 | 0.207 / 0.533 | −0.084 |

Reading a detection rate at 0.207 FAR against one at 0.152 FAR compares two
different points on an ROC curve. Once both arms are evaluated at equal achieved
FAR, the ordering flips and stays flipped.

**This is the finding most at risk of being reported backwards.** Any writeup
must compare at matched FAR.

## Two honest limits

**The unit of replication is the task, not the seed.** The 20 seeds per task
resample the same rollouts, so they measure split variance, not independent
evidence. The honest sample size is n=3 tasks. All three point the same way at
FAR ≥ 0.10, but three is three.

**Stage-aware cannot operate below FAR ≈ 0.10 with current data.** Bonferroni
puts each phase at α/4; at α=0.05 that is α_k=0.0125, which needs ≥79 calibration
rollouts *per phase* against the ~51 available. The threshold goes to +inf and
the monitor never alarms — visible as the 0.000/0.000 rows. The global arm at
α=0.05 needs only ≥19 and stays feasible. If a low-false-alarm regime matters,
that is a real limitation of the stage-aware arm, and more calibration data is
the only fix.

## What this means for collection

The thesis survives its first real test, so the 4th task is worth its cost — but
as a **precision** improvement, exactly as `docs/status.md` §7 already framed it,
not as a prerequisite. The effect at FAR 0.15-0.30 is +0.07 to +0.15 TPR with
per-seed sd of ~0.10; more failures would tighten that interval and would also
raise the per-phase calibration count that currently blocks the low-FAR regime.
