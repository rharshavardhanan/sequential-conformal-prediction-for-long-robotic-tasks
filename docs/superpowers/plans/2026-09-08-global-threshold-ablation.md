# Global-threshold ablation — implementation plan

**The thesis test.** `docs/technical-report.md:441` states the claim: split CP on
a trajectory score is standard; calibrating a *separate threshold per stage* is
the novel part. This measures whether that separation buys anything.

## The comparison

Two arms, both proper split conformal with a per-rollout guarantee, both on the
same calibration/held-out split:

| arm | thresholds | alpha per test |
|---|---|---|
| **stage-aware** (K=4) | one per reach/grasp/transport/release | `alpha/4` (Bonferroni) |
| **global** (K=1) | one, over all steps | `alpha` |

## The methodological trap this must avoid

Bonferroni makes the stage-aware arm conservative: its *achieved* false-alarm
rate lands below `alpha`, while the global arm sits near it. Comparing detection
rates at unequal false-alarm rates is not a comparison — it reads off two
different points of an ROC curve and calls the higher one better.

**So the comparison is detection vs *achieved* FAR, not detection at fixed
nominal alpha.** Sweep alpha over a grid for both arms, plot detection against
the FAR each arm actually achieves on held-out nominal rollouts, and ask which
curve dominates. If the stage-aware curve sits above the global curve across the
usable FAR range, the claim holds. If they lie on top of each other, the
segmentation is decoration.

Two further controls:

- **Paired splits.** Both arms see the identical calibration/held-out partition
  at each seed, so the difference is paired and split noise cancels.
- **Many seeds.** With 103 successes and `cal_frac=0.5`, one split is ~51 cal /
  52 held-out. A single draw is far too noisy to conclude from. 20 seeds.

---

## Task 1 — Write the ablation driver

- **Goal:** one command produces the detection-vs-achieved-FAR curve for both
  arms, on all three tasks, averaged over seeds.
- **Why:** this is the experiment. Without it the project has a monitor and no
  evidence that its central design choice matters.
- **Files:** `scripts/ablation_global.py` (new)
- **Steps:** import `load`, `phase_agg`, `conformal_quantile` from
  `scripts.cp_monitor` and `PHASES`/`PHASE_ID` from `scripts.subphases` — do
  **not** modify `cp_monitor.py`, the working monitor stays untouched. Implement
  the K=1 arm (per-rollout max over all steps, one conformal quantile at alpha).
  Loop seeds x alpha grid, paired splits, collect (achieved FAR, detection) per
  arm. Write JSON + print a table.
- **Done when:** `--help` works and a 2-seed smoke run on task 0 prints both arms
  without error.

## Task 2 — Run the full ablation

- **Goal:** the numbers exist for tasks 0, 1 and 8.
- **Why:** a result on one task is an anecdote; the claim is about the method.
- **Files:** `data/results/ablation_global.json`
- **Steps:** 20 seeds, alpha grid spanning roughly 0.01-0.5, all three tasks.
- **Done when:** the JSON holds both arms x 3 tasks x full grid, and the printed
  table shows achieved FAR and detection for each.

## Task 3 — Interpret and record the verdict

- **Goal:** a stated answer to "does stage-aware beat global", with the number
  that supports it.
- **Why:** `docs/status.md:215` makes the next ~6.5 h of collection conditional
  on this answer. An unrecorded result cannot gate anything.
- **Files:** `docs/status.md` (new section), `docs/ablation-result.md`
- **Done when:** the verdict names the paired detection difference at matched
  FAR, its spread across seeds, and says plainly whether it supports collecting
  a 4th task.
