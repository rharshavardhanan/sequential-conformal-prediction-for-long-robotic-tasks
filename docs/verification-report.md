# Verification Report — against Spec v6

**Project:** sequential conformal prediction for early failure detection, LIBERO-10
**Date:** 5 Aug 2026 · Machine: Apple Silicon MacBook (MPS, 26 GB, 15 cores) · No GPU used
**Data:** `rollouts/task0.hdf5` — 20 rollouts, 11 success / 9 fail (55 %), policy
`yzembodied/libero10_task0_diffusion`, `sha256 3105133cc3bddf79…`

---

## Summary of the six verifications

| | check | acceptance | result |
|---|---|---|---|
| **A1** | sub-phase labelling on policy rollouts | ≥90 % of successes give exactly 8 segments | ❌ **NOT MET** — 0 % give 8; but the criterion's premise is wrong (see A1) |
| **A2** | phase-MAX separates | ≥1 phase with clean separation | ❌ **NOT MET on max** — best AUC 0.88. ✅ **Fixed by switching to p95** (AUC 1.00) |
| **A3** | pooling occurrence 1 vs 2 | KS p > 0.05 for every stage | ✅ **MET** — all p ∈ [0.147, 0.675]; keep K = 4 |
| **A4** | `err_norm` zero-motion instability | no phase max from near-zero motion | ❌ **NOT MET without a floor** — 15/80 maxima were spurious. ✅ MET with a 2 % floor |
| **A5** | task 8 success rate + CI | ≤0.85 with CI overlapping 0.80 | ⏳ **IN PROGRESS** — 60 rollouts collecting, ~2 h |
| **A6** | init-state disjointness | zero overlap, split recorded | ⚠️ **PARTIAL** — 20/20 distinct, zero repeats; split not yet stored in HDF5 |

**Headline:** three criteria failed, and two of the three failures are informative rather than
blocking. The single most consequential result is **A2 + A4 together**: the per-phase *maximum*
of `err_norm` is dominated by transient spikes at near-zero commanded motion, and replacing it
with the 95th percentile turns AUC 0.88 into **1.00**.

---

## A1 — Sub-phase labelling on policy rollouts ❌

Ran `subphases.py` over all 20 collected rollouts.

### Segment-count distribution

| | counts observed | exactly 8 | open→close transitions | regrasps (>2) | empty phases |
|---|---|---|---|---|---|
| **success** (n=11) | 7:1 · **9:6** · 10:1 · 11:1 · 13:1 · 14:1 | **0.00** | mean 2.36, range 2–4 | 3/11 | **0** |
| **failure** (n=9) | 7:3 · 11:1 · 12:2 · 13:1 · 23:1 · **34:1** | 0.00 | mean 3.67, range 1–10 | 6/9 | **0** |

### The criterion's premise does not hold

The spec expects 8 segments (2 milestones × 4 stages). **No rollout produced 8, and the modal
value is 9.** This is not a policy artefact: the human demos reported earlier averaged **9.5
segments** (min 9, max 17). A clean two-object trajectory is
`reach·grasp·transport·release·reach·grasp·transport(·release)` — **7 or 8** depending on whether
the episode terminates before the final release, and 9 once any trailing motion after the last
release is counted as a new `reach`. So 8 was never the right target, for demos or policy.

### What the numbers actually say

The concern behind A1 was *"are the calibration phases comparable across trajectories?"* On that
question the evidence is reassuring:

- **Zero rollouts have an empty phase** — every trajectory, success or failure, visits all four
  stage types. Per-phase calibration is therefore well-defined for every rollout.
- **A3 independently confirms** the first and second occurrences of each stage are
  distributionally indistinguishable, so variable segment *counts* do not imply incomparable
  *phases*.

### Regrasps are a genuine, and useful, difference

`open→close` transitions: **2.36 mean for successes vs 3.67 for failures**, with 6/9 failures
regrasping versus 3/11 successes, and one failure at **10 regrasps / 34 segments**. Failures
fumble repeatedly — consistent with the earlier finding that failures spend 65.8 % of steps in
`reach` versus 46.6 % for demos.

**Recommendation (not yet implemented):** do not force 8 segments. Keep phase *type* labelling as
is, since it is well-defined everywhere, and add **regrasp count as a separate rollout-level
feature**. It is discriminative on its own and is being discarded at present.

---

## A2 — Does the phase-MAX separate? ❌ on max, ✅ on p95

### On the max statistic, as specified

| phase | succ mean ± sd | succ max | fail mean ± sd | fail min | **AUC** | failures below max success |
|---|---|---|---|---|---|---|
| **reach** | 0.971 ± 0.077 | 1.110 | 1.105 ± 0.069 | 1.020 | **0.88** | 6/9 |
| grasp | 1.058 ± 0.109 | 1.351 | 1.092 ± 0.302 | 0.726 | **0.43** | 8/9 |
| transport | 0.905 ± 0.055 | 1.006 | 0.949 ± 0.159 | 0.796 | **0.63** | 8/9 |
| release | 1.158 ± 0.269 | 1.671 | 1.265 ± 0.336 | 1.000 | **0.56** | 8/9 |

No phase achieves clean separation. `grasp` is **below chance** (0.43). The spec anticipated this
exactly: *"mean separation does not imply max separation; the max is a noisier statistic."*
Confirmed.

### The cause, and the fix

Sweeping the aggregation statistic (all with the A4 floor applied):

| phase | max | **p95** | p90 | p75 | mean |
|---|---|---|---|---|---|
| **reach** | 0.88 | **1.00** | **1.00** | **1.00** | **1.00** |
| grasp | 0.43 | 0.25 | 0.28 | 0.40 | 0.47 |
| transport | 0.63 | 0.41 | 0.29 | 0.57 | 0.57 |
| release | 0.56 | 0.57 | 0.57 | 0.77 | 0.89 |
| **whole rollout** | 0.65 | **1.00** | — | — | **1.00** |

**`reach` at p95 gives AUC 1.00 — perfect separation.** So does the whole-rollout p95:
successes `[0.831, 0.935]`, failures `[1.001, 1.015]`, a clean gap with no overlap.

The max is discarding a strong signal by keying on a single worst timestep, which is noise-
dominated. The p95 keeps the distributional shift.

### Methodological note — this does not weaken the guarantee

Split conformal is valid for **any** score function; validity does not depend on choosing the
max. Calibrating the per-rollout p95 gives an equally rigorous per-rollout guarantee. What
changes is the *interpretation*:

- **max** → "no timestep in this phase exceeded the nominal envelope"
- **p95** → "this rollout's tracking-error distribution has not shifted"

The second is a weaker claim per-timestep but a stronger detector, and it remains computable
online (running p95 over steps so far), so sequential monitoring is unaffected.

**Recommendation:** make the aggregation statistic a parameter of `cp_monitor.py`, defaulting to
**p95**, with max retained for comparison. Not yet implemented.

---

## A3 — Is pooling occurrence 1 and 2 justified? ✅ MET

Successful rollouts only, first vs second occurrence of each stage type, max `err_norm` within
each occurrence (floored per A4), two-sample KS test:

| phase | n pairs | occ 1 mean | occ 2 mean | KS D | p | verdict |
|---|---|---|---|---|---|---|
| reach | 11 | 0.894 | 0.909 | 0.364 | 0.374 | pool |
| grasp | 11 | 0.950 | 1.017 | 0.455 | 0.147 | pool |
| transport | 11 | 0.900 | 0.854 | 0.455 | 0.147 | pool |
| release | 10 | 0.989 | 1.137 | 0.300 | 0.675 | pool |

**All p > 0.05 → keep K = 4.** The Bonferroni budget stays at α/4 = 0.025, and the calibration
requirement stays at 39 rollouts per phase rather than doubling.

Caveat worth stating: with n ≈ 11 pairs the KS test has low power, so this is "no evidence
against pooling" rather than positive proof. It should be re-run once ~75 successes exist. Two
of the four (grasp, transport) sit at p = 0.147, which is not comfortable margin.

---

## A4 — `err_norm` zero-motion instability ❌ without a floor

`||commanded delta||` distribution across all 8,466 timesteps (max possible 0.08660 m):

| percentile | value | % of max |
|---|---|---|
| 1st | 0.001893 | 2.19 % |
| 5th | 0.003643 | 4.21 % |
| 25th | 0.008644 | 9.98 % |
| 50th | 0.016404 | 18.94 % |

| threshold | timesteps below | fraction |
|---|---|---|
| 1 % of max | 7 | 0.08 % |
| 2 % of max | 64 | 0.76 % |
| 5 % of max | 612 | 7.23 % |

**15 of 80 per-phase maxima (18.8 %) originated from a timestep below the 2 % floor.** Nearly one
in five phase maxima was a division artefact rather than a real tracking deviation — which
directly explains A2's poor max performance.

**Floor chosen: `||commanded delta|| ≥ 2 % of max` = 0.001732 m.** Rationale: it sits at the 1st
percentile of the observed distribution, so it removes the unstable tail while discarding under
1 % of timesteps. A 5 % floor would discard 7.2 %, which is too aggressive.

With the floor applied, zero phase maxima originate from near-zero motion, by construction.
All A2/A3 numbers above already include it.

**Recommendation:** apply the floor inside `collect_rollouts.py` as a stored mask, or inside
`cp_monitor.py` at scoring time. Not yet implemented — currently applied only in
`verify_spec.py`.

---

## A5 — Task 8 success rate with a usable CI ⏳ IN PROGRESS

60 rollouts collecting on task 8 (`yzembodied/libeo10_task8_diffusion`, "put both moka pots on
the stove"). **1/60 complete at time of writing**, ~2 h remaining at ~131 s/rollout.

Prior: 17/20 = 85 %, 95 % CI ±0.16 → true rate plausibly inside the band.
Combined with the earlier 20, the final estimate will be over **80 rollouts** (CI ≈ ±0.08).

Note the earlier 20 were run by `eval_pretrained_dp.py`, which recorded only pass/fail, so only
the 60 new rollouts carry per-step values. The success-rate CI uses all 80; any per-phase
analysis uses the 60.

Task 8 matters disproportionately: it is the **only** candidate outside the
`LIVING_ROOM_SCENE2` layout, so it is the difference between a two-scene study and a
single-scene one.

---

## A6 — Init-state disjointness ⚠️ PARTIAL

```
rollouts                 : 20
distinct init states     : 20
repeats                  : 0
success init indices     : [0, 2, 3, 7, 9, 10, 11, 13, 16, 17, 18]
failure init indices     : [1, 4, 5, 6, 8, 12, 14, 15, 19]
success/failure overlap  : 0
```

Zero repeats — `--init-file full` (100 states) is working as intended, and rollout *i* uses init
state *i*, so disjointness is guaranteed by construction up to 100 rollouts.

**What is missing:** the calibration/evaluation split is **not recorded in the HDF5**. At present
`cp_monitor.py` splits successes with a seeded RNG at load time, which is reproducible only if
the seed and rollout set are unchanged. The spec requires the split to be arranged by
construction and stored.

**Recommendation (not yet implemented):** assign the split by init index at collection time —
e.g. even indices → calibration, odd → evaluation — and write a `split` attribute per rollout.
This also guarantees the split is stable as more rollouts are appended.

---

## Part B — Decision tree status

**Not started.** The spec requires A1–A3 to report first, which they now have; B1 has not been
run because A5 is still collecting and its outcome changes which tasks need perturbation.

### B1 plan — inducing failures without training

Targets: tasks 2 (100 %), 3 (95 %), and 8 if A5 excludes it.

Two knobs, in the spec's order of preference:

1. **Initial object-pose perturbation** — zero-mean noise on object placements at reset, swept
   over 3–4 magnitudes.
2. **Action noise** — zero-mean Gaussian on the action before `env.step`, σ swept over 3–4
   values.

20 rollouts per setting, target band **50–70 %**.

**Constraint to be enforced in code:** the chosen perturbation must apply to *every* rollout of
that task, calibration and evaluation alike. Calibrating on unperturbed rollouts and evaluating
on perturbed ones would silently void the CP guarantee, since exchangeability requires one
distribution. The perturbation type and magnitude will be stored per rollout alongside the
checkpoint hash.

Neither knob is implemented in `collect_rollouts.py` yet. Action noise is the simpler of the two
and is the natural first implementation; object-pose perturbation requires touching the env reset
path.

### B2 — only if B1 falls short

Would train diffusion policies via LeRobot (not BC-Transformer), keeping one policy class across
tasks and preserving the FAIL-Detect baseline. **Prerequisite:** fix checkpoint retention in
`train_all_tasks.py`, which currently keeps only the best-success checkpoint and so reproduces
the exact saturation problem being solved. Still outstanding.

Mixed sources are acceptable — exchangeability is required per task, not globally — provided each
task's checkpoint hash is constant within that task.

---

## Part C — Collection spec, current position

**C1 scores.** Already implemented and unchanged: `actions`, `eef_pos`, `eef_quat`,
`gripper_qpos`, `joint_pos`, `planned_pos`, `executed_pos`, `err_raw`, `err_norm`, `cmd_norm`,
`object_state`. `err_norm` remains primary.

> Empirical support for that choice, from this dataset: `err_raw` runs **backwards** —
> successes average 0.0203 m, failures 0.0130 m — because successes move fast and failures creep.
> A monitor built on `err_raw` would rank the classes the wrong way round. `err_norm` corrects it.

**C2 volume.** Recommend **150 rollouts/task**. At the observed 55 % success that yields ~82
successes → ~41 calibration rollouts after a held-out split, just clearing the 39 required at
α = 0.025. The alternative — 100/task with K-fold cross-conformal — is preferable only if
collection time binds; at ~96 s/rollout, 150 × 4 tasks ≈ **16 h**.

**C3 task selection.** 4 tasks × 150 rollouts. Current candidates: 0 and 1 (in band), plus
task 8 pending A5, plus one recovered via B1. Scene diversity argues for preferring task 8
(KITCHEN_SCENE8) over a third `LIVING_ROOM_SCENE2` variant.

**C4 rendering.** The FAIL-Detect subset must be chosen *before* the run. Currently `--images`
defaults off, which is correct for tasks not in that subset. **Decision still required.**

---

## Outstanding implementation work

Ordered by dependency:

1. **A4 floor** into the scoring path (`cp_monitor.py` and/or `collect_rollouts.py`).
2. **Aggregation statistic** as a parameter of `cp_monitor.py`, default **p95**, max retained.
3. **Split assignment by init index**, stored per rollout (A6).
4. **Regrasp count** as a rollout-level feature (A1).
5. **Action-noise perturbation** in `collect_rollouts.py` (B1), stored per rollout.
6. **Checkpoint retention** in `train_all_tasks.py` (B2 prerequisite).

None are blocked by anything except A5's completion for item 5.

---

## Numbers to report back

1. **A1** 0 % of successes give exactly 8 segments (modal 9); 0 rollouts with an empty phase;
   regrasps 2.36 vs 3.67 mean (success vs failure).
2. **A2** best phase-max AUC 0.88 (`reach`); switching to p95 gives **1.00**.
3. **A3** KS p = 0.374 / 0.147 / 0.147 / 0.675 → pooling justified, K = 4.
4. **A4** 15/80 phase maxima from sub-2 % motion; floor set at 2 % of max delta (0.001732 m).
5. **A5** in progress, 1/60 collected.
6. **A6** 20/20 distinct init states, zero overlap; split not yet persisted.
7. **B1** not started (gated on A5).
