# Project Status — Sequential Conformal Prediction for Safe and Compute-Efficient Long-Horizon Robot Manipulation

**Purpose:** sole context needed to resume work in a new conversation.
**Supersedes all earlier status summaries** (the collection-side report it replaced is kept at
`docs/status-2026-09-08-collection.md`).

**Covers:** theory (locked), the policy-arbitration application, Review 2 materials, and real-data
results through Spec v11 across three tasks.

---

## 1. Identity

- **Title:** Sequential Conformal Prediction for Safe and Compute-Efficient Long-Horizon Robot
  Manipulation — A Subtask-Conditioned Calibration Framework, and its Application to Runtime
  Policy Arbitration.
- **Team:** Akileshguhan A (23BRS1070) — theory, calibration framework, evaluation, arbitration
  design. Harshavardhanan R (23BRS1364) — MuJoCo/LIBERO rollout collection, policy integration.
- **Programme:** B.Tech CSE (AI & Robotics), VIT Chennai, graduating 2027.
- **Benchmark:** LIBERO-Long (`libero_10`), Franka Panda, MuJoCo/robosuite.
- **Target venues:** ICRA, IEEE RA-L, CoRL workshops.
- **One-line:** calibrate an independent conformal threshold per execution phase, combine via
  Bonferroni into a certificate holding simultaneously across the task horizon, and use the same
  calibrated signal to arbitrate between a cheap and an expensive policy at runtime.

## 2. Course context

VIT Chennai AI & Robotics rubric: 30 marks across three reviews (R1 proposal, R2 domain/literature/
methodology/design — 20 marks, four 5-mark components, no live demo required — R3 final live demo
+ documentation). Individual viva at every review. Expo 22 October.

**Review 2 delivered:** a formal Word document and two PPTX decks (a styled version and a plain
version matching the guide's requested reference style), built to the exact rubric order —
domain/problem, literature (24 verified sources across four tables), methodology, design — with a
native-PowerPoint system architecture diagram (host/edge split, monitor/arbiter block, policy
arms, GPIO output). No results section per guide's explicit instruction; replaced with
"Implementation Progress" and "Design Decisions Taken and Justified" (the six rejected-alternative
decisions, reframed as engineering rationale) and "Expected Outcomes."

## 3. The theory — locked

**Original flaw, corrected:** pooling per-timestep scores for calibration gives only marginal
per-timestep coverage and violates exchangeability (timesteps within a trajectory are
autocorrelated). Fixed by changing the calibration unit to the per-trajectory phase maximum:

```
S_k(tau) = max { s(t) : t in window_k(tau), motion_mask(t) }
```

**Threshold:** order statistic, `q_k = ceil((1-alpha_k)(n+1))`-th smallest of
`{S_k(tau_1),...,S_k(tau_n)}`. Infinite (vacuous) if that rank exceeds n — reported as a
diagnostic, never hidden. Two feasibility bounds, not one:

- **Finite threshold:** `n >= 1/alpha_k - 1`. At alpha=0.10, K=4 -> **n >= 39**.
- **Threshold strictly below the sample maximum** (`rank < n`): **n >= 79**. Below this the
  threshold is literally the largest calibration value — technically feasible but not efficient.
  This second bound, not the first, is what determines whether Subtask CP actually beats Flat CP.

**Joint guarantee:** Bonferroni, `alpha_k = alpha/K`, union bound -> `P(safe in every phase) >=
1-alpha`, valid under arbitrary dependence across phases — deliberate, since the project's premise
is that errors propagate across subtask boundaries, making tighter independence-assuming rules
inapplicable.

**Nonconformity score:** compliance residual, not raw tracking error —

```
s(t) = || e_hat(t) - e(t) ||_2 / || Delta_cmd(t) ||
```

Raw error is confounded by the OSC controller's near-constant compliance gain (0.251), which makes
it mostly report commanded speed; empirically it ranks failures *below* successes (successes
0.0203 m, failures 0.0130 m). The compliance residual separates cleanly (successes 0.7607+/-0.0059,
failures 0.8660+/-0.0296 on the original 20-rollout sample).

**Motion floor:** timesteps below 2% of the max commandable step are masked for scoring, never
deleted — near-zero motion is itself the stall signature.

**Stall detection:** separate mechanism, dwell-time quantile `d_k` at its own risk parameter beta,
not drawn from alpha. A stationary robot has a *low* deviation score, so the deviation machinery
alone cannot see it.

**Phases (K=4):** BDDL goal predicates as milestones (>=10-timestep continuity rule), subdivided by
gripper open/close transitions into reach/grasp/transport/release, pooled by category across
milestones. Real BDDL parsing gives only 1-3 predicates per task (not 4-6 as originally assumed);
predicate-only conditioning gives no efficiency gain over flat — the gain requires the sub-phase
level, confirmed both synthetically and on real data.

**Progress-weighted score:** tested and rejected. Weighting late deviations more heavily inflates
the threshold (calibrated on weighted nominal maxima, which peak late) and collapses early-failure
detection (0.807 -> 0.087 in synthetic testing). Reported as a negative-result ablation, not a
contribution.

**Aggregation statistic:** `max` is the only one currently trusted. A bug (calibrate on one
statistic, evaluate coverage as "no timestep exceeds q" — correct only for max) was found and
fixed in the pipeline; p95/p90/p75/mean tables are parked pending a proper held-out replication,
not adopted.

## 4. The application — policy arbitration

Reuses the same calibrated thresholds for a second decision: escalate from a cheap policy
(BC-RNN) to an expensive one (ACT) when the signal departs from nominal, instead of only raising a
flag.

**Critical correction:** the original coverage guarantee does **not** transfer to the arbitrated
trajectory, since it's calibrated on cheap-policy-only rollouts and arbitration changes which
policy generates the trajectory. What transfers is a **bounded spurious-escalation rate**: a
second threshold `q_k,esc` is drawn from the same calibration set at its own larger risk budget
`alpha_esc > alpha`, so `q_k,esc < q_k` and escalation happens strictly before a safety violation
would. `P(escalate anywhere on a healthy rollout) <= alpha_esc` — an efficiency guarantee, not a
safety certificate on the arbitrated system.

A naive "escalate at a fraction of q_k" rule was tested and rejected: it escalates below the
calibrated quantile, so the bound doesn't cover it (0.9x q_k gave 56.7% spurious escalation against
a 10% budget; 0.8x gave 100%). The dual-threshold fix was verified to hold the bound at every
tested alpha_esc.

**Offline-replay results (synthetic, not yet run on real data):** at alpha_esc=0.30, ~95.7% of
timesteps on the cheap policy, 94% of failing rollouts still escalated, flagged ~26% into the
episode. Hysteresis (asymmetric escalate-immediately / de-escalate-reluctantly with a chunk-
boundary constraint) reduced switching on failing rollouts from 6.7 to 1.5 transitions/episode.

**Hardware:** target Jetson Orin Nano (over the original Nano — ACT is ~80M parameters, needs
tensor cores). Availability unconfirmed; fallback is a Nano with a reduced-capacity ACT. Latency
benchmark on-device is the next concrete step and needs no simulator.

**Status:** design complete, offline-validated. Not yet integrated with real rollout data, not yet
deployed to hardware. BC-RNN/ACT not yet trained (training is offline, needs no MuJoCo, can run in
parallel with rollout collection on a rented GPU).

## 5. Real-data collection — status by task

Pipeline built as a standalone package (`scp/`: `io.py`, `phases.py`, `scoring.py`, `calibrate.py`,
`methods.py`, `evaluate.py`) operating purely on recorded HDF5 files — no simulator dependency.

| task | scene | rollouts | success rate | cal_frac (pre-registered) | n_cal | status |
|---|---|---|---|---|---|---|
| 0 | LIVING_ROOM_SCENE2 | 150 | 68.7% (103/47) | 0.80 | 82 | done |
| 1 | LIVING_ROOM_SCENE2 | 150 | 72.7% (109/41) | 0.75 | 81 | done |
| 8 | KITCHEN_SCENE8 | 150 | 85.3% (128/22) | 0.65 | 83 | done |
| wave-2 (task 2 or 3) | — | target ~200 | target ~50% | — | — | paused, see section 8 |

All three: single checkpoint, single fingerprint, 150/150 complete, zero errors, zero duplicate
init states, session-effect test passed (no significant difference in success rate or err_norm
across session boundaries once de-confounded from init-state source).

**Known limitation, disclosed, not corrected for:** the init-state pool for these three tasks is a
mixture of LIBERO's shipped 100 states plus 50 generated via `env.reset()`. Diagnostic confirmed
no significant difference in err_norm between shipped and generated among successes (pooled KS
p=0.47); mixture thresholds are systematically *lower* than shipped-only (conservative direction).
Stated as: coverage claims are with respect to this disclosed mixture distribution. Wave-2
collection switches to Option B (uniform sampling with replacement from the shipped 100 only) to
avoid repeating this.

## 6. Real-data results — current best (post Spec v11, cal_frac pre-registered per task)

### 6.1 Calibration efficiency — reversed from the first assessment

At the original default (cal_frac 0.5, task 0 only), Subtask CP was **wider** than Flat CP
(1.517 vs 1.410 — every phase threshold was the sample maximum, n_cal=52 < 79). Diagnosed and
fixed by pre-registering cal_frac per task as the smallest value clearing the n>=79 bound:

| task | Flat band | Subtask band | ratio |
|---|---|---|---|
| 0 | 1.376 | 1.362 | 0.990 |
| 1 | 1.270 | 1.133 | 0.892 |
| 8 | 1.334 | 1.234 | 0.926 |

Subtask CP is now tighter on all three tasks, both arms at target coverage (0.89-0.92 vs 0.90),
100% feasibility. **This is the corrected, current headline** — the earlier "no efficiency
advantage" conclusion was a sample-size artifact, not a property of the method.

**A hypothesis that was tested and killed:** longer phase windows inflate `S_k` just from having
more draws (duration bias). Checked directly — duration ordering is identical across tasks, but
threshold ordering differs in all three, and the correlation is *negative* throughout (Pearson
-0.30 / -0.93 / -0.42, opposite sign to the hypothesis). Do not resurrect this explanation.

### 6.2 Detection at matched false-alarm rate — the headline, with a real boundary

Comparing at matched *nominal alpha* is invalid — Bonferroni makes the two arms land at different
*achieved* FAR for the same alpha, so it's reading two points off two different ROC curves. Matched
*achieved FAR* is the correct comparison and is confirmed calibrated (achieved FAR tracks nominal
alpha reasonably for both arms on all three tasks).

| achieved FAR | task 0 dTPR | task 1 dTPR | task 8 dTPR | mean |
|---|---|---|---|---|
| 0.05 | +0.042 | +0.093 | +0.068 | +0.068 |
| 0.10 | +0.044 | +0.112 | +0.129 | +0.095 |
| 0.15 | +0.018 | +0.059 | +0.056 | +0.044 |
| 0.20 | -0.030 | +0.093 | -0.029 | +0.012 |
| 0.25 | -0.029 | +0.089 | -0.017 | +0.014 |

**All three tasks agree in sign at FAR <= 0.15**, peaking at FAR 0.10. **At FAR >= 0.20 they
disagree** — task 1 stays positive, tasks 0 and 8 go slightly negative with only 25-35% of paired
draws favouring Subtask CP. This disagreement is reported explicitly, not averaged away — the
honest claim is an advantage in the FAR <= 0.15 operating window, not everywhere.

**Low-FAR boundary:** Subtask CP cannot operate below FAR ~ 0.04-0.05 on any task (Bonferroni
floor `4/(n_cal+1) ~ 0.048`); Flat CP remains feasible down to 0.005-0.017. The usable window is
bounded on both sides, not just below by sample size.

**Open discrepancy, not yet resolved:** an earlier matched-FAR ablation report (task 0 only)
showed gains *increasing* with FAR (+0.114 at 0.20, +0.151 at 0.25) — the opposite shape from the
v11 result above. The two were computed by different implementations; v11 used the pre-registered
per-task cal_frac and is the more trustworthy of the two, but the discrepancy has not been
diffed/explained and should be before either number is reported as final.

### 6.3 What replicated cleanly from synthetic predictions

- Coverage validity: both arms land near the 0.90 target on all three tasks.
- The n>=39 feasibility bound: confirmed exactly (vacuous below, feasible at/above).
- Single-split coverage variance: real spread (~82-97% at n~40-scale) matches synthetic
  predictions closely.
- Detection earliness: Subtask CP flags consistently earlier than Flat CP at matched coverage.

### 6.4 A confirmed pipeline bug, now fixed

Coverage was being evaluated as "no timestep exceeds q" regardless of which aggregation statistic
calibrated q — correct only for `max`. This produced impossible results for other statistics
(coverage 0.000 with detection 1.000). Fixed by making the coverage event use the same statistic
as calibration; `max` results were unaffected by the bug and are the ones reported above.

**A second asymmetry, fixed 2026-09-09.** `scripts/cp_monitor.py:first_alarm()` evaluated alarms
over *every* timestep while calibration excluded low-motion steps via the stored motion mask. A
masked step is the extreme tail of the score (mean err_norm 0.99, max 2.78, against 0.82/2.37 for
scorable steps), so the evaluation arm could alarm on steps calibration never saw. `first_alarm`
now applies the same mask. On task 0, seed-0 split, cal_frac 0.80, alpha 0.10, Bonferroni, agg
`max`: **FAR 0.238 -> 0.190 and detection 0.617 -> 0.447**. The four thresholds are unchanged
(calibration always masked); only the evaluation numbers move.

### 6.5 A caught-before-damage bug in the (paused) noise-injection code

For wave-2 collection, `planned_pos` was computed from the clean (pre-noise) action while the
controller received the noised one — meaning the score would partly measure injected noise as
tracking error. Caught before any real collection ran. **Fix applied 2026-09-08** (not yet
probe-verified): both `planned_pos` and `cmd_norm` now derive from the actually-executed (noised)
action, and the gripper channel (dim 6) is excluded from noise injection entirely — which also
makes the executed gripper channel equal the policy's, so phase segmentation is unambiguous by
construction.

## 7. Evaluation protocol (unchanged, still enforced)

1. Coverage computed on successful rollouts only — failures are not exchangeable with a
   success-only calibration set; they're what should be flagged, not covered.
2. Coverage reported as a distribution over repeated calibration/evaluation splits, never one
   split.
3. Threshold feasibility reported alongside coverage — a vacuous monitor scores perfect coverage
   while detecting nothing.
4. Never pool rollouts across tasks. Never pool across different init-state regimes, checkpoints,
   or perturbation settings within one task.
5. cal_frac is fixed by a pre-registered rule (smallest value clearing n>=79 per phase) **before**
   results are seen, per task independently — not tuned to the outcome afterward.
6. The unit of replication across tasks is the task, not the resampled split. Three tasks
   agreeing is three points of evidence, not sixty.

## 8. Open items, in priority order

1. **Diff the two matched-FAR ablation implementations** (6.2) — they disagree in shape at high
   FAR and this needs resolving before either is reported as final.
2. **Wave-2 collection** — paused pending: (a) the noise-injection fix (6.5) verified via a short
   probe, (b) the B1 sigma sweep on task 3 (target ~50% success, ~200 rollouts planned to give
   ~99 successes and ~99 failures), (c) switch to Option B init-state sampling for this task.
3. **Jetson latency benchmark** for BC-RNN and ACT — no simulator needed, just checkpoints and a
   saved observation. Determines whether the arbitration hardware plan is viable as specified.
4. **BC-RNN and ACT training** — offline, no MuJoCo, can run in parallel with wave-2 collection.
5. **FAIL-Detect baseline** — needs the recorded action chunks and images (already being
   collected) plus a trained scoring network; not started.
6. **Integrate real rollout data into the arbitration offline-replay harness** — currently
   validated only on synthetic data.
7. **Empty-phase rollouts — RESOLVED 2026-09-09, count corrected from 2 to 3, all in task 0:**

   | rollout | split | gripper closed | phases present | cause |
   |---|---|---|---|---|
   | 0036 | cal | 40 steps, 1 interval | reach, transport, release | starts closed at t=0, opens at t~40, never re-closes -> no `grasp` |
   | 0075 | eval | 21 steps over 4 intervals (2/6/4/9) | reach, grasp, release | no closed interval >10 steps, so the +/-5 windows consume them all -> no `transport` |
   | 0076 | cal | 0 steps | reach only | gripper never closes in 600 steps -> no grasp/transport/release |

   Tasks 1 and 8 are clean (0 each). All three are failures, so **calibration is unaffected**
   (successes only). **Action required in `scp/evaluate.py`:** rollout 0075 is `eval`-split and
   yields `NaN` for `transport`; if detection reduces across per-phase `S_k` with `np.max`, the
   `NaN` propagates and silently converts a detected failure into a miss. Use `np.nanmax`, or
   alarm per-timestep against `q_{phase(t)}` (immune, since every timestep carries a label).
   Worth up to 1/47 ~ 0.021 on the task-0 detection rate, in the direction of understating it.

## 8b. RoboTurk Part 5.3 — compute efficiency (measured 2026-09-12)

Detail, charts, regeneration: `roboturk/benchmark/README.md`. Side by side only — different
action space, different score, different hardware from the LIBERO results above; do not pool.

| per control step | CPU | MPS |
|---|---|---|
| BC-RNN (1.15 M) | 0.059 ms | **0.185 ms** |
| ACT v2 (9.81 M), chunked ÷16 | 0.132 ms | **0.124 ms** |
| **ACT-alone ÷ arbitrated** @ f=11.16% | **1.80×** | **0.62×** |

**H3 holds on CPU and fails on MPS.** On MPS, chunked ACT is cheaper *per step* than BC-RNN,
so arbitration costs more than running ACT everywhere at every escalation fraction. Cause:
BC-RNN is a 2-layer LSTM stepped one frame at a time, so on GPU it is dominated by per-step
kernel-launch overhead it cannot amortise, while ACT emits 16 actions from one parallel
forward. Action chunking, not parameter count, decides per-step cost. Arbitration also cannot
go below BC-RNN-alone, because the monitor needs BC-RNN's score every step.

Two caveats that must travel with any quote of these numbers: ACT is priced **chunked**
(deployed) and the **per-step** figure is an upper bound only, differing by up to 16×; and
timing is **throughput**, not per-call latency, because a per-call `torch.mps.synchronize()`
charges ~0.20 ms to BC-RNN that a real pipelined loop never pays. Apple-Silicon-specific.

## 9. Files produced

| file | purpose |
|---|---|
| `Sequential_CP_Draft_Rev2.docx` | Paper draft with corrected theory, Appendix A changelog |
| `Review2_Submission.docx` | Formal R2 document, all four rubric components |
| `Review2_Final.pptx` / plain variant | R2 slide decks, rubric-ordered, no results section |
| `System_Architecture_Diagram.pptx` | Native-shape host/edge/monitor/arbiter diagram |
| `cp_theory_validation.py` | Monte Carlo validation of the corrected calibration theorem |
| `benchmark_harness.py` | Synthetic 4-method, 5-metric benchmark, repeated-split protocol |
| `granularity_tradeoff.py` | Phase-granularity synthetic experiment |
| `cp_arbiter.py` / `tune_arbiter.py` / `dual_threshold.py` | Policy arbitration state machine + validation |
| `bddl_parser.py` | Standalone BDDL goal-predicate parser, no simulator dependency |
| `scp/` package (`io`, `phases`, `scoring`, `calibrate`, `methods`, `evaluate`) | Real-data calibration pipeline, operates on HDF5 only |
| `Spec_v6` -> `Spec_v11` | Sequential rollout-collection and calibration instruction files |
| `SCP_Pipeline_Bootstrap.md` | Zero-context bootstrap instructions for the pipeline |
| `SCP_Code_Audit_and_Fixes.md` | Bug list (aggregation/coverage mismatch, sample-max threshold, duration-bias test) |
| `roboturk/compute_benchmark.py` -> `roboturk/benchmark/` | Part 5.3 compute efficiency: latency, cost sweep, estimate-vs-direct check |
| `roboturk/benchmark_charts.py` | The three Part 5.3 charts, rendered from the JSON only |
| `roboturk/calibrate_bcrnn.py` -> `partD_v2.json` | RoboTurk BC-RNN thresholds on the 7 continuous dims, with an 8-dim reproduction gate |

## 10. Do not re-litigate

- **Don't pool per-timestep scores for calibration** — phase-level max aggregation only.
- **Don't use cal_frac=0.5 as a default** — it puts every task below the n>=79 efficiency bound.
  Pre-register per task via the smallest-cal_frac-clearing-79 rule.
- **Don't compare Subtask CP and Flat CP at matched nominal alpha** — compare at matched achieved
  FAR.
- **Don't use `err_raw` as the score** — the compliance confound ranks classes backwards.
- **Don't drop low-motion timesteps** — mask them; they feed the stall detector.
- **Don't pool rollouts across tasks, or across different init-state/checkpoint/perturbation
  regimes within a task.**
- **Don't adopt p95 (or any non-max aggregation) as default** — parked pending held-out
  replication on data not used to select it; also the coverage-evaluation bug made all non-max
  results computed before the fix invalid.
- **Don't claim the coverage guarantee transfers to the arbitrated trajectory** — only the bounded
  spurious-escalation rate does.
- **Don't claim a Subtask CP advantage at FAR > 0.15** — the three tasks disagree there.
- **Don't skip recording images or the action chunk during collection** — neither is recoverable
  afterward, and FAIL-Detect needs both.
- **Progress-weighted scoring is a negative-result ablation**, not a contribution.
- **The duration-bias hypothesis for band-width differences is refuted** — do not reintroduce it
  without new evidence; the correlation runs the opposite direction on real data.
