# RoboTurk — ACT Training and Full Arbitration Benchmark

**Delivery:** hand this file to Claude Code with the single prompt `execute`.

**Supersedes:** Part C of `RoboTurk_Benchmark_and_Arbitration.md` if that file has not yet been run
— this file's Part 2 subsumes it. If Part C's escalation-threshold results already exist, Part 0
below detects and reuses them rather than recomputing.

**No MuJoCo, robosuite, or LIBERO anywhere in this file.**

---

## PART 0 — Verify what already exists before doing anything

Check for, and report the status of, each of the following. Do not retrain or recompute anything
that already exists and is complete — reuse it.

| Artifact | Expected location | If missing |
|---|---|---|
| BC-RNN checkpoint | `roboturk_bcrnn_state_only.pt` | Stop — this file requires it |
| BC-RNN fingerprint | `fingerprint.json` | Stop — cannot verify checkpoint identity without it |
| Resampled data cache | `roboturk_20hz.npz` | Stop — required for ACT training on the same episodes/split |
| Safety threshold `q̂_k` | `partD.json` | Stop — required for the dual-threshold check in Part 3 |
| Escalation threshold `q̂_k,esc` | output of `RoboTurk_Benchmark_and_Arbitration.md` Part C, if run | If absent, compute in Part 3 below |

**Confirm BC-RNN's training is sufficient to proceed:** checkpoint frozen and fingerprinted (yes,
per `fingerprint.json`), predicted-vs-demonstrated scores already computed on the fixed 79/80
calibration/evaluation split (yes, per `partD.json`). This is sufficient — no further BC-RNN
training is needed or should be attempted here.

---

## PART 1 — The corrected hypothesis: read before writing any comparison code

Do not target or imply "arbitration beats both BC-RNN alone and ACT alone on fidelity." That is
not achievable by construction: arbitration runs the expensive policy on a subset of timesteps,
so its fidelity to the human is bounded above by ACT-alone's fidelity, never below it. The correct,
falsifiable hypotheses are:

- **H1:** Arbitrated fidelity is measurably better than BC-RNN-alone fidelity (lower mean
  divergence from the human). Expected to hold.
- **H2:** Arbitrated fidelity is close to, but not better than, ACT-alone fidelity. Near-parity is
  the expected and acceptable outcome — do not treat failing to beat ACT here as a negative result.
- **H3:** Arbitrated compute cost is substantially lower than ACT-alone compute cost. This is the
  primary claimed benefit of the whole design.
- **Composite:** fidelity achieved per unit of compute spent is better for arbitration than for
  either policy alone. This is the metric that can legitimately show arbitration "winning" outright
  — report it, and let H1–H3 stand as the supporting evidence for it.

Report whichever of H1–H3 the data actually supports. If H1 fails to hold, or arbitration's fidelity
is *not* close to ACT's, report that plainly — it would mean the escalation threshold is not
triggering appropriately, which is a real finding requiring investigation, not something to smooth
over in the writeup.

---

## PART 2 — Train ACT

Same conventions as BC-RNN, so the two are trained on identical inputs and directly comparable.

### 2.1 Data

Reuse `roboturk_20hz.npz` — the same 531-episode, 20Hz-resampled cache. Reuse the **exact same**
train/val split (316/56) for policy training, and the exact same calibration/evaluation split
(79/80) held out and untouched, matching `fingerprint.json`'s recorded split exactly. Do not
redraw any split.

### 2.2 Input and output

- **Input:** state-only, same 37-dim proprioception (`eef_poses` + `joint_states_arm` +
  `joint_states_gripper`) as BC-RNN — no images, per the same D2 decision. This keeps the fidelity
  comparison clean: both policies see identical inputs, so any difference is attributable to
  architecture and capacity, not to one policy having more information than the other.
- **Output:** the same 8-dim D1 action (`user_control[:,0:7]` + grasp bool), same space as BC-RNN's
  target, so scores are directly comparable without any cross-space conversion.

### 2.3 Architecture

Action Chunking Transformer per the standard design: transformer encoder/decoder over the state
input, action-chunk output (choose a chunk size appropriate to the resampled 20Hz grid — a chunk of
8–16 steps, i.e. 0.4–0.8s, is a reasonable starting point given the episode lengths already
measured). State plainly that this is a state-only ACT variant, analogous to the state-only BC-RNN,
for the same reason: keep the compute-cost comparison meaningful (a vision-based ACT would not be
directly comparable to a state-only BC-RNN on cost).

### 2.4 Training

Bounded, checkpointed epochs, same discipline as BC-RNN training (periodic saves, log train/val
loss). Given ACT is the "expensive" arm, it is fine and expected for it to have materially more
parameters than BC-RNN — that difference is precisely what the compute-cost benchmark exists to
quantify, not something to minimize.

### 2.5 Checkpoint and fingerprint

Same handoff discipline as BC-RNN: SHA-256, library versions, full config, and this file's D1–D3
decisions (inherited unchanged from the BC-RNN run) recorded in the saved config.

---

## PART 3 — Escalation threshold (compute if not already done)

If `RoboTurk_Benchmark_and_Arbitration.md` Part C was already run, load and reuse its
`q̂_k,esc` values. Otherwise, compute them now:

- Using the same 79 calibration episodes and BC-RNN's phase scores (unchanged by ACT's existence),
  compute `q̂_k,esc` per phase (`closed`, `open`) at `α_esc = 0.30` as the primary operating point
  (justified by the earlier simulation-side finding that this budget gave a good efficiency/
  detection balance) — report `α_esc ∈ {0.20, 0.40}` as secondary comparison points if time allows.
- **Confirm `q̂_k,esc < q̂_k` in both phases.** If this fails, stop — the arbitration guarantee
  requires it and the benchmark in Part 5 would be invalid without it.

---

## PART 4 — Sanity verification before the full benchmark

Do not proceed to Part 5 until all of the following pass. Report each explicitly.

- **ACT checkpoint sanity:** training and validation loss both decreased monotonically or
  plateaued sensibly (no divergence, no NaN). Report the final values, same format as BC-RNN's
  loss table.
- **Dual-threshold check:** `q̂_closed,esc < q̂_closed` and `q̂_open,esc < q̂_open`, confirmed
  numerically (Part 3).
- **Input parity check:** confirm both policies were trained and are evaluated on identical
  (state, action) pairs — same episodes, same timesteps, same normalization statistics. Any
  mismatch here would silently invalidate the fidelity comparison.
- **Compute-measurement environment check:** confirm both policies' inference timing (Part 5.3)
  is measured on the same hardware, in the same session, back-to-back — not measured at different
  times or on different machines, which would make the comparison noise rather than signal.

If any check fails, stop and report which one, rather than proceeding to a benchmark whose
premises don't hold.

---

## PART 5 — The three-way benchmark

Run on the same 80 held-out evaluation episodes used throughout.

### 5.1 Fidelity — divergence from human teleoperation

For each of the three configurations, compute the predicted-vs-demonstrated score
`s(t) = ‖a_pred(t) − a_demo(t)‖₂` across every evaluation timestep:

- **BC-RNN alone** — BC-RNN predicts every timestep.
- **ACT alone** — ACT predicts every timestep.
- **Arbitrated** — BC-RNN predicts by default; at each timestep, if BC-RNN's own score would
  exceed `q̂_k,esc` for the current phase, switch to ACT's prediction for that timestep instead
  (replay `cp_arbiter.py`'s logic exactly as specified previously, now actually invoking ACT's
  output rather than only logging the decision).

Report, per configuration: mean score, per-phase (`closed`/`open`) mean and distribution, and the
same coverage/feasibility framing used in Part D and the prior benchmark file, so this table sits
consistently alongside the earlier ones.

### 5.2 Confirm H1 and H2 numerically

- **H1:** Arbitrated mean score vs. BC-RNN-alone mean score — report the difference and its
  direction explicitly.
- **H2:** Arbitrated mean score vs. ACT-alone mean score — report how close, not whether
  arbitration "wins."

### 5.3 Compute efficiency — the primary motivation, measured properly

- **Inference cost per policy, in isolation:** wall-clock time per forward pass for BC-RNN and for
  ACT, measured on the same hardware, same batch size, averaged over enough calls to be stable
  (hundreds, not a handful). Report mean and standard deviation.
- **Parameter count and, if straightforward to obtain, FLOPs per forward pass** for both policies
  — a hardware-independent complement to the wall-clock number, useful since wall-clock alone can
  be noisy or hardware-specific.
- **Fraction of evaluation timesteps on the expensive arm** under arbitration (from Part 5.1's
  replay) — this is the number that, combined with the per-policy costs above, gives the
  arbitrated system's actual average compute cost:

  ```
  cost_arbitrated ≈ frac_cheap × cost_BC-RNN + frac_expensive × cost_ACT
  ```

  Report this computed estimate **and**, if feasible, a direct measurement (actually timing the
  full arbitrated replay end to end and dividing by total steps) as a check that the estimate and
  the measurement agree.
- **The headline efficiency number:** `cost_ACT_alone / cost_arbitrated` — how many times cheaper
  arbitration is than always running the expensive policy.

### 5.4 The composite metric

Report **fidelity per unit compute** for all three configurations — e.g. mean score divided by
mean per-step cost, or an equivalent normalized form — and state which configuration is best on
this composite measure. This is the metric that can legitimately support an "arbitration wins
outright" claim, if the data supports it.

---

## PART 6 — Consolidated report

One document covering:

1. Part 0's artifact-status table.
2. ACT training summary (architecture, parameter count, loss curves).
3. The escalation thresholds and the dual-threshold check.
4. Part 4's sanity checks, explicitly pass/fail.
5. The three-way fidelity table (5.1–5.2) with H1/H2 stated as confirmed or not.
6. The compute-efficiency table (5.3) with the headline efficiency number.
7. The composite metric (5.4).
8. A short paragraph, reusable in the paper or the panel presentation, stating the result in the
   corrected framing: arbitration versus BC-RNN alone, arbitration versus ACT alone, and why the
   compute-efficiency number is the one that matters most — worded so it cannot be read as a claim
   that arbitration out-predicts ACT.

---

## Standing rules

- **Do not claim or imply arbitration beats ACT-alone on fidelity.** Report H1–H3 and the
  composite metric as specified; do not reframe a near-parity result as a win on the wrong axis.
- **No detection rate, false-alarm rate, or "catches failures" language anywhere in this file's
  output** — same scope discipline as the prior benchmark file; nothing here involves real
  failures.
- **Do not pool or compare these numbers against the LIBERO/simulation results** — different
  action space, different score, different hardware. Report side by side only.
- **Measure compute cost on the same hardware, back-to-back, for both policies** — a mismatched
  measurement environment would invalidate the entire efficiency claim.
- **A failed or ambiguous check is a result.** If H1 or H2 doesn't hold, or the composite metric
  doesn't favor arbitration, report that plainly.
