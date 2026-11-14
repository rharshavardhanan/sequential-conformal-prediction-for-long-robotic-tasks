# Design — compute-efficiency benchmark (Part 5.3) + live arena readout

**Date:** 2026-09-12
**Status:** implemented 2026-09-12
**Executes:** `roboturk/RoboTurk_ACT_and_Full_Benchmark.md` Part 5.3.

## 1. Purpose

Measure what the arbitration design actually claims: that running the cheap policy by
default and escalating to the expensive one only when the calibrated monitor says so costs
materially less than running the expensive policy everywhere.

Two deliverables, one number set:

- **Recorded** — an offline benchmark table + charts in `roboturk/benchmark/`.
- **Live** — a real-time compute readout in `scripts/roboturk_arena.py`.

Policies: **BC-RNN** (1.15 M, cheap arm), **ACT v2** (`roboturk_act_best.pt`, 9,805,128
params, chunk 16, history 20, epoch 11), and **arbitrated**. ACT v1 is out of scope.

## 2. The chunking decision — the one that can invalidate everything

ACT is an Action Chunking Transformer: one forward emits **16** actions. Its per-step cost
is therefore `cost_forward / 16` when used as designed.

`scripts/roboturk_arena.py:140` currently calls `act(...)[0][0, 0]` — a **full forward every
timestep, discarding 15 of the 16 actions**. That is correct for the arena's purpose (it
needs ACT's opinion at every step to draw the comparison) but it is *not* how ACT would be
deployed, and quoting it as ACT's cost overstates the expensive arm by up to **16×** —
which would flatter arbitration by the same factor.

Both regimes are reported:

| regime | ACT forwards per control step | status |
|---|---|---|
| **chunked** | 1/16 | **headline** — ACT as designed |
| **per-step** | 1 | stated upper bound — what the arena does today |

No efficiency claim may cite the per-step number without naming it as the upper bound.

## 3. What is measured

### 3.1 Per-policy latency (the only hard measurement)

Batch 1, `torch.no_grad()`, ≥500 timed calls after ≥50 warmup calls, mean and sd.

- **Back-to-back, one session, one machine.** Part 5.3 requires it; a mismatched measurement
  environment turns the comparison into noise. ACT training has finished, so the machine is
  quiet — this is the first point at which the measurement is valid.
- **Both devices.** MPS (what the arena runs) and CPU (what the scoring ran). They are
  reported separately and never mixed into one ratio.
- **MPS needs `torch.mps.synchronize()`** around the timed region. MPS dispatch is
  asynchronous; timing without a sync measures enqueue time, not compute, and yields
  implausibly fast numbers.
- Input shapes matched to real inference: BC-RNN `(1, 1, 37)` stepped with carried LSTM
  state; ACT `(1, 20, 37)`.

### 3.2 Arbitrated cost — estimate and direct measurement

Part 5.3 asks for both, as a cross-check:

```
estimate:  cost_arb(f) = (1-f)*cost_BCRNN + f*cost_ACT_per_step
direct:    time the full arbitrated replay end to end / total steps
```

where `f` is the fraction of steps on the expensive arm. If the two disagree by more than
measurement noise, the cost model is wrong and that is the finding — the prettier number is
not reported in preference to the other.

Note the cost model charges BC-RNN on **every** step, including escalated ones: the monitor
needs BC-RNN's score to decide whether to escalate, so the cheap arm always runs. Arbitration
adds ACT on top rather than replacing BC-RNN.

### 3.3 Escalation fraction — swept, not assumed

Three different triggers exist in the repo and they disagree by two orders of magnitude:

| source | trigger | escalation fraction |
|---|---|---|
| `partD_v2.json` `escalation.max.0.30` (what the arena loads) | score vs per-episode-max `q_esc` | **0.15%** |
| p75 cross-conformal (`cross_conformal.json`) | score vs p75 `q_esc` | **2.00%** |
| `partD_corrected.json` α=0.10 | BC-RNN predicted sigma | **11.16%** |

Picking one would bury the disagreement. So cost is reported as a **function of `f`** over a
sweep, with those three operating points marked. This is also what makes the cost chart
informative rather than a three-bar picture.

## 4. Live readout

Additive instrumentation in `scripts/roboturk_arena.py`. It must not touch the scoring, the
thresholds or the arbiter logic — only observe and display.

Shown: rolling per-step latency of each arm, which arm is currently active, cumulative
compute spent, and compute saved versus always-ACT.

The arena evaluates **both** arms every step to draw its comparison, so its own wall clock is
not the arbitrated cost. The readout therefore displays *attributed* cost — the cost the
arbitrated system would have incurred given the arm the arbiter selected — computed from the
measured per-call latencies. Displaying the arena's raw wall clock would be wrong and is
explicitly not done.

Also flagged to the reader: `--stride 4` skips 3 of every 4 frames for display. Cost is
reported per **control step**, never per displayed frame, or the numbers inflate 4×.

## 5. Output

New folder `roboturk/benchmark/`:

| file | content |
|---|---|
| `compute_benchmark.json` | every measured number, the cost sweep, the estimate-vs-direct check |
| `latency.png` | per-policy latency, mean ± sd, both devices |
| `cost_vs_escalation.png` | arbitrated cost vs `f`, with the three operating points marked |
| `efficiency_summary.png` | headline `cost_ACT / cost_arb` at each operating point |
| `README.md` | the table, the headline number, and the caveats |

## 6. Out of scope

- Retraining or re-fingerprinting anything. Both checkpoints are frozen.
- ACT v1.
- Changing the arena's thresholds, scoring or arbiter logic.
- FLOP counting — `fvcore`/`ptflops` are not installed and adding a dependency for a
  hardware-independent complement to a measurement we can take directly is not justified.
  Parameter counts serve that role.
- Re-litigating which escalation trigger is correct. The sweep reports all three; choosing
  between them is a separate decision.
