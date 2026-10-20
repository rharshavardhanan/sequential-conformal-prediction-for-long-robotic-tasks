# Design — live experiment viewer for the sequential-CP safety monitor

**Date:** 2026-09-09
**Status:** design, awaiting review
**Supersedes for demo purposes:** nothing. Extends `docs/status.md` §4 (arbitration) and §6 (results).

## 1. Purpose

A single-screen instrument that shows the calibrated monitor deciding, step by step, on a
LIBERO-Long rollout — such that a viewer who has not read the paper can follow the entire
decision chain in a few seconds, and a viewer who *has* read it can check every number
against the methodology.

It is an **experiment viewer, not a dashboard**. Every displayed quantity is one the
algorithm actually consumes. Nothing is invented for legibility.

## 2. Reality check — what this repo actually has

The demo brief assumed a `scp/` package (`scoring.py`, `phases.py`, `calibrate.py`).
That package is on the other half of the project and is **not in this repo**. What is here:

| brief assumed | this repo has |
|---|---|
| `scp/phases.py` | `scripts/subphases.py:segment()` |
| `scp/scoring.py` | `err_norm` precomputed per step into the HDF5 by `collect_rollouts.py` |
| `scp/calibrate.py` | `scripts/cp_monitor.py` |
| `cp_arbiter.py` | nothing — no arbiter code in this repo |
| trained BC-RNN / ACT | neither. `scripts/train_bc_rnn.py` exists; training not finished. ACT not started |

Every frame is already on disk: each rollout group carries `images/agentview` and
`images/eye_in_hand` at 128x128 uint8, 150 rollouts each for tasks 0, 1, 8. A replay demo
therefore needs no simulator at all.

**The brief's reference thresholds do not reproduce here.** At the pre-registered settings
(alpha=0.10, cal_frac=0.80, Bonferroni, agg `max`, task 0), `cp_monitor.py` gives
reach 1.230 / grasp 1.321 / transport 1.072 / release 1.922, against the brief's
1.40 / 1.32 / 1.29 / 1.68. Only `grasp` agrees. This is the same shape as the open
discrepancy in `status.md` §8 item 1 and is **not resolved here**. The demo displays what
this repo computes, and prints the command that produced it.

## 3. Three honesty rulings

These follow the project's own rule: *the interface must never show a quantity that is not
part of the experiment.*

**3.1 The policy row shows what generated the trajectory.** All 450 rollouts came from the
frozen `DiffusionPolicy` checkpoint `3105133c...`. The viewer displays that, with its
SHA-256. It does not display `BC-RNN` or `ACT`, neither of which exists yet.

**3.2 Escalation is drawn as a counterfactual until a second policy arm exists.** `q_esc` is
computable today — it is a second conformal quantile from the same calibration set at a
larger budget `alpha_esc`, needing no second policy. So the escalation *threshold* and the
*crossing event* are real and are shown. What is not real is a policy switch: the recorded
trajectory continued under DiffusionPolicy regardless. The event is therefore labelled
`would escalate (counterfactual — single-arm rollout)`. When BC-RNN and ACT exist, the same
panel becomes live with no change to the viewer.

**3.3 An arbitrated trajectory is never labelled "CP-certified safe".** Per `status.md` §4,
only the bounded spurious-escalation rate transfers to the arbitrated system, not the
coverage guarantee. The run summary reports escalation rate against its budget, never
coverage.

## 4. Architecture

The UI is a viewer of the experiment, not part of it.

```
  frame source                experiment core                 viewer
  ------------                ---------------                 ------
  replay: HDF5      -->   SafetyMonitor.step(...)   -->   matplotlib figure
  live:   MuJoCo    -->   returns a StatePacket     -->   (renders packets only)
          + policy
```

**`scripts/monitor.py` (new)** — `SafetyMonitor`, the only place monitor logic exists.
Holds the calibration artefact read-only. `step(action, eef_before, eef_after)` returns a
`StatePacket`. No file I/O, no rendering, no simulator import.

**`scripts/demo_monitor.py` (new)** — frame sources and the viewer. A generator yields
`(frame, action, eef_before, eef_after)`; `--replay <hdf5> -r N` reads it from
`images/agentview`, `--live --task-id T --init N` steps `OffScreenRenderEnv` +
`DiffusionPolicyRunner`. Downstream rendering is identical; a banner reads `[REPLAY]` or
`[LIVE]`.

**`scripts/cp_monitor.py` (edit)** — extended to emit the full artefact and to compute the
escalation and dwell constants. Not replaced.

### 4.1 The StatePacket

Exactly the quantities the algorithm consumes:

```
t                int      timestep
phase            str      reach | grasp | transport | release
score            float    err_norm = ||planned - executed|| / ||cmd||
cmd_norm         float    ||action[:3] * 0.05||, metres
scorable         bool     cmd_norm >= motion_floor_abs
q_safe           float    threshold in force for this phase
q_esc            float    escalation threshold in force for this phase
dwell            int      steps in the current phase segment
dwell_limit      float    d_hat for this phase
violated         bool     latched: score exceeded q_safe on a scorable step
escalated        bool     latched: score exceeded q_esc on a scorable step
stalled          bool     latched: dwell exceeded d_hat for this phase
verdict          str      display state = highest severity latched, in the order
                          VIOLATION > STUCK_TIMEOUT > ESCALATE > SAFE
event            str|None one-line log entry, emitted only when a latch flips
```

Latches are independent and never clear within an episode: a rollout that escalates and
later violates keeps both flags, and `verdict` reads VIOLATION. `event` fires once per latch,
so the log records the step each mechanism first triggered rather than repeating every step.

`planned = eef_before + action[:3] * 0.05`, `executed = eef_after` — identical to
`collect_rollouts.py`. The 0.05 factor is the OSC output range; omitting it is wrong by 20x.

## 5. Causal phase labelling — a fixed 5-step lag

`subphases.segment()` relabels indices `t-4 .. t+5` around a transition at diff-index `t`,
and `t` is only observable at time `t+1` (it compares `closed[t+1]` against `closed[t]`).
Finalising the label for index `i` therefore requires observations through `i+5`. A live
monitor cannot see that coming. Rather than invent a forward-only rule producing labels
calibration never saw, the monitor emits its decision for step `t-5` at time `t`.

- Cost: 250 ms decision latency at 20 Hz, against a median lead time of 438 steps.
- Benefit: labels are provably identical to those the thresholds were calibrated on, which
  is what keeps the guarantee attached to the displayed threshold.

### 5.1 Mechanism, and a trap inside it

The backward half of the relabel window (`d-4 .. d`) is applied in place to steps already
buffered. The forward half (`d+1 .. d+5`) lands on steps **that do not exist yet**, so it
must be carried as an open window and applied on append. A prototype that only applied the
in-place half mismatched on 149 of 150 rollouts. This is the single trap in the whole
component and the test exists to catch it.

**Verified on the prototype:** all 150 task-0 rollouts produce labels byte-identical to
`subphases.segment()`, with a constant observed lag of exactly 5. The final 5 steps of an
episode are flushed at the end — unavoidable, and they are the last 5 of up to 600, long
after any alarm would have fired.

The equality is a test, not a claim (§9).

## 6. The calibration artefact

Emitted by `cp_monitor.py --out`, loaded read-only by the viewer, never recomputed at demo
time. Contents: the four `q_safe`, the four `q_esc`, the four `d_hat`, plus `alpha`,
`alpha_esc`, `beta`, `cal_frac`, `agg`, `phase_window`, `motion_floor_abs`, `score`,
`policy_sha256`, and the exact command that produced it.

**Reference values, task 0, alpha=0.10, alpha_esc=0.30, beta=0.10, cal_frac=0.80, Bonferroni,
agg max, seed 0:**

| phase | q_esc | q_safe | d_hat (steps) |
|---|---|---|---|
| reach | 1.1387 | 1.2303 | 128 |
| grasp | 1.1850 | 1.3213 | 10 |
| transport | 1.0369 | 1.0721 | 80 |
| release | 1.4575 | 1.9220 | 10 |

### 6.1 Stall detection — calibrated like everything else

The brief's `d_hat` = "(1-beta) quantile of phase durations" was tested and **fails**: taken
over individual segments at beta=0.10 it fires on 47% of successful rollouts. Cause: a
rollout contains several segments per phase, so the per-segment quantile is not a per-rollout
bound.

Fixed by calibrating dwell exactly as deviation is calibrated — the **per-rollout maximum**
dwell per phase, conformal quantile at `beta_k = beta/K`. Measured on task 0: fires on
1/21 held-out successes (0.048, budget 0.10) and 44/47 failures; `rollout_0076` fires.

**Two limitations stated on screen, not in a footnote:**

1. `grasp` and `release` are defined as fixed +/-5-step windows, so their dwell is
   structurally always 10 steps. **Those two phases can never fire STUCK_TIMEOUT.** Dwell
   detection is meaningful only for `reach` and `transport`.
2. Every failure in this dataset ran the full 600 steps, so *any* duration feature separates
   the classes near-perfectly. The dwell detector's value is that it fires around step 129,
   not that it scores 0.936.

### 6.2 The escalation band

`q_esc < q_safe` per phase by construction (`alpha_esc > alpha`). Measured spurious-escalation
rate on held-out nominal tracks its budget: 0.238 at alpha_esc=0.20, 0.333 at 0.30, 0.381 at
0.40.

**Measured caveat:** on task 0, escalation fires on the *same timestep* as violation in 21/21
paired cases. The second threshold buys failure coverage (0.702 vs 0.447), **not lead time**.
The gap is 3-5% of the threshold for reach/grasp/transport. Two dashed lines would visually
coincide, so the region between them is **shaded as a band** — the plot reads as three
regions: below the band = SAFE, inside = ESCALATE, above = VIOLATION.

### 6.3 Motion mask applied at alarm time

`cp_monitor.first_alarm()` currently ignores the motion mask that calibration applies. That
asymmetry lets a low-motion step (mean err_norm 0.99, max 2.78 — the extreme tail) raise an
alarm calibration never saw. The monitor masks at alarm time, and `first_alarm` is fixed to
match. **This changes numbers this repo has published**: on the seed-0 split, FAR 0.238 ->
0.190 and TPR 0.617 -> 0.447. The change is recorded in the plan's done-when so it is not
silent.

## 7. The screen

Single window, four regions. Fixed camera — never orbiting — so runs are visually comparable.

```
+----------------------------------------------------------------------+
| Sequential CP safety monitor      [REPLAY] task 0 - r0083   184/600   |
+-------------------------------+--------------------------------------+
|            TRANSPORT          | CURRENT STATE                        |
|                               |   phase      TRANSPORT               |
|         agentview             |   policy     DiffusionPolicy 3105133c|
|      (128^2, upscaled)        |   monitor    SAFE                    |
|                               |   score      0.71                    |
|                               |   q_esc      1.037                   |
|                               |   q_safe     1.072                   |
|                               |   0.71 < 1.037 < 1.072  -> SAFE      |
|                               +--------------------------------------+
|                               | TRACKING                             |
|                               |   commanded  |========>              |
|                               |   actual     |===>                   |
|                               |   residual   0.71                    |
|                               +--------------------------------------+
|                               | MOTION                               |
|                               |   |##############....|  MOVING       |
|                               |   dwell  44 / 80 steps               |
+-------------------------------+--------------------------------------+
| LIVE SIGNAL   s(t) solid - escalation band shaded - phase bands       |
|               masked steps greyed - violation latched with a red rule |
+----------------------------------------------------------------------+
| EVENT LOG   184  TRANSPORT  SAFE   s=0.71                             |
+----------------------------------------------------------------------+
```

The phase label sits directly above the robot, so the core idea — *the same robot judged
under different calibrated thresholds* — is visible without reading the side panel.

The per-phase threshold panel (all four `q_safe` with the current phase marked) answers
"what are the other thresholds?" without anyone asking.

## 8. Run summary

Printed and rendered when the episode ends, in terms of quantities that exist:

```
result              FAIL (600 steps, 30.0 s sim time)
verdict             VIOLATION at step 40   (560 steps before episode end)
first escalation    step 40   [counterfactual - single-arm rollout]
steps above q_esc   18.4%
steps above q_safe  11.2%
steps masked        0
max err_norm        1.31   (transport, step 213)
stall               not triggered
policy              DiffusionPolicy 3105133c (single arm - no arbitration in this rollout)
```

Expensive-policy fraction and switch count are **omitted, not zeroed** — there is no second
arm to measure. They appear when one exists.

## 9. Tests

`tests/test_monitor.py`, run with `.venv-lerobot/bin/python -m pytest`:

1. **Causal labels equal offline labels.** For all 150 task-0 rollouts, the streaming label
   emitted for index `i` equals `subphases.segment()[i]` exactly, for every `i` the monitor
   finalises. This is what makes the displayed threshold the calibrated one. The test also
   asserts the lag is exactly 5 — that the monitor finalises index `i` no earlier than time
   `i+5` and no later.
2. **Streaming score equals the recorded score.** For every step of a rollout, the monitor's
   `err_norm` matches the stored `err_norm` to 1e-9. This is what makes it the same
   experiment.
3. **Streaming verdict equals the batch verdict.** Per rollout, the *step index* at which
   the streaming monitor first latches VIOLATION equals the batch `first_alarm` computed from
   the same thresholds with masking applied. The indices must match exactly; the streaming
   monitor merely reports that index 5 steps later in wall time, which the test asserts
   separately rather than conflating with the index itself.

Together these are what make it a monitor rather than a picture of one.

## 10. Demo cases, by index

| case | rollout | why |
|---|---|---|
| success | `rollout_0041` | 267 steps, held-out nominal, stays SAFE throughout |
| failure | `rollout_0083` | VIOLATION at step 40 of 600 — a 560-step lead |
| stall | `rollout_0076` | gripper never closes; `reach` only; STUCK_TIMEOUT |

`--list` prints candidates with their alarm step so cases are chosen by index, not by hope.

## 11. Non-goals

- Real policy arbitration (no second arm exists). The lane is built; the switch is not faked.
- GPIO / Jetson output.
- Flat-CP side-by-side (brief §5.4) — needs a second calibration arm this repo lacks. Add
  after the main screen works.
- Resolving the threshold discrepancy with the `scp/` implementation (§2). Out of scope,
  logged in `status.md` §8 item 1.
- Any change to `collect_rollouts.py`, `dp_policy.py`, or `fingerprint.py`.

## 12. Risks

**Committing changes the fingerprint.** `fingerprint.compare()` diffs every key including
`git_revision`, and the three rollout files store `194426ea...-dirty`. Committing moves HEAD,
so any future *resume* of `task{0,1,8}_v10.hdf5` would abort. Those three are 150/150
complete, so nothing is blocked today, and wave-2 writes a new file with a new fingerprint.
Stated so it is not discovered later.

**No new dependency.** matplotlib 3.10.9 and its `macosx` backend are already in
`.venv-lerobot`. PySide6 + pyqtgraph would be a ~200 MB Qt install into an environment
deliberately frozen against `requirements-lerobot.txt`; the state-packet boundary makes
swapping the viewer a UI-only change if refresh rate ever demands it.

**Live mode is slow and stochastic.** ~217 ms/step on MPS is ~4.4x slower than the 20 Hz
control rate, and a rollout that failed during collection may not fail again. Replay is the
demo you rely on; live is the proof it is not a video.
