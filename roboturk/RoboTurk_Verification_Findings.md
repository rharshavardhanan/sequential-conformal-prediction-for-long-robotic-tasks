# RoboTurk Pre-Commit Verification — Findings

**Scope:** inspection only. No policy trained, no training setup started, no MuJoCo /
robosuite / LIBERO installed. Dataset: `lerobot/roboturk` (HuggingFace), LeRobot v3.0 format.
**Target task:** Tower Creation (`create tower`).

**Notebook:** [`RoboTurk_Verification.ipynb`](RoboTurk_Verification.ipynb) — Colab-ready.
**Execution status: RUN END-TO-END AGAINST THE REAL DATASET.** All 24 cells executed with
no errors on 2026-09-09 (Python 3.13, local). The low-dimensional data for the whole
dataset is only ~100 MB of Parquet, so the run used the complete dataset (187,507 frames /
1,995 episodes), not a sample. The only skipped step is the *optional* video-frame pull in
2.6 (needs an AV1 decoder; `pip install av` enables it in Colab) — the state/action
sanity check stands in for it and is sufficient for go/no-go.

> Implementation note: the notebook reads the Parquet directly via `huggingface_hub` +
> `pandas`/`pyarrow` rather than `datasets.load_dataset`, which does not cleanly open
> LeRobot v3.0 datasets with video features. No functionality is lost — every low-dim
> stream is in the Parquet.

---

## Part 3 acceptance table

| Question | Reported result |
|---|---|
| **Per-task episode count** | `layout laundry` **872**, `object search` **594**, `create tower` **529**. Total 1,995. Tower Creation is **529**, not the assumed ~650–700 — **materially lower; use 529.** |
| **Gripper dimension** | **action dim 6** (last dim). Values are **exactly `{0, 1}`** — a **discrete binary** command. Not a graduated `[-1, 1]` bang-bang, not a continuous position. `extreme-mass = 1.00`, 2 distinct values. Dims 0–5 are all continuous (570–948 distinct values each, zero extreme-mass). `observation.state` has **no** gripper channel (see below). Plot: `gripper_channels.png` / cell 2.4. |
| **Delta vs. absolute action** | **Delta (relative).** The intended residual test `s(t+1) ≈ s(t)+a(t)` vs `s(t+1) ≈ a(t)` is **not constructible** here — `observation.state` is identically zero, so both residuals collapse to `\|0 − …\| = 0.1125` and carry no information. Fallback evidence, all consistent with delta: action dims are zero-mean with `\|mean\|/std ≤ 0.03` (a[0]:µ=+0.0013 σ=0.051; a[3]:µ=+0.0015 σ=0.092; etc.); ranges are small (±0.5 for 0–2, ±1.8 rad for 3–5); the per-episode cumulative sum of a[0:3] stays bounded (~0.2–0.4, e.g. ep 7: `[0.231, 0.405, 0.225]` over 156 steps) — i.e. position *increments*, not workspace-absolute targets (which would sit at x≈0.5, z≈1.0). |
| **Empty-phase rate** | **30 / 529 = 5.7%** of Tower Creation episodes have no gripper-close **or** no gripper-open transition → at least one empty phase category. **Non-zero — FLAG.** Of these, 18 have *zero* gripper transitions at all; 2 are degenerate length-1 episodes. |
| **Segment-count consistency** | Gripper-transition segments per episode: **range 1–43, median 15, IQR [11, 19], mean 15.1, std 7.4, CV 0.49.** Grasp (0→1) events per episode: median 7, max 21. This is **broad — not LIBERO-like.** A fixed small set of phase categories is not consistently populated across episodes. |

---

## Open questions from Part 0

**1. What does each action / state dimension represent?**
The metadata `names` are placeholders (`motor_0…motor_7`, `motor_0…motor_6`) — **no semantic
map is shipped.** Inferred empirically:

- **`action` (7-dim):** `0–2` = end-effector **position deltas** (~metres, σ≈0.05, range ±0.5);
  `3–5` = end-effector **orientation deltas** (axis-angle, σ≈0.09, range ±1.8 rad);
  `6` = **binary gripper** `{0,1}`. This matches a robosuite OSC/IK delta controller, but was
  established from the data distribution, **not** assumed from the Franka/LIBERO layout.
- **`observation.state` (8-dim): identically zero for every dimension of every frame**
  (confirmed both from the full 187,507-frame table and from `meta/stats.json`, where
  min = max = mean = std = 0 for all 8 dims). The 8-dim state is a **placeholder — it
  contains no proprioception.**

**2. Which dim is the gripper; discrete or continuous?**
Action dim 6, **discrete binary `{0,1}`**. Overall 57.1% of frames are at `1`; the gripper
sits at `1` at t=0 in 519/529 Tower episodes, so `1` is the rest/initial state. Whether
`0`/`1` maps to open/close cannot be pinned down without video, but this does not affect
segmentation (transitions are what matter).

**3. Delta or absolute action?**
**Delta.** See the acceptance table — note the intended metadata-based test is impossible
on this dataset because state is zero, so this rests on distribution shape and the
bounded-integration argument. This should be treated as **high-confidence but not
metadata-verified.**

**4. How many Tower Creation episodes?**
**529** (26.5% of the dataset). Not ~650–700.

**5. Does gripper-transition segmentation give a sensible, zero-empty-phase structure?**
Partly. Transitions land at local minima of commanded translation speed (the arm slows to
actuate the gripper — see `transition_sanity.png`), so detected transitions correspond to
plausible pick/place events. **But:** the empty-phase rate is **5.7%, not 0%**, and the
LIBERO reach/grasp/transport/release 4-tuple **does not apply** — a Tower Creation episode
contains ~7 grasp cycles (median), range 0–21. The natural unit here is a **repeated
pick-and-place cycle**, and the number of cycles varies widely across episodes.

---

## Unanticipated issues

1. **`observation.state` is empty (all zeros).** Any nonconformity / "commanded vs achieved"
   score that assumes a populated proprioceptive state must be **redesigned**, or state must
   be sourced elsewhere (e.g. re-derived from video, or from the original RoboTurk release
   rather than the LeRobot mirror). This is the single biggest finding for downstream design.
2. **No reward / success labels.** `next.reward` is `0` for all 187,507 frames; `next.done`
   is `True` only on the last frame of each episode. There is no per-episode success signal
   in this dataset.
3. **Multi-cycle structure.** ~7 grasp cycles per Tower Creation episode ⇒ per-phase
   calibration in the LIBERO style needs a cycle-relative formulation (phase within cycle,
   variable cycle count), not a fixed absolute phase index.
4. **Degenerate episodes.** 2 Tower episodes have length 1; 8 are shorter than 10 frames;
   18 have no gripper motion at all. These need an explicit filter before any use.
5. **Strategy diversity is real.** Segment-count CV ≈ 0.49 confirms RoboTurk's documented
   property that Tower Creation demonstrations vary substantially in approach.

---

## Go / no-go

| Item | Verdict |
|---|---|
| Tower Creation is a usable target task with enough data | **Go** — 529 episodes, median 144 frames, coherent gripper-cycle structure |
| Action space is understood well enough to train BC-RNN/ACT | **Go** — 6 delta pose dims + 1 binary gripper; standard for BC |
| A "commanded vs achieved" / proprio-based nonconformity score as used for LIBERO | **No-go as-is** — state is empty; needs a different formulation or a different state source |
| LIBERO-style fixed 4-phase per-phase calibration | **No-go as-is** — 5.7% empty-phase rate, wide segment-count spread; needs a cycle-relative phase model |
| Delta-vs-absolute assumption carried from Franka/LIBERO | **Confirmed (delta)** but by distribution, not by the metadata residual test — re-verify if a proprio source is added |

**Stop point:** these findings are for review before any training setup, policy-architecture
work beyond the already-fixed BC-RNN/ACT choice, or further data collection. No such work
has been started.

---

## Artifacts

- [`RoboTurk_Verification.ipynb`](RoboTurk_Verification.ipynb) — executed, with embedded output and 3 figures
- `gripper_channels.png`, `phase_hist.png`, `transition_sanity.png` — also embedded in the notebook
