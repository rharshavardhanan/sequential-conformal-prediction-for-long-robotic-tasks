# RoboTurk BC-RNN — resumed from Part 3 (D1–D3 resolved): results

**Executes:** `RoboTurk_Resume_Training.md`.
**Deliverable:** [`RoboTurk_BCRNN_Training.ipynb`](RoboTurk_BCRNN_Training.ipynb) — Colab-ready,
no MuJoCo/robosuite/LIBERO.
**Execution status:** run **end-to-end against the real full release** — the aligned HDF5
`SawyerTowerCreation_aligned_dataset.hdf5` (1.12 GB, 551 demos, 43 users), obtained by
**selective ranged extraction from the 23 GB zip** (no video download). Parts A–D all
executed; a state-only BC-RNN was trained on GPU and a first predicted-vs-demonstrated
calibration computed.

Parts 1–2 (schema verification, phase model) are carried over from the prior pass and not
re-run except as noted (Part B re-runs the filter/segmentation on all 531 usable episodes).

---

## Decisions, as recorded in the saved training config (`fingerprint.json`)

- **D1 — action = `user_control[:,0:7]` (IK joint targets) + `user_control[:,20]` (grasp bool)**,
  an 8-dim vector. Native recorded fields; **no reconstruction, no `eef_poses`
  finite-differencing, no robosuite.** This is **joint-target space** — a different space
  from the LeRobot mirror (delta-EEF) and from the Franka/LIBERO sim side (delta-position
  OSC). No cross-source comparison is made; the Part D score is self-contained
  (predicted vs demonstrated, both in this space).
- **D2 — state-only.** Input = `eef_poses`(7) + `joint_states_arm`(27) +
  `joint_states_gripper`(3) = **37-dim** proprioception. No visual encoder, no video decode.
- **D3 — resample to 20 Hz.** See Part A: the aligned file is *not* on a fixed-rate grid.

---

## Part A — D3 verification (ran on the real aligned file)

- The aligned HDF5 **does** cross-sync its streams: `eef_times`, `joint_states_arm_times`,
  `joint_states_gripper_times`, `user_control_times` are **identical arrays** per demo.
- **But those `*_times` datasets are a miscalibrated clock** — they equal the true time
  scaled by **≈ 1/40.66**. The trustworthy timeline is **`user_control[:,21]` (UTC seconds)**.
- On the UTC clock the native cadence is **~20 Hz** (median dt 0.0501 s) — already this
  project's Franka/LIBERO control frequency — **but not a clean grid**: per-demo dt CV is
  large (median ≈ 3) with many multi-hundred-ms pause-gaps (teleop pauses).
- **Finding: NOT aligned-as-shipped in time → resample.** Every proprioceptive stream and
  the D1 action fields are resampled onto a **fixed 20 Hz grid on the UTC clock**:
  **linear** interpolation for continuous fields (`eef_poses`, `joint_states_arm`,
  `joint_states_gripper`, `user_control[:,0:7]`), **nearest-sample** for the discrete
  `user_control[:,20]` grasp bool (verified to stay in {0,1} after resampling).

---

## Part B — filter + K=2 segmentation on the full release (531 demos)

| check | full release |
|---|---|
| total demos | 551 |
| length == 1 | 0 |
| length < 10 samples | 0 |
| **zero grasp transitions** | **20** → dropped |
| **usable** | **531** |
| segment-count / episode | min 2, p25 11, **median 16**, p75 21, p90 27, max 46 |
| grasp-cycle count / episode | min 1, p25 5, **median 8**, p75 10, p90 13, max 22 |
| **K=2 pooled-window empty-phase rate (after filter)** | **0 / 531 = 0.00%** |

The mini-set numbers (median 8 segments / 4 cycles) understated it — at full scale the
multi-cycle structure is even more pronounced (median 8 grasp cycles, matching the LeRobot
pass's ~7). Every usable episode has both a `closed` and an `open` pooled window, so K=2 is
non-degenerate for all 531 — confirmed, not assumed.

> Note: an earlier proxy (`n_close==0 or n_open==0`) reported 2.07%; that over-counts
> episodes that start closed and only ever open. The correct pooled-window check
> (`(gb==1).any() and (gb==0).any()`) is **0%**.

---

## Part C — state-only BC-RNN

- **Data:** 531 episodes → resampled to 20 Hz → **2,652,847 timesteps**, state 37-d,
  action 8-d. Median episode ≈ 5,885 steps (~294 s).
- **Splits (physically separate, per the standing rule):**
  policy **train 316 / val 56 episodes**; Part D **calibration 79 / evaluation 80 episodes**
  (the 30% hold-out, never seen in training). 48,415 train windows (len 64, stride 32).
- **Architecture:** state-only BC-RNN — MLP encoder `37→256→256`, LSTM `256×2`, **5-mode
  GMM** head over the 8-d action. No vision layers (D2). Standardised state; action
  standardised except the grasp bool kept in {0,1}.
- **Training:** 15 epochs, Adam 3e-4, batch 128, GMM NLL, grad-clip 1.0, checkpoint every
  epoch. GPU: RTX 3050 6 GB.
- **Loss (executed):**

| | epoch 1 | epoch 5 | epoch 10 | epoch 15 |
|---|---|---|---|---|
| train NLL | −0.43 | −9.87 | −11.84 | **−12.53** |
| val NLL | −6.21 | −10.36 | −12.18 | **−13.02** |

  Monotone decreasing, val tracks train (no overfitting). Negative NLL is expected — a
  continuous GMM density can exceed 1.
- **Fingerprint (`fingerprint.json`):** SHA-256 of the checkpoint, `torch 2.6.0+cu118`,
  `numpy 2.2.3`, Windows-11, RTX 3050; full config incl. D1–D3 text. **Frozen once
  fingerprinted — no further training against this hash.**

---

## Part D — predicted-vs-demonstrated calibration

Score `s(t) = ‖a_pred(t) − a_demo(t)‖₂`, both in the **D1 joint-target action space**.
**This is not the simulation side's commanded-vs-executed score** and is not presented as
the same quantity. K=2 pooled categories (`closed`/`open` via `user_control[:,20]`),
`α = 0.10`, equal split `α_k = 0.05`, `scp/calibrate.py` order-statistic threshold,
per-episode aggregate = `max` (matches `scp/scoring.py`).

| category | n_cal | rank | feasible (`n ≥ 19`) | q̂_k | n_eval | eval coverage (target 0.90) |
|---|---|---|---|---|---|---|
| closed | 79 | 76 | **yes** | **2.284** | 80 | **1.000** |
| open | 79 | 76 | **yes** | **2.211** | 80 | **1.000** |
| **joint** (both within threshold) | | | | | 80 | **1.000** |

- **Feasibility bound cleared comfortably** — 79 calibration episodes per category vs the
  mini set's ≤ 8, and vs `n_min = 19` (`calibrate.py`) / the LIBERO milestones' `n ≥ 79`.
- **Coverage is 1.000, above the nominal 0.90.** Split-conformal guarantees only
  `≥ 1 − α`, so this is valid but **conservative**: with n_cal = 79 the threshold is the
  76th-largest calibration score, and all 80 nominal eval episodes fall under it. The
  per-window `max` aggregator over ~5,000-step episodes makes q̂ a *worst-single-timestep*
  joint-target error (~2.2 rad) that nominal demos stay below by construction.
- **Interpretation:** the machinery is wired correctly for this predicted-vs-demonstrated
  score. It makes **no** claim about detecting real failures or OOD behaviour — none are
  present. All demos are treated as "successes" (no failure labels exist in RoboTurk).
  Per D2, if the score proves uninformative in a way vision would fix, that is a
  **stretch upgrade**, not a default.

---

## Part 5.3 — compute efficiency (measured 2026-09-12)

Full detail, charts and regeneration commands: [`benchmark/README.md`](benchmark/README.md).
BC-RNN 1,150,037 params vs ACT v2 9,805,128 (`roboturk_act_best.pt`, chunk 16, history 20).

| per control step | CPU | MPS |
|---|---|---|
| BC-RNN | 0.059 ms | **0.185 ms** |
| ACT v2, chunked ÷16 | 0.132 ms | **0.124 ms** |
| arbitrated @ 11.16% escalation | 0.073 ms | 0.199 ms |
| **ACT-alone ÷ arbitrated** | **1.80×** | **0.62×** |

**The headline is device-dependent and on MPS the cheap/expensive framing inverts.** On CPU
arbitration is 1.80–2.25× cheaper than running ACT everywhere (H3 holds). On MPS chunked ACT
is cheaper *per step* than BC-RNN, so arbitration costs **more** than ACT-alone at every
escalation fraction — H3 fails there, and no choice of `α_esc` fixes it.

BC-RNN is a 2-layer LSTM stepped one frame at a time: sequential, so on GPU its cost is
dominated by per-step kernel-launch overhead it cannot amortise, while ACT emits 16 actions
from one parallel forward. **Action chunking, not parameter count, decides per-step cost** —
ACT has 8.5× more parameters and is still cheaper per step on MPS. Arbitration cannot go
below BC-RNN-alone either, because the monitor needs BC-RNN's score on every step, so
arbitrated cost is `bc + f·act`.

**Chunking caveat.** ACT is priced two ways and they differ by up to 16×: *chunked* (1/16
forward per step, how it would be deployed — the headline) and *per-step* (one full forward
per step, what `scripts/roboturk_arena.py` does so it can show ACT's opinion every step).
Never cite the per-step figure without naming it as an upper bound; doing so overstates the
expensive arm 16× and flatters arbitration by the same factor.

Timing is **throughput** (sync once per run), not per-call latency. On MPS a per-call
`synchronize()` adds a fixed ~0.20 ms to BC-RNN — roughly half its apparent per-call time —
that a real pipelined loop never pays. The estimate-vs-direct cross-check caught this: the
additive cost model over-predicted a measured arbitrated replay by 39% when fed per-call
numbers, and predicts it to 0.5% when fed throughput. The model was right; the measurement
method was wrong.

These numbers are Apple-Silicon-specific; the MPS inversion is a property of how a sequential
LSTM maps onto this GPU and is not a general claim about BC-RNN vs ACT.

---

## Out of scope (unchanged): genuine failure detection.

## Artifacts (in `_tmp_roboturk/`)

- `SawyerTowerCreation_aligned_dataset.hdf5` (1.12 GB) — selectively extracted, no videos
- `roboturk_20hz.npz` — the 20 Hz resampled cache (317 MB)
- `roboturk_bcrnn_state_only.pt` — the checkpoint
- `fingerprint.json`, `partD.json` — config + calibration results
- [`RoboTurk_BCRNN_Training.ipynb`](RoboTurk_BCRNN_Training.ipynb) — executed end-to-end
