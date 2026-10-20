# RoboTurk ACT training — Part 2 of RoboTurk_ACT_and_Full_Benchmark.md

**Date:** 2026-09-10
**Status:** in progress

## Why (brainstorm — thin, and here is why)

The brainstorm output is thin on *what* because `RoboTurk_ACT_and_Full_Benchmark.md`
Part 2 already specifies data, split, input, output, architecture family and
checkpoint discipline, and the user has explicitly instructed not to vary any of
them. The one genuinely open question was *where to train*; that was settled by
measurement (below), not discussion.

The unstated constraint the request did not carry, surfaced here: **Part 5.3
requires both policies' inference timing to be measured back-to-back on one
machine.** That machine must be this Mac. It makes local training strictly
simpler than Colab regardless of raw throughput, and it is the reason the
where-to-train question has a right answer rather than a preference.

**Known-broken, deliberately not fixed here:** the escalation score is dominated
by the grasp bool (median 98.5% of the threshold-setting error), so `q_esc` fires
on 0.16% of timesteps. See `memory/roboturk-score-dominated-by-grasp-bool.md`.
This is a *scoring* defect, not a *policy* defect — it is fixed after training by
re-running Part D on the 7 continuous dims, with no ACT retrain. Training now
loses nothing.

## Measurements that drove the decision

| | value |
|---|---|
| ACT (state-only, d=512, 4 enc / 7 dec / 4 CVAE, chunk 16) | 72.5 M params (63x BC-RNN's 1.15 M) |
| train fwd+bwd, batch 128, MPS | 197 ms |
| same, CPU | 984 ms (5.0x slower) |
| inference, batch 1, MPS | 3.03 ms |
| train split | 316 episodes / 1,565,492 timesteps |
| val split | 56 episodes / 282,601 timesteps |
| stride 8 | 195,686 samples = 1,529 batches = 5.0 min/epoch -> **4.2 h for 50 epochs** |

## Frozen decisions (inherited, do not vary)

- Data: `roboturk_20hz.npz`, unmodified.
- Split: seed-0 permutation, 316/56 train/val, 79/80 cal/eval untouched.
- Input: 37-dim state only. **No images** — parity with BC-RNN is the point.
- Output: same 8-dim D1 action (`user_control[:,0:7]` + grasp bool).
- Normalisation: reuse the BC-RNN checkpoint's `s_mu/s_sd/a_mu/a_sd` verbatim,
  including `a_mu[7]=0, a_sd[7]=1`. Recomputing would break Part 4 input parity.

## Tasks

### Task 1 — trainer exists and is correct on a smoke run
- **Goal:** `roboturk/train_act.py` trains a state-only ACT on the frozen split
  and writes a resumable checkpoint every epoch.
- **Why:** without per-epoch checkpointing an MPS stall costs the whole run —
  exactly how the BC-RNN attempt died at epoch 10.
- **Files:** `roboturk/train_act.py` (new). Deviates from the repo's
  `scripts/`-only convention because the RoboTurk track is self-contained in
  `roboturk/` alongside its own npz, checkpoint, notebooks and JSON, and shares
  no imports with the LIBERO pipeline.
- **Steps:** assert split reproduces 316/56/79/80 -> build chunk index ->
  ACT + CVAE loss (L1 + beta*KL) -> loop with per-epoch save -> `--max-batches`
  escape hatch for the smoke run.
- **Done when:** `--epochs 1 --max-batches 30` prints a falling train loss and
  leaves `roboturk_act.pt` on disk with keys `model/epoch/cfg`.

### Task 2 — full 50-epoch run completes
- **Goal:** a trained ACT checkpoint with logged train/val loss per epoch.
- **Why:** Part 4 gates the benchmark on "loss decreased monotonically or
  plateaued sensibly, no divergence, no NaN" — which needs the curve, not just
  the final weights.
- **Files:** `roboturk/roboturk_act.pt`, `roboturk/act_loss_history.json`.
- **Steps:** run 50 epochs at stride 8 in the background; check the curve at
  epochs 1, 5 and 25.
- **Done when:** 50 epochs logged, no NaN, final val L1 below epoch-1 val L1.

### Task 3 — fingerprint and Part 4 sanity checks
- **Goal:** `act_fingerprint.json` mirrors `fingerprint.json`'s discipline, and
  every Part 4 check is reported pass/fail.
- **Why:** Part 2.5 requires the same handoff discipline; Part 4 forbids
  proceeding to the benchmark on unverified premises.
- **Files:** `roboturk/act_fingerprint.json`.
- **Steps:** SHA-256 the checkpoint, record libs/config/D1-D3 -> assert
  normalisation stats are byte-identical to BC-RNN's -> assert cal/eval episode
  ids are untouched.
- **Done when:** the file exists and the input-parity assertion passes.

## Out of scope

Parts 3, 5 and 6 (thresholds, three-way benchmark, report). Part 3's `q_esc`
should be recomputed on the corrected score, which is a separate decision.
