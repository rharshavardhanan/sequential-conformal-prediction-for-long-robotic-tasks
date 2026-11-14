# Design — RoboTurk BC-RNN threshold calculation (safety + escalation)

**Date:** 2026-09-10
**Status:** implemented 2026-09-10
**Executes:** `roboturk/RoboTurk_ACT_and_Full_Benchmark.md` Part 3.
**Supersedes:** `roboturk/partD.json` — superseded in place by a new file, not edited.

## 1. Purpose

Produce the threshold values the arbitration benchmark needs from the frozen RoboTurk
BC-RNN: the safety threshold `q̂_k` and the escalation threshold `q̂_k,esc`, per phase
(`closed` / `open`), on a score that is not dominated by the grasp bool.

Part 3 of the benchmark file gates Part 5 on `q̂_k,esc < q̂_k` in both phases. That check
is only meaningful if both thresholds are computed on the *same* score, so both are
recomputed here in one pass.

## 2. Why the existing thresholds cannot be used

`partD.json` records `q̂_closed = 2.284`, `q̂_open = 2.211`, eval coverage `1.000`. Those
numbers are valid split-conformal output and simultaneously useless:

The Part D score `‖a_pred − a_demo‖₂` is taken over the **8-dim** action, whose 8th column
is the grasp bool. Training deliberately sets `a_mu[7]=0, a_sd[7]=1` to keep it in {0,1}
(notebook cell 26), so that column enters the L2 norm **unnormalised** while the seven
continuous joint targets are z-scored to roughly unit variance and carry errors of ~0.1.

Measured on the 79 calibration episodes: at the timestep that *sets* each phase maximum,
the grasp bool owns a **median 98.5%** of the squared error, in **83%** of episodes.
`|grasp err|` median 0.973 against `‖7 continuous‖` median 0.119.

The threshold is therefore a readout of gripper-toggle timing, not of trajectory
divergence. It explains three observations at once: `q̂_k = 2.28` against an eval max of
1.213 (vacuous); `q_esc` barely moving across `α_esc` (0.989 → 0.983); and escalation
firing on **0.16%** of eval timesteps at `α_esc = 0.30`, which would collapse arbitration
to BC-RNN-alone and make H1 unfalsifiable.

This is the same failure class as `osc-compliance-gain-confounds-tracking-error` on the
simulation side: the score is dominated by a nuisance quantity rather than the signal.

**The corrected score is the 7 continuous dims only** — `‖â[:7] − a[:7]‖₂`. The grasp bool
still defines the phase (it is the K=2 phase variable, per `fingerprint.json`
`phase_model`), it just stops contributing to the magnitude being thresholded.

### 2.1 Post-implementation correction — the grasp bool was not the operative cause

Written before the measurement, this section claimed the grasp bool "explains three
observations at once". **It explains one of them.** Implementation measured the following,
and the claim above is wrong on the part that matters:

| | 8-dim | 7-dim | verdict |
|---|---|---|---|
| median share of sq. error owned by the grasp bool at the phase-max timestep | 0.985 | — | **confirmed** |
| `q̂_k` closed / open | 2.284 / 2.211 | 2.200 / 2.186 | moved 3.6% |
| eval coverage | 1.000 | 1.000 | **still vacuous** |
| eval firing rate at `α_esc = 0.30`, per-episode max | 0.16% | 0.15% | **unchanged** |

`argmax(err7)` and `argmax(err8)` coincide in only **12.7%** of phase-windows: once the
grasp bool stops setting the magnitude, the episode maximum relocates to a different
timestep rather than falling to the ~0.119 continuous error measured at the old one.

The operative cause is the one §3.1 already identified independently — **the
max-over-thousands-of-timesteps aggregation**. Pooled `err7` is heavy-tailed (p50 0.081,
p90 0.204, p99 1.457, p100 5.353), so a max over ~2,500 timesteps per phase-window lands
deep in that tail whatever the score is. Only 0.31% of calibration timesteps exceed
`q̂ = 2.2005`.

The 7-dim score is still the right score — the grasp bool is genuinely a nuisance quantity
and the diagnostic is real. It simply was not the bottleneck. §3.1's reasoning survives
intact; §2's "explains three observations" does not.

## 3. What is computed

Two thresholds × two aggregations × two phases.

| threshold | α | aggregation | carries a guarantee? |
|---|---|---|---|
| `q̂_k` (safety) | `α/K = 0.05` | per-episode max | **yes** — split-conformal, exchangeable episodes |
| `q̂_k,esc` (escalation) | `α_esc/K`, `α_esc ∈ {0.20, 0.30, 0.40}` | per-episode max | **yes** — bounded spurious-escalation rate |
| `q̂_k` per-timestep | `α/K` | pooled timesteps | **no** |
| `q̂_k,esc` per-timestep | `α_esc/K` | pooled timesteps | **no** |

`α = 0.10`, `K = 2`, equal Bonferroni split — inherited unchanged from Part D.
`α_esc = 0.30` is the primary operating point per the benchmark file; 0.20 and 0.40 are
reported as secondary comparison points.

### 3.1 The aggregation decision, stated explicitly

Calibrating on per-episode maxima while *triggering* per-timestep is a category error, and
it is the reason the per-episode-max `q_esc` fires on 0.16% of timesteps: a threshold set
at the 95th percentile of ~5,900-step episode maxima is, by construction, above almost
every individual timestep.

Pooling timesteps fixes the firing rate and breaks exchangeability — ~500k timesteps within
an episode are strongly autocorrelated and are not 500k independent draws. A pooled
quantile therefore has **no coverage guarantee of any kind**.

Both are computed and both are reported. The per-episode-max numbers are the ones that
carry the guarantee. The per-timestep numbers are the operating point for Part 5's replay,
labelled in the output as a **compute-budget heuristic with no coverage guarantee**. The
safety guarantee lives in `q̂_k` alone and no output may imply otherwise.

## 4. Architecture

One script, `roboturk/calibrate_bcrnn.py`. It sits in `roboturk/` rather than `scripts/`
for the same reason `train_act.py` does: the RoboTurk track is self-contained alongside its
own npz, checkpoint, notebooks and JSON, and shares no imports with the LIBERO pipeline.

| unit | responsibility | depends on |
|---|---|---|
| `train_act.load_everything()` (imported) | rebuild the frozen split, reload BC-RNN normalisation | npz, BC-RNN checkpoint |
| `BCRNN` (redefined) | the architecture the `state_dict` loads into | checkpoint |
| `episode_scores()` | one forward pass → per-timestep err + phase mask | model, normalisation |
| `conformal_threshold()` | order-statistic quantile | nothing (pure) |
| `main()` | cache, threshold, verify, report | all of the above |

`load_everything()` is imported rather than reimplemented. It already asserts the split
reproduces 316/56/79/80 and that `a_sd[7] == 1.0`; duplicating it would let the two copies
drift and silently break the parity Part 4 depends on.

`BCRNN` must be redefined because the checkpoint stores a bare `state_dict`, not a pickled
module. Copied minimally from notebook cell 28 — encoder, LSTM, GMM head, `gmm_mean`.
`gmm_nll` is not copied; nothing here trains.

### 4.1 Data flow

```
roboturk_20hz.npz ─┐
                   ├─→ load_everything() ─→ split (cal 79 / eval 80), s_mu/s_sd/a_mu/a_sd
BC-RNN checkpoint ─┘                                    │
                                                        ▼
                          for each of 159 episodes: one full-sequence forward (CPU)
                                                        │
                                                        ▼
                              bcrnn_scores.npz  ← per-timestep err7, err8, grasp mask
                                                        │
                          ┌─────────────────────────────┼─────────────────────────────┐
                          ▼                             ▼                             ▼
                  8-dim reproduction            7-dim per-episode-max          7-dim per-timestep
                  vs partD.json (GATE)          → q̂_k, q̂_k,esc                → q̂_k,esc,ts
                                                        │
                                                        ▼
                                                  partD_v2.json
```

### 4.2 The score cache

`roboturk/bcrnn_scores.npz` holds per-timestep `err7`, `err8` and the grasp mask for all
159 cal+eval episodes, flat with an episode-boundary index.

The forward pass is the only expensive step (159 episodes × ~5,900 steps). Every downstream
question — a different `α`, Part 5's arbitration replay, a re-threshold after a framing
change — is then a quantile over an array rather than a re-run of the network. `err8` is
cached alongside `err7` specifically so the reproduction gate in §5 costs nothing.

The cache is keyed by the checkpoint SHA-256 recorded in `fingerprint.json`; a mismatch
invalidates it and forces a recompute, so a checkpoint swap cannot silently reuse stale
scores.

### 4.3 Compute environment

ACT training currently holds MPS. Scoring runs on **CPU** to avoid contending with it —
BC-RNN is 1.15 M params and the pass is a one-off, so the CPU cost is acceptable and the
alternative risks perturbing a 4-hour training run whose timing measurements Part 5.3
depends on.

This is scoring, not timing. Part 5.3's back-to-back inference measurement is a separate
task on a quiet machine and is out of scope here.

## 5. Verification — the reproduction gate

The script recomputes the **8-dim** per-episode-max thresholds from the same cached scores
and asserts they reproduce `partD.json`:

```
q̂_closed = 2.2835631…   q̂_open = 2.2110543…
```

to **1e-3 relative** (~0.002 on 2.28), with the measured delta always printed and recorded.
Bit-identity is not achievable and demanding it would fail a correct implementation: the
published numbers came from CUDA (torch 2.6.0+cu118, RTX 3050) and this runs on CPU with
torch 2.10, and LSTM and matmul reductions differ in association order between those
backends. 1e-3 is nonetheless far tighter than any real reconstruction error — a wrong
slice, a missed denormalisation or a redrawn split moves the number by whole units, not
thousandths. The rank and `n_cal` must match exactly.

If they match, the loader, the model, the scoring and the quantile are
provably the same pipeline that produced the published Part D result, and the 7-dim numbers
inherit that trust. If they do not match, something in the reconstruction differs from the
notebook and **the 7-dim numbers mean nothing** — the script writes no output and exits
non-zero.

This gate replaces a separate unit-test file. It is a strictly stronger test of the same
code: it exercises the whole chain against a known-good published result, where a unit test
would only re-check rank arithmetic copied verbatim from a notebook that already produced a
validated number.

### 5.1 Other checks reported

- **Dual-threshold check:** `q̂_closed,esc < q̂_closed` and `q̂_open,esc < q̂_open`, per
  aggregation, per `α_esc`. Part 3 stops the benchmark if this fails.
- **Grasp-bool diagnostic:** the median share of squared error owned by column 7 at each
  phase-max timestep, on the 8-dim score — confirms the 98.5% figure independently and
  demonstrates the 7-dim fix removes it.
- **Eval coverage** at `q̂_k`, per phase and joint, target `≥ 0.90`.
- **Firing rate:** fraction of the 80 eval episodes' timesteps that would escalate, per
  aggregation and `α_esc`. This is the number that decides whether Part 5 is viable at all.

## 6. Output

`roboturk/partD_v2.json`:

```json
{
  "supersedes": "partD.json",
  "score": "||a_pred - a_demo||_2 over the 7 CONTINUOUS dims (user_control[:,0:7]); grasp bool excluded from the magnitude, retained as the K=2 phase variable",
  "alpha": 0.10, "K": 2, "alpha_k": 0.05,
  "checkpoint_sha256": "...",
  "reproduction_gate": { "partD_q_closed": 2.2836, "recomputed": 2.2836, "passed": true },
  "grasp_diagnostic": { "median_share_of_sq_err": 0.985, "episodes_dominated": 0.83 },
  "safety":     { "closed": {...}, "open": {...} },
  "escalation": { "0.20": {...}, "0.30": {...}, "0.40": {...} },
  "caveats": "per-timestep thresholds pool autocorrelated timesteps, break exchangeability, and carry NO coverage guarantee; they are a compute-budget heuristic. The safety guarantee lives in the per-episode-max q_hat_k alone."
}
```

## 7. Error handling

- Missing npz / checkpoint / `fingerprint.json` → stop with the missing path named. Part 0
  of the benchmark file already specifies "stop" for each of these.
- Split assertion fails inside `load_everything()` → propagates; the frozen split is a
  correctness precondition, not a warning.
- Reproduction gate fails → report both values and the delta, write nothing, exit 1.
- A phase absent from an episode → that episode contributes to the other phase only, per
  Part D's `if closed.any()` / `if (~closed).any()` handling. Part B established this is
  0/531 in practice, but the guard is kept.
- `conformal_threshold` infeasible (`rank > n`) → returns `inf` and `feasible: false`,
  verbatim from cell 34. At n=79, `α_k=0.05` needs rank 76, so this cannot fire here; the
  branch is kept because the α_esc sweep is parameterised.

## 8. Out of scope

- Part 5's three-way benchmark, and any ACT involvement whatsoever.
- Part 5.3's inference timing.
- Editing `partD.json` in place. It stays as the provenance record of what the 8-dim score
  produced.
- Retraining or re-fingerprinting BC-RNN. The checkpoint is frozen.
- Any detection-rate or false-alarm language. No failure labels exist in RoboTurk; every
  demo is a success. Per the benchmark file's standing rules, none of this output may be
  described as catching failures.
- Comparing these numbers against the LIBERO/simulation results. Different action space,
  different score, different hardware — side by side only.
