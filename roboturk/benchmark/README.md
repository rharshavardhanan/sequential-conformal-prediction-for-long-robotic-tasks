# Part 5.3 — compute efficiency

**Measured 2026-09-12** on this machine, back-to-back in one session with nothing else
running, which is what Part 5.3 requires. Regenerate with:

```bash
.venv-lerobot/bin/python roboturk/compute_benchmark.py   # -> compute_benchmark.json
.venv-lerobot/bin/python roboturk/benchmark_charts.py    # -> the three PNGs
```

Policies: **BC-RNN** 1,150,037 params (cheap arm) · **ACT v2** `roboturk_act_best.pt`
9,805,128 params, chunk 16, history 20, epoch 11 (expensive arm). ACT v1 is out of scope.

## Headline

**The result is device-dependent, and on the GPU the cheap/expensive framing inverts.**

| | CPU | MPS |
|---|---|---|
| BC-RNN, per control step | 0.059 ms | **0.185 ms** |
| ACT v2, per control step (chunked ÷16) | 0.132 ms | **0.124 ms** |
| arbitrated @ 11.16% escalation | 0.073 ms | 0.199 ms |
| **ACT-alone ÷ arbitrated** | **1.80×** | **0.62×** |

On **CPU**, arbitration is 1.80–2.25× cheaper than running ACT everywhere. H3 holds.

On **MPS**, chunked ACT (0.124 ms) is *cheaper per step than BC-RNN* (0.185 ms), so
arbitration costs **more** than simply running ACT everywhere, at every escalation
fraction. H3 fails on this device — not marginally, and not fixable by tuning `α_esc`.

**Why:** BC-RNN is a 2-layer LSTM that must be stepped one frame at a time — inherently
sequential, so on a GPU its cost is dominated by per-step kernel-launch overhead that it
cannot amortise. ACT does one parallel forward covering 16 steps. Action chunking, not
parameter count, decides per-step cost: ACT has **8.5× more parameters** and is still
cheaper per step on MPS.

Arbitration also cannot escape the floor, because **BC-RNN runs on every step regardless** —
the monitor needs its score to decide whether to escalate. Arbitrated cost is therefore
`bc + f·act`, which starts at BC-RNN-alone and only rises.

## The chunking caveat — read before quoting any number

ACT emits 16 actions per forward. Two regimes, and they differ by up to 16×:

| regime | ACT forwards per control step | use |
|---|---|---|
| **chunked** | 1/16 | **headline** — ACT as it would be deployed |
| per-step | 1 | **upper bound only** — what `scripts/roboturk_arena.py` does |

The arena re-runs ACT every step and keeps 1 of 16 actions. That is correct for a viewer
that needs ACT's opinion at every step to draw the comparison, but it is not deployed cost.
**Never cite the per-step number without naming it as the upper bound** — doing so overstates
the expensive arm by up to 16× and flatters arbitration by the same factor.

## Measurement method, and a correction it forced

Timing is **throughput** (sync once per run), not per-call latency:

| device | policy | per-call latency | throughput | sync tax |
|---|---|---|---|---|
| CPU | BC-RNN | 0.059 ms | 0.059 ms | +0.001 |
| CPU | ACT (full fwd) | 2.115 ± 0.260 ms | 2.116 ms | −0.001 |
| MPS | BC-RNN | 0.380 ± 0.045 ms | 0.185 ms | **+0.196** |
| MPS | ACT (full fwd) | 2.630 ± 0.075 ms | 1.985 ms | **+0.645** |

MPS dispatch is asynchronous. Calling `torch.mps.synchronize()` after *every* call serialises
the pipeline and charges a fixed tax — ~0.20 ms on BC-RNN, roughly half its apparent per-call
time. A real control loop enqueues the next op while the GPU works and never pays it.

This was caught by the estimate-vs-direct cross-check, which is exactly what that check is
for. Built from per-call numbers, the additive cost model over-predicted the measured
arbitrated replay by **39%** on MPS. Rebuilt from throughput it predicts it to **0.5%**:

| device | estimate | direct measurement | rel. error | |
|---|---|---|---|---|
| CPU | 0.281 ms/step | 0.288 ms/step | 2.6% | AGREES |
| MPS | 0.393 ms/step | 0.395 ms/step | 0.5% | AGREES |

The cost model was right; the measurement method was wrong. Per-call latency is still
reported in `compute_benchmark.json` because it answers a different question — *can one
inference meet a control deadline* — which is not the same as *what does a step cost*.

## Escalation fraction is swept, not assumed

Three triggers exist in this repo and they disagree by two orders of magnitude, so cost is
reported as a function of `f` with all three marked rather than picking one:

| trigger | source | `f` | CPU speedup | MPS speedup |
|---|---|---|---|---|
| score vs per-episode-max `q_esc`, α=0.30 | `partD_v2.json` (what the arena loads) | 0.15% | 2.25× | 0.67× |
| score vs p75 `q_esc`, cross-conformal | `cross_conformal.json` | 2.00% | 2.16× | 0.66× |
| BC-RNN predicted σ, α=0.10 | `partD_corrected.json` | 11.16% | 1.80× | 0.62× |

Choosing between these triggers is a separate decision and is not made here.

## Charts

| file | shows |
|---|---|
| `latency.png` | per-control-step cost, both policies, both devices — the bars flip between CPU and MPS |
| `cost_vs_escalation.png` | arbitrated cost vs `f`, with BC-RNN-alone and ACT-alone reference lines and the three operating points |
| `efficiency_summary.png` | `cost_ACT ÷ cost_arbitrated` at each operating point, against the 1.0× break-even line |

## Live readout

`scripts/roboturk_arena.py` shows rolling ms/step, cumulative compute and compute saved
versus always-ACT, priced from this benchmark's measured numbers:

```bash
.venv-lerobot/bin/python -m scripts.roboturk_arena --episode 0 --max-steps 400
```

The displayed figure is **attributed** cost — the cost the arbitrated system *would* have
incurred given the arm the arbiter selected — not the arena process's wall clock. The arena
deliberately runs both arms every step so it can draw the comparison, so its raw wall clock
is not an arbitrated cost and is never shown as one. If `compute_benchmark.json` is absent,
the panel hides itself rather than inventing numbers.

## Caveats

- Cost is per **control step**. The arena's `--stride` is display-only; conflating the two
  would inflate every number 4×.
- No FLOP counts: `fvcore`/`ptflops` are not installed, and adding a dependency for a
  hardware-independent proxy of a measurement we can take directly is not justified.
  Parameter counts serve that role.
- These numbers are Apple-Silicon-specific. The MPS inversion is a property of how a
  sequential LSTM maps onto this GPU; a CUDA box with different launch overhead could land
  anywhere, and nothing here should be quoted as a general claim about BC-RNN vs ACT.
- Nothing here is a fidelity claim. Which policy predicts the human better is Part 5.1,
  in `partD_corrected.json` and `compare_policies.py`.
