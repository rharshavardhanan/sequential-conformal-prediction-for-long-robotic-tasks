# Part D corrected + Part 3 escalation on a deployable trigger

**Date:** 2026-09-11
**Supersedes for Part 3:** `RoboTurk_ACT_and_Full_Benchmark.md` Part 3, which
calibrates `q_esc` on the error score. Two reasons that cannot stand:

1. **It is an oracle.** `||a_pred - a_demo||` needs the human's action, which does
   not exist at runtime. A trigger that cannot run on a robot cannot support an
   arbitration claim.
2. **It is contaminated.** The 8-dim score is ~98.5% grasp bool at the
   threshold-setting timestep, so `q_esc` fires on 0.16% of eval timesteps.

Both are fixed by changing basis: score the **7 continuous dims**, and trigger on
**BC-RNN's own predicted sigma**, which is available at every timestep with no
ground truth. Measured 2026-09-11: escalating the top 10% by sigma gives ACT a
21.3% error reduction on that slice versus 7.7% overall, and the gain rises
monotonically as the slice tightens (30.5% at 5%) -- the ranking is informative.

## Frozen inputs (unchanged)

`roboturk_20hz.npz`, seed-0 split 316/56/79/80, BC-RNN checkpoint
`a3f61533...`, ACT v2 `roboturk_act_best.pt` (epoch 11, 9.8M, history 20).

## Tasks

### Task 1 — corrected safety threshold `q_hat_k`
- **Goal:** a non-vacuous `q_hat_k` per phase on the 7-dim score.
- **Why:** the shipped `q_hat_k = 2.28` has eval coverage 1.000 against a target
  of 0.90 and an observed eval max of 1.213 -- it can never be violated, so it
  certifies nothing.
- **Files:** `roboturk/part_d_corrected.py`, `roboturk/partD_corrected.json`.
- **Steps:** per-episode phase max of the 7-dim BC-RNN error -> conformal
  quantile at `alpha_k = 0.05` -> eval coverage.
- **Done when:** coverage lands near 0.90-0.97 rather than exactly 1.000, and
  `q_hat` is inside the observed score range.

### Task 2 — escalation threshold on sigma, with a guarantee that is actually valid
- **Goal:** `lambda_phase` firing at target rates (5/10/20%), plus a conformal
  bound on the per-episode escalation rate.
- **Why:** pooling timesteps breaks exchangeability, so a pooled quantile is a
  budget heuristic with no coverage claim. The per-episode escalation *rate* is
  an episode-level statistic and episodes ARE exchangeable -- conformal on that
  is valid, and it is the quantity a compute budget actually cares about.
- **Files:** same.
- **Steps:** pooled cal quantile of sigma per phase -> lambda -> per-episode rate
  on cal episodes -> conformal upper bound -> verify on eval.
- **Done when:** the reported bound holds on >= 90% of eval episodes.

### Task 3 — arbitrated fidelity (Part 5.1 / 5.2)
- **Goal:** H1 and H2 as numbers.
- **Why:** the whole point; H1 is arbitration vs BC-RNN alone, H2 vs ACT alone.
- **Files:** same.
- **Steps:** arbitrated error = ACT where sigma > lambda else BC-RNN -> mean per
  configuration -> report H1, H2, and the fraction of ACT's full-time advantage
  recovered.
- **Done when:** all three configurations reported at each escalation rate.

## Note on the dual-threshold check

Part 4 asks to confirm `q_esc < q_hat_k`. That check assumes both live on the
same axis. Once the trigger is sigma and the score is error, they have different
units and the comparison is meaningless. Replaced by the check that matters:
escalation must fire strictly more often than the safety threshold is exceeded,
i.e. it is a leading indicator, not a lagging one.
