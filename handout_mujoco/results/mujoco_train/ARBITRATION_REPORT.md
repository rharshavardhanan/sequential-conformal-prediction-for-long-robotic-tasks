# Live MuJoCo policy arbitration — results

Generated from re-runnable scripts; every number below has a script that produces it.

## Claims (verbatim, as agreed)

> "Live MuJoCo closed-loop policy arbitration. Reach stage: scripted P-servo (100%,
> median 4 steps). Push expert: TQC trained in MuJoCo, 300k steps, 0.930 on 200
> held-out seeds (CI95 0.895-0.965), single seed. Cheap arm: 755-parameter MLP
> distilled from this composed expert. Escalation threshold q_esc calibrated by
> split conformal on 1,000 disjoint calibration seeds, alpha=0.10, K=2, Bonferroni.
> All policies run online; every observation comes from MuJoCo physics.
>
> Pending: learned reach expert (Gate 3), 3-seed push variance (Gate 2), composed
> handoff refit (Gate 4). The bounded quantity is the spurious-escalation rate, not
> safety of the arbitrated trajectory. CP is not claimed to improve success — at a
> matched firing rate it is the same decision rule as a tuned fixed threshold."

## Gate 6 — score informativeness (push phase, held-out calibration half)

| candidate | h=2 | h=4 | h=8 | all steps |
|---|---:|---:|---:|---:|
| (a) compliance residual | 0.658 | 0.602 | **0.610** | 0.898 |
| **(b) ensemble disagreement** | 0.628 | 0.701 | **0.813** | 0.924 |
| (c) predicted action std | 0.580 | 0.567 | 0.668 | 0.853 |
| (d) ‖ee − cube‖ | 0.574 | 0.543 | 0.575 | 0.802 |

Selected on seeds 300,000–300,499, reported on 300,500–300,999. **(a) — the score the
PyBullet system used — fails the 0.65 gate at decision time (0.610)** while looking strong
in hindsight (0.898 over all steps). Reach-phase rows are not reach failures: reach
succeeds 99.7%, so they measure whether reach-phase signals foresee a push failure.

## Step 4 — budget compliance under shift (the CP claim)

Regimes split on reset geometry only (d0 = ‖cube − task_goal‖, calibration median
0.1581 m). Test regime HARD, 92 episodes, 60 calibration draws.

| method | cal regime | test | target | actual | \|err\| |
|---|---|---|---:|---|---:|
| fixed | EASY | HARD | 0.20 | 0.775 ± 0.015 | 0.575 |
| cp_recal | HARD | HARD | 0.20 | 0.273 ± 0.018 | 0.073 |
| cp_carry | EASY | HARD | 0.20 | 0.775 ± 0.015 | 0.575 |
| random | n/a (matched rate) | HARD | 0.20 | 0.266 ± 0.052 | 0.066 |
| fixed | EASY | HARD | 0.30 | 0.816 ± 0.010 | 0.516 |
| cp_recal | HARD | HARD | 0.30 | 0.347 ± 0.021 | 0.047 |
| cp_carry | EASY | HARD | 0.30 | 0.816 ± 0.010 | 0.516 |
| random | n/a (matched rate) | HARD | 0.30 | 0.342 ± 0.045 | 0.042 |
| fixed | EASY | HARD | 0.40 | 0.886 ± 0.021 | 0.486 |
| cp_recal | HARD | HARD | 0.40 | 0.423 ± 0.016 | 0.023 |
| cp_carry | EASY | HARD | 0.40 | 0.886 ± 0.021 | 0.486 |
| random | n/a (matched rate) | HARD | 0.40 | 0.424 ± 0.062 | 0.024 |

**CP 4.7 pp (gate < 5 pp: PASS). Fixed 52.5 pp (gate > 15 pp: PASS).**
Control identity holds: CP-carried-over is bit-identical to the fixed threshold
(`True`), which is what isolates the advantage to *recalibration being possible
on unlabelled scores*, not to conformal being intrinsically better.

Sampling floor at n=92 is 3.7 pp — CP is close to the limit the test-set size allows.

**This is budget compliance, not success.** The drifted fixed threshold fires on
77%–89% of episodes against a 20%–40% budget; it will often score HIGHER on
success precisely because it spends compute it was not authorised to spend.

### Exclusion variant (regime feature removed from the trigger)

Masking the task_goal columns drops d0 recoverability from R² 0.998 to
**0.395** (MLP probe, held-out). **The exclusion is PARTIAL** — the trajectory state
still leaks the goal — and that is measured rather than assumed. A linear probe was tried
first and reported ~0 for both, because it cannot represent a norm; it proved nothing.

With the masked trigger: CP 2.8 pp, fixed 47.9 pp. Both gates still pass.

## Step 5 — paired evaluation, seeds 400,000–400,199, all timing CPU batch-1

| arm | success (Wilson) | reach | ep_len | expert calls | expert % | switches | 1st esc | ms/step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| expert_only | 0.925 [0.880, 0.954] | 1.000 | 14.1 | 0.00 | 0.0 | 0.00 | — | 0.446 |
| cheap_only | 0.680 [0.612, 0.741] | 0.995 | 22.8 | 0.00 | 0.0 | 0.00 | — | 0.420 |
| arbitrated_a0.20 | 0.820 [0.761, 0.867] | 1.000 | 16.6 | 0.61 | 30.0 | 0.74 | 6 | 0.494 |
| random_a0.20 | 0.815 [0.755, 0.863] | 0.995 | 18.6 | 0.39 | 12.8 | 0.45 | 21 | 0.581 |
| arbitrated_a0.30 | 0.850 [0.794, 0.893] | 1.000 | 15.8 | 0.70 | 37.6 | 0.78 | 6 | 0.500 |
| random_a0.30 | 0.895 [0.845, 0.930] | 1.000 | 13.8 | 1.00 | 49.6 | 1.06 | 6 | 0.508 |
| arbitrated_a0.40 | 0.885 [0.833, 0.922] | 1.000 | 14.4 | 0.76 | 42.0 | 0.82 | 6 | 0.530 |
| random_a0.40 | 0.895 [0.845, 0.930] | 1.000 | 13.8 | 1.00 | 50.0 | 1.04 | 6 | 0.514 |

gain_recovered = (arb − cheap)/(expert − cheap): **0.571** at α=0.20,
**0.694** at α=0.30, **0.837** at α=0.40.

### The negative result, stated plainly

| α_esc | arbitrated | random | Δ | discordant | McNemar p |
|---|---:|---:|---:|---|---:|
| 0.20 | 0.820 | 0.815 | +0.005 | 19/18 | 1.000 |
| 0.30 | 0.850 | 0.895 | -0.045 | 12/21 | 0.163 |
| 0.40 | 0.885 | 0.895 | -0.010 | 13/15 | 0.851 |

**Arbitration is not distinguishable from random escalation.** No comparison reaches
significance (p = 1.000, 0.163, 0.851), and at α=0.20 random attains the same success
(0.815 vs 0.820) using **12.8% expert steps against arbitration's 30.0%** — i.e. better
per unit of expert compute.

This is the same finding as the PyBullet system, and it survives a score with AUC 0.813 at
decision time. A high AUC is necessary but not sufficient: the *value* of escalating
depends on whether the expert's rescue probability varies with **when** it is called. Here
it does not — the push expert rescues from almost anywhere, so knowing the right moment
buys nothing over escalating at a random one.

A methodological note that matters for reading the table: the random control could not be
matched exactly on expert-step share. A memoryless Bernoulli trigger plus the 8-step
handback yields a coarse achievable set — measured points 12.8% and 49.6% with nothing
between, because one early escalation captures most of a ~15-step episode. The control is
therefore **bracketed, not exactly matched**, and matching on the nominal per-step
probability (the first attempt) compared two arms at different operating points.

## Step 6 — live demo

`live/demo_reach_push.py`. Closed loop, real models, fresh MuJoCo observation every step,
nothing replayed. Score gauge, q_esc (orange) and q̂ (red) drawn in-scene; SPACE / RIGHT /
R / Q; JSONL log.

Demo seeds (cheap FAILS, arbitrated SUCCEEDS), {ds['n_rescues']} such rescues in {ds['n_eval']} eval episodes:

- **400016** — first escalation step 10 (push), expert on 47% of steps, 1 call

- **400167** — first escalation step 10 (push), expert on 61% of steps, 1 call

- **400169** — first escalation step 6 (reach), expert on 71% of steps, 1 call

**On the visual claim, accurately:** q_esc < q̂ holds by construction in every phase, so
escalation always fires at the lower line first. It does **not** follow that the
trajectory stays under q̂ afterwards — measured on the three demo seeds, s(t) stays below
q̂ on 400016 and 400167 but crosses it twice on 400169 (max 1.55× q̂). Do not narrate the
demo as "the trajectory never reaches the red line".

## What is NOT claimed

- Never quote 0.930 as a composed-success prediction.
- No arbitrated run is "CP-certified safe". The bounded quantity is the spurious-escalation rate.
- No MuJoCo number is compared to a PyBullet number as the same experiment.
- No CP-vs-fixed **success** comparison is reported; at a matched firing rate they are the
  same decision rule.
