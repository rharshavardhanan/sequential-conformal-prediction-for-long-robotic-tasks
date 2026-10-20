# Arbiter interface — recovered logic, exact constants, one unverified equivalence

**Status: PARTIAL (substantially recovered).** The score, thresholds, budgets and
hysteresis are all pinned to artifacts. One gap: the deployed module was
`scp.arbitration`, which is absent; a structurally equivalent implementation exists in
this repo at `scripts/arbiter.py` but has **not** been verified identical.

## The score

From `results/part45_benchmark.json: .doc` (verbatim):

```
s(t) = || clip(a)*0.05 - (ee(t+1) - ee(t)) || / || clip(a)*0.05 ||
```

A compliance residual: commanded end-effector displacement versus achieved. Masked by
`MOTION_FLOOR` (`scp.scoring`) — steps whose commanded motion is ~0 are unscorable and
excluded. Measured share below the floor: **2.087%** (`part4.frac_steps_below_floor`).

`meta.json: arrays.score` confirms "NaN/inf possible when commanded motion ~0".

## Calibration (Part 4)

| item | value | source |
|---|---|---|
| calibration episodes attempted | 600 | `part4.n_cal_episodes` |
| successful cheap episodes kept | 256 | `part4.cheap_successes` / `part4.kept` |
| degenerate / empty-phase discarded | 0 | `part4.degenerate_empty_phase` |
| phases K | 2 (`reach`, `push`) | `part4.feasibility.K` |
| α (safety) | 0.10 | `part4.feasibility.alpha` |
| α_k after Bonferroni | 0.05 | `part4.feasibility.alpha_k` |
| minimum n per phase | 19 | `part4.feasibility.n_min_per_phase` |
| per-episode statistic | `S_k = max scorable s in phase k` | `part45_benchmark.json: .doc` |
| threshold routine | `scp.calibrate.calibrate_phases` (Bonferroni) | `.doc` |

## Thresholds (exact, as deployed)

| | reach | push | source |
|---|---|---|---|
| `q_hat` (safety, α=0.10) | 1.491860231071659 | 1.4520648299347214 | `meta.json: arbiter.q_hat_alpha0.10` |
| rank / n | 245 / 256 | 245 / 256 | `part4.q_hat.*` |
| `q_esc` @ α_esc=0.30 | 1.2373437836048184 | 1.2070184534633481 | `meta.json: arbiter.q_esc_alpha0.30` |
| `q_esc` @ α_esc=0.20 | 1.3086935877414265 | 1.3475776783180935 | `part4.escalation.0.2.q_esc` |
| `q_esc` @ α_esc=0.40 | 1.1334308402932578 | 1.1530400662657327 | `part4.escalation.0.4.q_esc` |

`q_esc < q_hat` in both phases at all three budgets (`q_esc_below_q_hat: true`).

## Decision rule

From `CLAIMS.md`: *"escalate to the expert when s(t) > q̂_esc[phase]; hand back after 8
quiet steps."* From `part45_benchmark.json: .doc`: *"`scp.arbitration` hysteresis, hold
8, chunk 1."*

```
out_of_band := scorable(t) and s(t) > q_esc[phase(t)]
if out_of_band:              -> EXPERT immediately;  calm := 0
elif current is EXPERT:      calm += 1
                             if calm >= HOLD (=8): -> CHEAP
HOLD = 8      chunk_boundary_only = False  (chunk 1 = every step is a boundary)
```

Asymmetric by design: escalate instantly, de-escalate only after sustained calm.

## Expert-call accounting

| quantity | α=0.30 value | source |
|---|---|---|
| expert calls per episode | 2.932 | `efficiency_summary.json: 3_arbitrated_a0.30.expert_calls_per_ep` |
| fraction of steps on expert (pooled) | 0.0899 | `frac_expensive_pooled_steps` |
| fraction per-episode mean | 0.1281 | `frac_expensive_mean_over_episodes` |
| episodes that ever escalated | 0.314 | `frac_episodes_escalated` |
| mean switches per episode | 0.37 | `switches_mean` |
| median first escalation step | 4 | `first_escalation_step_median` |

## The one unverified equivalence

`scripts/arbiter.py` in this repo implements exactly this dual-threshold rule —
immediate escalation on `packet.score > packet.q_esc`, hysteretic de-escalation after
`calm_steps`, switch counting, per-policy step accounting. Its docstring records the
same reasoning (a naive "fraction of q_safe" rule was tested and rejected at 56.7%
spurious escalation against a 10% budget).

**Differences that prevent calling it identical:**

- default `calm_steps = 20`, whereas the panda-gym run used **hold 8**
- it carries an extra `packet.stalled` escalation trigger; whether `scp.arbitration`
  had one is unknown
- it belongs to the LIBERO/RoboTurk track (`docs/status.md` §4), not the panda-gym one

So the logic is recoverable and the constants are exact, but confirming the deployed
implementation requires `scp/arbitration.py`.

## What the guarantee is (carry this wording)

From `scripts/arbiter.py`, and consistent with `CLAIMS.md`: the CP coverage guarantee
does **not** transfer to an arbitrated trajectory, because arbitration changes which
policy generates the data. What transfers is a bounded **spurious-escalation** rate:
`q_esc` is drawn from the same calibration set at a larger budget `α_esc > α`, so
`q_esc < q_hat` and escalation fires strictly before a safety violation would. Never
label an arbitrated run "CP-certified safe".
