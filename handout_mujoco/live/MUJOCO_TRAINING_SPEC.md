# MuJoCo-native Reach→Push training specification

**Date:** 2026-09-14 · branch `mujoco-arbitration` · **Phase 1 delivered** (two envs, Gate 1 passed 31/31). Nothing trained.

This specifies a **MuJoCo-native** benchmark. No PyBullet checkpoint, threshold or
success number is used. The only thing taken from the recorded episodes is the **task
definition** — the reset distribution and success radii — so the two systems solve the
same task and can be reported side by side without being pooled.

Everything below is read from source or measured live. Numbers cite the script that
produced them; nothing here is typed from memory.

---

## 1. What exists today

| File | State |
|---|---|
| `live/mj_env.py` | `MJLive`: real closed loop, tuned controller, `reset()`, `step()`, `ee()`, `cube()`, `success()`. **Not** a Gymnasium env. |
| `live/build_live_scene.py` | `scene_live.xml` — collidable cube (mass 0.05 kg, friction 1.0/0.005/0.0001, `condim=4`, `solref="0.004 1"`), valid keyframe. Loads: `nq=16 nv=15 nu=8 timestep=0.002`. |
| `live/smoke_test.py` | 4/4 PASS — action moves the arm, motion is caused by the action, direction follows the command, period is 0.04 s. |
| `live/tune_controller.py` | Controller selected: `kp×1, kv/kp=0.035, forcerange×10` → achieved/commanded 0.645. |

| `live/envs.py` | **Phase 1**: `ReachOnly` (24-dim Box) and `PushOnly` (Dict goal space), both wrapping `MJLive`. Gate 1 passed 31/31. |
| `live/mujoco_reach_push_env.py` | the composed single-agent env. **Retired** — it reached 4% success; kept as the record, and for `quat_to_euler`. |

### Gaps closed by Phase 1

1. ~~No Gymnasium interface~~ — two envs, both passing `check_env`.
2. ~~`observation()` raises~~ — each env defines its own layout (§3).
3. ~~No random reset~~ — `envs.sample_task()`, shared by both (§7).
4. ~~No phase state machine~~ — superseded: the phases are now *separate envs*, and the
   Phase-4 composer owns the latch.
5. ~~`stable_baselines3` / `sb3_contrib` NOT INSTALLED~~ — both at 2.9.0 in `.venv-lerobot`.

---

## 2. Action space — ESTABLISHED

| | |
|---|---|
| dimension | **3** |
| range | `[-1, 1]³`, `Box(low=-1, high=1, shape=(3,), float32)` |
| meaning | desired end-effector displacement, `a × 0.05 m` per control step |
| realisation | damped-least-squares Jacobian step on the `tcp` site (`mj_env.action_to_ctrl`), λ = 0.05 |
| gripper | blocked closed (`ctrl[7] = 0`), as in the reference task |
| control period | **0.04 s (25 Hz)** = 20 substeps at `dt = 0.002` |
| measured response | achieved/commanded **0.645** (reference system: 0.655) |

Verified: commanding `[1,0,0]` moves the EE +x by 0.0104 m in one control period; a zero action moves it 0.0025 m.

---

## 3. Observation — ESTABLISHED, two single-phase layouts

Phase 1 split the composed env into `ReachOnly` and `PushOnly` (§11). Each carries the
layout its algorithm needs, and both are slices of one 26-dim vector so the Phase-5
cheap MLP can run a single flat observation across both phases.

**Shared core, 18 dims** — identical in both envs, `envs._Base._core()`:

| # | field | dims | source |
|---|---|---|---|
| 0–2 | `ee_pos` | 3 | `d.site_xpos[tcp]` |
| 3–5 | `ee_vel` | 3 | `mj_jacSite(tcp) @ d.qvel` |
| 6–8 | `cube_pos` | 3 | `d.xpos[cube]` |
| 9–11 | `cube_rot` (euler xyz) | 3 | from `d.xquat[cube]` (w,x,y,z) |
| 12–14 | `cube_lin_vel` | 3 | `d.qvel[cube_dofadr : +3]` |
| 15–17 | `cube_ang_vel` | 3 | `d.qvel[cube_dofadr+3 : +6]` |

| env | space | contents |
|---|---|---|
| `ReachOnly` | `Box(-inf, inf, (24,), float32)` | core 18 + `reach_goal` 3 + `task_goal` 3 |
| `PushOnly` | `Dict` | `observation` core 18 · `achieved_goal` = `cube_pos` 3 · `desired_goal` = `task_goal` 3 |
| **cheap policy (Phase 5)** | `Box(-inf, inf, (26,), float32)` | core 18 + `reach_goal` 3 + `task_goal` 3 + phase one-hot 2 |

`PushOnly` must be a `Dict` goal space: `HerReplayBuffer` relabels `desired_goal` and
recomputes the reward, which requires the three keys by name. `ReachOnly` keeps the cube
fields although reach never touches the cube — dropping them would make the two layouts
stop being slices of the 26-dim cheap-policy vector.

**Deliberately excluded:** the 7 arm joint angles, as before — the task is
end-effector-space and 7 extra dims cost the ~8k-parameter cheap policy
disproportionately. Revisit only if training stalls.

**Normalization — PROPOSED:** none inside the env. For `ReachOnly` use SB3's
`VecNormalize` (observations only, `norm_reward=False`) and save `vecnormalize.pkl`
alongside the checkpoint. `PushOnly` is goal-conditioned and sparse-rewarded; normalize
it only if Gate 2 needs it, and record the decision either way.

---

## 4. Reward — ESTABLISHED per env

```
ReachOnly:  r = −‖ee − reach_goal‖ − 0.01·‖a‖²
PushOnly:   r = 0.0  if ‖cube − task_goal‖ < 0.05  else  −1.0
```

`PushOnly` is **sparse on purpose**. HER relabels a transition's `desired_goal` to a goal
that was actually achieved; a dense reward computed against the original goal is not
comparable to one computed against the relabelled goal, so relabelling under a dense
reward injects inconsistent targets. `compute_reward(achieved_goal, desired_goal, info)`
is vectorized over a leading batch axis, which is how `HerReplayBuffer` calls it.

The composed env's dense push reward (`−d_push − 0.2·d_hand`, +10/+50 bonuses) is
**retired**. It was measured to have a hovering optimum: a single TQC on the composed
task reached 4% success, median closest approach to the cube 10.7 cm, cube moved in 3%
of push phases. The agent banked the +10 reach bonus and hovered.

---

## 5. Episode / termination — ESTABLISHED, two independent budgets

Each env runs its own 50-step budget; there is no composed 100-step episode at training
time. Phase 4 composes them with a 50 + 50 = 100 step cap.

| env | event | condition | flag |
|---|---|---|---|
| `ReachOnly` | success | `‖ee − reach_goal‖ < 0.05` | `terminated` |
| `ReachOnly` | budget spent | 50 steps | `truncated` |
| `PushOnly` | success | `‖cube − task_goal‖ < 0.05` | `terminated` |
| `PushOnly` | budget spent | 50 steps | `truncated` |
| `PushOnly` | cube off table | `cube_z < −0.05` | `terminated` |

**Budget expiry truncates; it never terminates.** A time limit is not an MDP-terminal
state: under `terminated=True` SB3 zeroes the bootstrap, and the critic learns that
running out of time has value 0 rather than "the episode was cut here". `live/test_envs.py`
asserts the flag pair for each of the five conditions separately, so the two cannot drift
back into one.

Success/failure is reported through `info["is_success"]` in both envs, independently of
which flag fires.

---

## 6. Phase representation — ESTABLISHED

`labels`: **0 = reach, 1 = push** (`meta.json: arrays.labels`). Latching: once reach is
achieved the phase never returns to REACH. Phase is causal and observable, so it can
appear in the observation and condition the conformal thresholds.

---

## 7. Reset distribution — MEASURED, with one rejection rule

Both envs share `envs.sample_task()`, so the two experts see one task geometry.
Every number in this section comes from `analysis/phase1_measurements.py`.

| quantity | distribution |
|---|---|
| cube position | `x, y ~ U(−0.15, 0.15)`, `z = 0.02` (resting on the table) |
| `task_goal` | `x, y ~ U(−0.15, 0.15)`, `z = 0.02` |
| **rejection** | **resample while `‖cube − task_goal‖ < 0.05`** |
| `reach_goal` | **derived** — the Option-A waypoint, §11 |
| initial arm `qpos` | `ReachOnly`: constant `[0, 0.41, 0, −1.85, 0, 2.26, 0.79]`, fingers `[0, 0]`. `PushOnly`: placed at the cube, §12 |
| cube orientation | identity |

**Why the rejection rule exists.** Over 20,000 raw uniform pairs, **7.7%**
land inside the 0.05 success radius, so the episode is won at t = 0 before any action is
taken (minimum separation 0.0015 m; 0.33% below 0.01 m). Those are not
task instances: leaving them in would put a ~8% floor under every success number
this project reports, at every gate, and the floor would be invisible in the JSON.
Rejecting also conditions `unit(cube − task_goal)`, which the waypoint depends on.

This departs from the reference PyBullet task, which had no rejection (measured min
0.007 m). The two systems are reported side by side and never pooled, so the departure
costs nothing — but it is one more reason a MuJoCo success rate is not comparable to a
PyBullet one, before the physics even differ.

**Reachability, re-verified for the waypoint set.** The earlier "all 8 corners of the
±0.15 × [0, 0.30] box, worst-case error 0.0125 m" result does not cover the targets the
task actually uses: **62.3%** of Option-A waypoints fall outside that box
(max |x| 0.229, |y| 0.229). Re-measured over 400 sampled waypoints with the
kinematic DLS IK of `envs.place_ee`: residual median 7e-05 m, **max 1e-04 m,
0% unreachable**.

**Seed policy:** `train` 0–99 999 · `calibration` 300 000–300 999 · `evaluation`
400 000–400 999. Disjoint by construction; calibration never sees evaluation seeds.

---

## 8. Conformal score — carried forward as a definition only

```
s(t) = ‖ a_clipped · 0.05 − (ee(t+1) − ee(t)) ‖  /  ‖ a_clipped · 0.05 ‖
```

Unscorable when the commanded displacement is below a motion floor. The *formula* is
reused (it is a property of the task, not of PyBullet); **all thresholds will be
recalibrated from MuJoCo cheap-policy trajectories.** No PyBullet `q̂` is imported.

---

## 9. SB3 / TQC compatibility — VERIFIED

Both envs pass `gymnasium.utils.env_checker.check_env`, which is the conformance SB3
relies on. `ReachOnly` is a standard `Box(24)` / `Box(3)` continuous-control env.
`PushOnly` is a `Dict` goal env, which is what `HerReplayBuffer` requires:
`observation` / `achieved_goal` / `desired_goal` by name, plus a vectorized
`compute_reward`.

Installed in `.venv-lerobot` and confirmed: `stable_baselines3 2.9.0`,
`sb3_contrib 2.9.0`, `gymnasium 1.3.0`, `mujoco 3.2.7`, `torch 2.10.0`, `numpy 2.2.6`.
(The earlier "NOT INSTALLED" note is resolved.)

---

## 10. Files — Phase 1, delivered

| File | State |
|---|---|
| `live/envs.py` | **new** — `ReachOnly`, `PushOnly`, and four shared reset helpers |
| `live/test_envs.py` | **new** — 31 checks, Gate 1 runner, exit code is the gate |
| `live/mujoco_reach_push_env.py` | kept as the record of the 4% composed attempt; only `quat_to_euler` is imported from it |

`MJLive` is reused rather than replaced, so there is exactly one simulator.

```bash
cd handout_mujoco
PYTHONPATH=live ../.venv-lerobot/bin/python live/test_envs.py   # Gate 1
```

---

## 11. Reach goal semantics — Option A, decided 2026-09-14

`reach_goal` is a **pre-push waypoint**, not an arbitrary point in space:

```python
u          = unit(cube_pos − task_goal)   # points away from the goal
reach_goal = cube_pos + 0.08 · u          # at cube height, z = 0.02
```

**Rationale.** An arbitrary waypoint (Option B) spends up to 50 steps travelling
somewhere irrelevant and hands the push phase an arbitrary arm pose. Option A makes
reach *set up* the push, and it makes the two phases genuinely different regimes —
free-space servoing versus contact-rich manipulation. That difference is the whole
reason a phase-conditioned q̂ is meaningful rather than decorative: if both phases were
free-space motion, one threshold would do.

**Superseded:** the earlier `reach_goal ~ U(−0.15, 0.15)², z ~ U(0, 0.30)` reset is the
Option-B sampling and is no longer used. Under Option A the waypoint is derived and its
z is pinned to cube height, so reach is a table-plane approach task.

**Difficulty check** (`analysis/phase1_measurements.py`, `p_servo`, **N = 1,000**): a plain
P-servo hits the waypoint **100%** of the time in a median of 4 steps
(p95 6, max 7) against the 50-step budget. Gate 3 (reach ≥ 0.95) is not where
this project is at risk. The same run refutes the obvious worry about a 0.05 m success
radius around a z = 0.02 target: arrival `ee_z − cube_z` is median +0.017 m, max
+0.033 m, and **0%** of arrivals exceed 0.04 m — the policy cannot
satisfy the radius by hovering clear of the cube.

---

## 12. `PushOnly` reset — reproducing the hand-off

The push expert must start where reach leaves it, or Phase 4 composes two policies that
never met. Two things have to be reproduced, and they are not independent: **where the
cube ends up** (reach shoves it) and **where the arm ends up relative to it**.

All figures below: `analysis/phase1_measurements.py` → `results/mujoco_train/phase1_measurements.json`.

### 12.1 What reach actually does to the cube — N = 1,000

Measured by running the P-servo reach stand-in over 1,000 sampled task geometries and
recording the cube displacement in the approach frame (`u` = the approach ray, `n = u⊥`):

| | all 1,000 episodes | the 31.7% where the cube moved |
|---|---|---|
| along `u` | mean +0.0060, sd 0.0119 | mean +0.0188, sd 0.0144 |
| lateral `n` | mean -0.0000, sd 0.0065 | mean -0.0001, sd 0.0116 |
| vertical `z` | mean -0.0005, sd 0.0030 | mean -0.0014, sd 0.0053 |

**31.7% of reach episodes displace the cube by more than 1 mm** (25.0% by more than
1 cm, max 0.0765 m, median 0.0203 m when it moves at all). The waypoint sits only
8 cm behind a 4 cm cube, so the arm arrives through the cube's neighbourhood. The
remaining 68.3% carry the zero-displacement mass.

### 12.2 Disturbance and offset are resampled as a PAIR, not fitted

| correlation | along | lateral | vertical |
|---|---|---|---|
| `d` vs `o` | **-0.812** | -0.645 | -0.327 |

`offset_along` is **+0.0569** on average, but splits into
**+0.0682 when the cube was untouched** and
**+0.0324 when it was shoved** — the gripper pushes the cube ahead of itself and
closes its own stand-off. Drawing the disturbance and the offset from separate fitted
distributions would therefore manufacture hand-off poses reach never produces (a cube
shoved 5 cm with the gripper still 6.8 cm behind it).

So the reset does not fit either distribution. `analysis/phase1_measurements.py` saves
the **joint sample** — one row per reach episode,
`[d_along, d_lat, d_z, o_along, o_lat, o_z]` — to
`results/mujoco_train/reach_handoff_sample.npy` (**1,000 × 6**), and `envs.sample_handoff`
draws a whole row and rebuilds it in the new episode's approach frame. The zero mass
comes along for free; no mixture model is needed.

The **vertical component of the disturbance is dropped** (sd 0.0030 m): a training
reset must start from a settled cube, and the measured z excursion is transient bounce.
The offset keeps its z.

### 12.3 Placement: posture ladder, not rejection

The arm is placed at `cube + offset` by kinematic IK (`envs.place_ee`) — DLS on the `tcp`
site with `mj_forward` only, so the placement never disturbs the cube. A single solve from
the `NEUTRAL` posture is not enough:

| | one IK solve from NEUTRAL (n = 300) | shipped reset (n = 500) |
|---|---|---|
| cube↔robot overlap beyond −2 mm | **25.3%** (deepest -0.0199 m) | **0%** |
| cube ejected > 5 mm by a zero action | 6.7% | 0% |

**Rejecting and redrawing the pair is the wrong recovery, and it was measured to be
wrong.** The penetrating draws are precisely the disturbed ones — a shoved cube sits
closer to the gripper — so resampling on contact failure preferentially discards them:
it cut the reset's disturbance rate from 31.7% to **13.0%**, silently undoing the
reason §12.1 exists at all.

The recovery ladder instead keeps the drawn cube and offset and changes only *how the arm
gets there*:

1. **Re-seed the IK** from a jittered start posture (σ = 0.35 rad), up to 8 times. The arm
   is redundant, so the same EE target is reachable through many postures; measured, 3 of
   8 jittered seeds clear a collision the NEUTRAL seed could not, at identical EE position
   (residual ~5e-5 m). This preserves the disturbance **and** the offset exactly.
2. Only if all 8 fail, **retreat** the target along `unit(offset)` in 4 mm steps.

The residual failures are not the fingertips: they are `geom70`, the **hand body**
(0.032 × 0.049 × 0.104 m), clipping the cube from above while the tcp is still 8–11 cm
clear. Moving the target cannot fix that; re-posturing the arm can.

Shipped reset, measured over 500 seeds: **2.68** placements per reset (max 16),
**13.0%** need any retreat (mean 11.4 mm, max 28 mm), **0.0%** born
successful (min separation 0.0529 m), and the cube is disturbed in **29.2%** of
resets against reach's 31.7%.

**The one residual bias, stated:** the 13.0% of resets that need a retreat are
disproportionately the disturbed ones, and their `‖ee − cube‖` ends at 0.0427 m median
versus 0.0683 m for the rest. They still start closer than an undisturbed
hand-off, so the direction is right and the magnitude is ~1 cm; it is not corrected
further because doing so would mean moving the cube.

**Separation re-check after perturbing:** if the nudge alone would carry the cube inside
the success radius, the reset falls back to the undisturbed cube so the episode is still a
task instance. Measured firing rate: **0.0%** — the disturbance points away from
the goal, so this is insurance rather than a common path.

### 12.4 PROVISIONAL — Phase 4 must re-measure this against `ReachExpert`

Everything in §12 is fitted to a **P-servo**, not to `ReachExpert`, which does not exist
until Phase 3. `results/mujoco_train/reach_handoff_sample.npy` is the single re-fit point.
The Phase 4 plan carries a named, non-skippable checkpoint for this; see
`docs/superpowers/plans/2026-09-14-mujoco-split-envs.md`, "Phase 4 checkpoint".

---

## 13. Inference device — CPU, decided by measurement

**Every Phase-9 arm is timed batch-1 on CPU.** Measured by `analysis/inference_latency.py`
→ `results/mujoco_train/inference_latency.json`, on the seed-0 push actor
(541,190 actor parameters), 1,000 individually-timed calls after
100 warmup calls, with `torch.mps.synchronize()` bracketing the MPS timed region:

| device | mean | sd | p50 | p95 | min |
|---|---|---|---|---|---|
| **cpu** | **0.110 ms** | 0.014 | **0.106** | 0.130 | 0.096 |
| mps | 3.376 ms | 2.058 | 2.658 | 7.621 | 1.116 |

CPU is **25.0x faster at p50** for batch-1 inference. The ranking is
size-dependent and inverts: MPS wins on the *training* gradient step at batch 2048, which
is why training runs on MPS and inference is benchmarked on CPU. That inversion is the
hazard: the cheap policy will be ~8k parameters and the push actor is 541,190, so
timing the expert on MPS and the cheap policy on CPU would turn a device switch into an
apparent compute saving. Mixing devices across arms is therefore prohibited, and the
device must be stated in the Phase 9 report.

Note the MPS p95 of 7.621 ms against a mean of 3.376 ms: dispatch latency
dominates at batch 1 and is highly variable, which is the other reason it is unusable as
a comparison baseline.

---

## 14. Parameter accounting — actor, not total

`sum(policy.parameters())` for a TQC checkpoint counts the actor plus 2 critics plus 2
target critics. Only the **actor** runs at inference. For the push expert that is
**541,190 actor** against **2,751,082 total** — a 5.1x overstatement if the
wrong one is quoted.

Every eval JSON now carries `n_params_actor` and `total_params` as separate keys, and the
ambiguous `n_params` key has been removed (`analysis/fix_param_accounting.py`, idempotent).
**Phase 9's compute ratio uses `n_params_actor`.**
