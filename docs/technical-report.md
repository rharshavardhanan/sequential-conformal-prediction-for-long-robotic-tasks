# PROJECT_1 — Complete Technical Report

Built by reading the actual source, not filenames. Every claim below is traced to a file and
line. Where something cannot be established from the repository it is marked
**NOT DETERMINABLE FROM CODE**.

Inspected: 5 Aug 2026. No files modified except this one.

---

# PART I — EXECUTION TRACE

## 0. Repository layout and the two-environment split

The repo has **two Python environments that cannot coexist**, and this is the single most
important structural fact:

| venv | numpy | purpose | scripts |
|---|---|---|---|
| `.venv` | 1.23.5 | **training** LIBERO's own BC policies | `common.py`, `train_single_task.py`, `train_all_tasks.py`, `eval_checkpoint.py`, `profile_train_step.py`, `watch_training.py`, `check_*.py` |
| `.venv-lerobot` | 2.2.6 | **everything currently active** — rollouts + monitor | `dp_policy.py`, `collect_rollouts.py`, `subphases.py`, `cp_monitor.py`, `verify_spec.py`, `show_rollout.py`, `eval_pretrained_dp.py`, `inspect_dataset.py` |

`robomimic 0.2.0` (needed for LIBERO training) uses `np.bool`, removed in numpy 1.24;
`lerobot 0.4.4` requires numpy 2.x. Hence the split (`pyproject.toml:14-18`).

**The training half of the repo is currently dormant.** Nothing in `.venv` is on the active path.

---

## 1. COLLECTION PIPELINE — full trace

### 1.1 Entry point

**File:** `scripts/collect_rollouts.py` · **function:** `main()` (line 40)
**Invocation:** `.venv-lerobot/bin/python -m scripts.collect_rollouts --task-id 0 --n-rollouts 100`

### 1.2 LIBERO task loading

```python
# collect_rollouts.py:70-72
suite = benchmark.get_benchmark_dict()["libero_10"]()
task  = suite.get_task(args.task_id)
bddl  = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
```

- **Input:** integer `task_id` (0–9)
- **Output:** a `Task` object carrying `.name`, `.language`, `.problem_folder`, `.bddl_file`,
  `.init_states_file`
- `get_libero_path()` reads `~/.libero/config.yaml`, written by `scripts/bootstrap.sh`
- Task order is `[0..9]` (`task_order_index=0`), so `task_id` indexes the benchmark's own order,
  **not** alphabetical filename order

### 1.3 Policy loading

**File:** `scripts/dp_policy.py` · **class:** `DiffusionPolicyRunner.__init__` (line 38)

```python
self.policy = DiffusionPolicy.from_pretrained(repo).to(device).eval()   # line 46
raw = load_file(hf_hub_download(repo, "model.safetensors"))             # line 48
S = {k: v for k, v in raw.items() if k.startswith(("normalize_", "unnormalize_"))}
```

Default repo (line 64): `yzembodied/libero10_task{id}_diffusion`.

**Critical non-obvious step.** `from_pretrained()` alone is insufficient. These checkpoints are
pre-0.4 LeRobot format storing normalization statistics *inside* `model.safetensors` as
`normalize_inputs.*` / `unnormalize_outputs.*`. lerobot 0.4.4 moved to external processor files
and **discards those keys on load**. The code therefore re-opens the safetensors and extracts
seven tensors by hand (lines 57–64):

```
normalize_inputs.buffer_observation_images_image.mean / .std   (image MEAN_STD)
normalize_inputs.buffer_observation_state.min / .max           (state MIN_MAX)
normalize_inputs.buffer_observation_state_joint.min / .max     (joint MIN_MAX)
unnormalize_outputs.buffer_action.min / .max                   (action MIN_MAX)
```

Line 54 raises if these are absent — a fail-loud guard against silently running an unnormalized
policy.

### 1.4 Init-state loading

```python
# collect_rollouts.py:78-86
init_name = task.init_states_file                      if --init-file pruned  (50 states)
            task.init_states_file.replace(".pruned_init", ".init")   if full  (100 states)
init_states = torch.load(..., weights_only=False)      # shape (100, 123) for task 0
```

Rollout `i` uses `init_states[i % n]` (line 129). With `--init-file full` (the default) and
≤100 rollouts, **every rollout gets a distinct start**. Line 89 warns if `n_rollouts` exceeds
the state count.

### 1.5 Environment construction

```python
# collect_rollouts.py:95
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
```

256×256 because the policy's `input_features` declare `[3, 256, 256]`. LIBERO's own demos are
128×128 — **the env is deliberately rendered at a different resolution than the demo data**, to
match the policy.

### 1.6 One rollout, step by step

```python
# collect_rollouts.py:128-182
env.reset()
obs = env.set_init_state(init_states[i])
for _ in range(5): obs,_,_,_ = env.step(np.zeros(7))   # settle physics
runner.reset()                                          # clear the 5-frame obs queue

while steps < 600 and not success:
    eef_pos = obs["robot0_eef_pos"].copy()              # position BEFORE acting
    act     = runner.act(obs)                           # 7-dim, [-1,1]

    cmd     = act[:3] * 0.05                            # POS_SCALE
    planned = eef_pos + cmd

    ... record ...
    obs, _, success, _ = env.step(act)                  # <-- env advances

    executed = obs["robot0_eef_pos"].copy()             # position AFTER acting
    cmd_n    = ||cmd||
    err      = ||planned - executed||
    err_norm = err / max(cmd_n, 1e-8)
```

**Inside `runner.act()`** (`dp_policy.py:92`):

```
obs dict
 → build_state(obs) = concat[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]  = 8-dim
 → images flipped vertically (a[::-1]), HWC→CHW, /255, (x-mean)/std
 → state & joint MIN_MAX-normalized to [-1,1]
 → policy.select_action(batch)          # returns normalized action
 → minmax_denorm(a, act_lo, act_hi)     # back to LIBERO's native [-1,1]
 → np array (7,)
```

The image flip (`FLIP_IMAGES = True`, line 24) is mandatory: LIBERO stores `agentview`
bottom-up (`macros_image_convention: opengl`) and the LeRobot conversions flipped it. The
comment records that unflipped rollouts score 0/20.

`select_action` internally buffers `n_obs_steps=5` frames and emits `n_action_steps=8` actions
per inference from a 16-step horizon — so the diffusion UNet runs roughly every 8th control
step, not every step.

### 1.7 Success determination

`env.step` returns `done`, and this is bound directly to `success` (line 173).

Tracing it: `bddl_base_domain.py:807` → `done = self._check_success()`, overridden per problem
class, e.g. `libero_living_room_tabletop_manipulation.py:136`:

```python
goal_state = self.parsed_problem["goal_state"]
result = True
for state in goal_state:
    result = self._eval_predicate(state) and result
return result
```

**Success = conjunction of the BDDL goal predicates, evaluated every step.** This is a real
task-completion check, not a proxy. A rollout is a failure iff it hits the 600-step cap without
the predicates ever all holding — which is why every observed failure has `steps == 600`.

### 1.8 HDF5 write

One file per task, `rollouts/task{id}.hdf5`, one group per rollout, written **immediately after
each rollout** (line 185) so a crash keeps prior work; re-running skips existing groups
(line 125–126).

**File-level attributes:**

| attr | example |
|---|---|
| `policy_repo` | `yzembodied/libero10_task0_diffusion` |
| `policy_sha256` | `3105133cc3bddf79…` |
| `task_id`, `task_name`, `task_language`, `bddl_file` | |
| `pos_scale` | 0.05 |
| `rot_scale` | 0.5 |
| `control_freq` | 20 |
| `max_steps` | 600 |
| `init_states_file` | `LIVING_ROOM_SCENE2_….init` |
| `osc_gain_reference` | 0.251 |

**Per-rollout datasets** (T = episode length):

| dataset | shape | contents |
|---|---|---|
| `actions` | (T, 7) | raw policy output, `[-1,1]` |
| `eef_pos` | (T, 3) | eef position **before** each action |
| `eef_quat` | (T, 4) | eef orientation before each action |
| `gripper_qpos` | (T, 2) | finger joint positions |
| `joint_pos` | (T, 7) | arm joint angles |
| `planned_pos` | (T, 3) | `eef_pos + action[:3]*0.05` |
| `executed_pos` | (T, 3) | eef position **after** the step |
| `err_raw` | (T,) | metres |
| `err_norm` | (T,) | dimensionless |
| `cmd_norm` | (T,) | metres |
| `object_state` | (T, D) | D = 112 (task 0), 28 (task 8) — task-dependent |

**Per-rollout attributes:** `success` (bool), `steps`, `init_state_index`, `wall_sec`,
`policy_sha256`.

Optional `images/agentview`, `images/eye_in_hand` under `--images` (default **off**),
downsampled `[::2, ::2]` to 128px and flipped.

### 1.9 The three error quantities — exact definitions

Let `a_t` be the action at step *t*, `p_t` the measured eef position before it, `p_{t+1}` after.

```
cmd_t       =  a_t[0:3] * 0.05                          # metres, the commanded delta
cmd_norm_t  =  ‖cmd_t‖₂
planned_t   =  p_t + cmd_t
executed_t  =  p_{t+1}
err_raw_t   =  ‖planned_t − executed_t‖₂                # metres
err_norm_t  =  err_raw_t / max(cmd_norm_t, 1e-8)        # dimensionless
```

**Why `0.05`:** the OSC_POSE controller's `output_max` is `[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`
(`action_space.json`), and robosuite's `set_goal_position(scale_action(delta), self.ee_pos)`
re-anchors the setpoint to the *measured* pose each step. So `planned` is the controller's
actual internal goal, not an approximation.

**Why `err_norm` exists:** OSC is compliant — realised motion is a near-constant 0.251 of the
commanded delta. So `err_raw ≈ 0.749 × cmd_norm`, i.e. mostly a readout of commanded speed.
Empirically in `rollouts/task0.hdf5`, `err_raw` runs **backwards**: successes average 0.0203 m,
failures 0.0130 m, because successes move fast and failures creep. `err_norm` corrects the sign.

> **Gap found:** `ROT_SCALE = 0.5` is defined (line 36) and written as an attribute (line 110)
> but **no rotational error is ever computed**. `planned_rot` does not exist anywhere in the
> repo. The signal is position-only. Orientation tracking error is unmeasured.

> **Gap found:** `object_state` is written but **never read by any script**. It is stored for
> future offline BDDL predicate evaluation; today it is dead data (~30 % of file size).

---

## 2. PHASE DETECTION — `scripts/subphases.py`

### 2.1 What it does

**Function:** `segment(gripper_cmd, window=5)` (line 32)
**Input:** `(T,)` array — in practice `actions[:, 6]`
**Output:** `labels (T,) int8` indexing `PHASES = ["reach","grasp","transport","release"]`, and
a list of `(start, end, phase_name)` segments.

### 2.2 The algorithm, exactly

```python
closed = g > 0                                    # bang-bang: exactly -1 or +1
labels = where(closed, transport, reach)          # baseline
trans  = diff(closed.astype(int8))                # +1 = closes, -1 = opens
for t in nonzero(trans):
    lo, hi = max(0, t-window+1), min(T, t+window+1)
    labels[lo:hi] = grasp if trans[t] > 0 else release
```

So:
- **reach** = gripper open, not near a transition
- **grasp** = 10-step window straddling an open→close event (5 before, 5 after)
- **transport** = gripper closed, not near a transition
- **release** = 10-step window straddling a close→open event

### 2.3 Regrasp detection

Not in `subphases.py`. It lives in `verify_spec.py:100-102`:

```python
closes = ((grip[1:] > 0) & (grip[:-1] <= 0)).sum()   # count of open->close transitions
# >2 for a 2-object task implies a regrasp
```

### 2.4 ⚠️ The phase labelling is NON-CAUSAL

The `grasp`/`release` window extends **backwards** from the transition (`lo = t-window+1`), so
the label at step *t−4* depends on an event at step *t*. Verified empirically: for a single
transition, **5 of 11 steps around it receive a different label from prefix-only data than from
the full trajectory.**

This is fine for offline analysis. It **invalidates the "sequential monitoring" framing** as
implemented — a deployed monitor cannot know at time *t* that a grasp begins in 5 steps.
See §11 (risks).

---

## 3. MONITOR PIPELINE — `scripts/cp_monitor.py`

### 3.1 Load

**Function:** `load(path, score_key)` (line 43) → list of dicts
`{name, success, score=(T,), labels=(T,)}` plus file attrs. Phase labels are recomputed on load
via `segment(g["actions"][:,6])` — **not read from the HDF5**, so labelling changes propagate to
old data automatically.

### 3.2 Score aggregation

**Function:** `phase_max(r)` (line 63)

```python
out[i] = r["score"][labels == i].max()      # MAX within each phase; nan if phase absent
```

**⚠️ The implementation uses `max`, not `p95`.** The A2 analysis recommended p95 (AUC 1.00 vs
0.88); that change is **NOT implemented**.

**⚠️ No A4 floor is applied here.** `verify_spec.py` excludes near-zero-motion steps before
taking maxima; `cp_monitor.py` does not. The thresholds in `cp_task0.json` were therefore
computed on maxima that A4 showed to be 19 % contaminated.

### 3.3 Calibration/evaluation split

```python
# cp_monitor.py:102-106
rng = np.random.RandomState(args.seed)          # seed default 0
idx = rng.permutation(len(succ))
ncal = max(1, int(len(succ) * args.cal_frac))   # cal_frac default 0.5
cal, test_ok = succ[idx[:ncal]], succ[idx[ncal:]]
```

- Only **successful** rollouts are split. All failures go to evaluation.
- **The split is regenerated at load time, not stored.** It is reproducible only if the seed
  *and the exact rollout set* are unchanged — appending rollout 21 silently reshuffles which
  rollouts calibrate.
- Initial states cannot be shared between cal and eval, because rollout *i* ↔ init state *i* is
  a bijection under `--init-file full`.

### 3.4 Conformal calibration — the mathematics

**Function:** `conformal_quantile(scores, alpha)` (line 32)

```python
n    = len(scores)
rank = ceil((n + 1) * (1 - alpha))
if rank > n:  return inf, rank, False        # INFEASIBLE
return sort(scores)[rank - 1], rank, True
```

This is the standard split-conformal quantile. For exchangeable calibration scores
`S₁…S_n` and a new score `S_{n+1}`,

```
P( S_{n+1} ≤ Ŝ ) ≥ 1 − α      where Ŝ = the ⌈(n+1)(1−α)⌉-th order statistic
```

Feasibility requires `⌈(n+1)(1−α)⌉ ≤ n`, i.e. **n ≥ 1/α − 1**:

| α | min n |
|---|---|
| 0.20 | 4 |
| 0.10 | 9 |
| 0.05 | 19 |
| 0.025 | 39 |

Below that the threshold is `+∞` — an alarm that can never fire while reporting perfect
coverage. Lines 133–139 detect and report this rather than shipping a silent no-op.

### 3.5 Multiple-testing correction

```python
alpha_p = alpha / len(PHASES) if correction == "bonferroni" else alpha   # line 108
```

Default is **bonferroni**, so α = 0.1 → α_phase = 0.025 → 39 calibration rollouts per phase.

### 3.6 Alarm rule

```python
# cp_monitor.py:141-145
def first_alarm(r):
    for t in range(len(r["score"])):
        if r["score"][t] > thresholds[r["labels"][t]]:
            return t
    return None
```

Scans forward; alarms at the first step whose score exceeds its phase's threshold.

**This is internally consistent with max-calibration:** since `max_phase(S) > q` ⟺
`∃t in phase: S_t > q`, calibrating the per-rollout max and alarming per-step yield the same
event. With Bonferroni over 4 phases the union bound gives
`P(any alarm | nominal) ≤ 4 × α/4 = α`. **The per-rollout guarantee is correctly constructed.**

### 3.7 What happens after an alarm

**Nothing.** `first_alarm` returns an index used only for metrics. There is no intervention,
no stopping, no escalation. The monitor is an offline evaluator, not a controller.

### 3.8 Metrics reported

```
false alarm rate = fraction of HELD-OUT SUCCESSES that alarm
detection rate   = fraction of FAILURES that alarm
median lead time = median over detected failures of (len(score) − t_alarm)
```

**⚠️ Lead time is measured from *episode end*, not from failure onset.** Failures are 600-step
timeouts; the true moment the run became unrecoverable is unknown and unlabelled. "480 steps of
lead time" means "alarmed 480 steps before the clock ran out", which is a much weaker claim than
"alarmed 24 s before failure".

### 3.9 `--per-step` baseline

Lines 166–178 pool all calibration timesteps, take one global conformal quantile, and alarm if
any step exceeds it. This deliberately reproduces the wrong thing (α bounds one test, not 600)
and measured FAR = 1.00 — every nominal rollout alarms.

---

## 4. `scripts/verify_spec.py` — what it actually tests

Standalone (does not import `cp_monitor`). Reimplements its own `phase_max` **with** the A4
floor (line 187) — hence the divergence from `cp_monitor.py`.

Helper functions:

| function | line | purpose |
|---|---|---|
| `auc(pos, neg)` | 20 | Mann-Whitney AUC, ties 0.5 — `P(failure score > success score)` |
| `ks_2samp(a, b)` | 30 | two-sample KS with asymptotic p — hand-rolled, no scipy |
| `occurrences(labels, id)` | 46 | contiguous runs of one phase, for A3's first-vs-second |

Constant `MAX_DELTA = 0.05·√3 = 0.08660` — the largest commandable position step.

---

# PART II — REQUESTED SECTIONS

# 1. PROJECT IN ONE SENTENCE

Build a runtime monitor that flags a failing robot manipulation episode **while it is still
running**, using stage-specific conformal thresholds on the gap between the controller's
commanded end-effector setpoint and the pose actually achieved — so the alarm carries a
distribution-free bound on its false-alarm rate rather than a hand-tuned number.

**The research contribution is the *stage-aware* part.** Split conformal prediction on a
trajectory-level score is standard. Segmenting an episode into reach / grasp / transport /
release and calibrating a *separate* threshold per stage is the novel claim, motivated by the
observation that a single global threshold is simultaneously too loose for delicate phases and
too tight for fast ones.

---

# 2. PROJECT IN PLAIN ENGLISH

**What the robot is doing.** A simulated Franka arm in LIBERO performs long two-part chores —
"put both the alphabet soup and the tomato sauce in the basket". Each attempt takes 250–600
control steps at 20 Hz.

**What the diffusion policy is doing.** It is the robot's brain, downloaded not built. Every few
steps it looks at two camera images plus joint/gripper state and denoises its way to a chunk of
8 future actions. Each action says "move the hand this much from where it is now, and open or
close the gripper". It succeeds about half the time.

**What our monitor is doing.** At every step it compares two numbers: where the controller *told*
the hand to go, and where the hand actually got to. It divides that gap by how far the hand was
told to move — because a compliant controller always undershoots proportionally, so the raw gap
mostly measures speed rather than trouble. That ratio is the health signal.

**What conformal prediction is doing.** It converts "here is what the ratio normally looks like"
into "alarm above this threshold, and I can prove this alarm fires on at most 10 % of healthy
runs". It is not a neural network and it learns nothing. It sorts the healthy runs' scores and
picks an order statistic. The proof needs only *exchangeability* — that calibration and test
runs come from the same process — which is why the policy weights are hashed into every file.

---

# 3. COMPLETE SYSTEM ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  OFFLINE ASSETS                                                                │
│                                                                                │
│  LIBERO repo (third_party/)          HuggingFace                               │
│   ├── bddl_files/libero_10/*.bddl     ├── yzembodied/libero10_task{0,1,2}_…    │
│   │      goal predicates              │   yzembodied/libeo10_task{3,8}_…       │
│   ├── init_files/*.init  (100×123)    │      1.16 GB DiffusionPolicy each      │
│   └── datasets/libero_10/*.hdf5       └── trained 80k steps on                 │
│          50 human demos/task              yzembodied/libero_10_image_task_N    │
│          (NOT used at runtime)            (LeRobot conversion of those demos)  │
└───────────────┬────────────────────────────────┬───────────────────────────────┘
                │                                │
                ▼                                ▼
        ┌───────────────┐              ┌──────────────────────┐
        │ LIBERO env    │              │ DiffusionPolicyRunner│
        │ OffScreen     │◄────obs──────│  dp_policy.py        │
        │ 256×256, 20Hz │              │  flip img, MIN_MAX   │
        │ OSC_POSE      │────action───►│  select_action()     │
        └───────┬───────┘              └──────────────────────┘
                │  done = ∧ BDDL goal predicates
                ▼
   ┌────────────────────────────────────────────────┐
   │  collect_rollouts.py    per step t:            │
   │    cmd      = a[0:3]·0.05                      │
   │    planned  = eef_pos[t] + cmd                 │
   │    executed = eef_pos[t+1]                     │
   │    err_raw  = ‖planned − executed‖             │
   │    err_norm = err_raw / ‖cmd‖                  │
   └───────────────────┬────────────────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │ rollouts/task{id}.hdf5           │
        │  rollout_0000 … rollout_NNNN     │
        │   11 arrays + success/steps/     │
        │   init_state_index/policy_sha256 │
        └──────────┬───────────────────────┘
                   │
     ┌─────────────┴─────────────┬───────────────────────┐
     ▼                           ▼                       ▼
┌──────────────┐        ┌─────────────────┐     ┌──────────────────┐
│ subphases.py │        │ verify_spec.py  │     │ show_rollout.py  │
│ segment()    │        │ A1 A2 A3 A4 A6  │     │ inspect / CSV    │
│ from grip cmd│        │ (own floored    │     └──────────────────┘
└──────┬───────┘        │  phase_max)     │
       │                └─────────────────┘
       ▼
┌────────────────────────────────────────────────────────────┐
│ cp_monitor.py                                              │
│                                                            │
│  successes ──RNG(seed)──► calibration (50%)                │
│                       └─► held-out nominal (50%)           │
│  failures ───────────────► evaluation                      │
│                                                            │
│  per phase k:  S_k = MAX err_norm over steps in phase k    │
│                q_k = ⌈(n+1)(1−α/4)⌉-th order statistic     │
│                      (+∞ if rank > n)                      │
│                                                            │
│  alarm at first t with err_norm[t] > q_{phase(t)}          │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
        false alarm rate │ detection rate │ lead time
                           ▼
                    cp_task0.json
                    (no intervention is taken)
```

---

# 4. MODELS

## Pretrained models we ARE using

| repo | tasks | what |
|---|---|---|
| `yzembodied/libero10_task{0,1,2}_diffusion` | 0, 1, 2 | LeRobot `DiffusionPolicy` |
| `yzembodied/libeo10_task{3,8}_diffusion` | 3, 8 | same (note the `libeo` typo in the repo name) |

Architecture, identical across all five (from each repo's `config.json`):
`type=diffusion`, `vision_backbone=resnet18`, `n_obs_steps=5`, `n_action_steps=8`,
`horizon=16`, `num_train_timesteps=100`, normalization
`{VISUAL: MEAN_STD, STATE: MIN_MAX, ACTION: MIN_MAX}`, ~1.16 GB each.

These are **frozen**. Loaded, `.eval()`, wrapped in `@torch.no_grad()`. Their weights are
SHA-256'd (`dp_policy.py:111`) into every rollout.

## Models we are TRAINING

**None. Zero. Nothing is being trained.**

## Models we are NOT training (but the repo can)

`scripts/train_single_task.py` / `train_all_tasks.py` can train LIBERO's **BC-Transformer**
(5.4 M params). One run reached 35 % success at epoch 10 and then deadlocked in
`MPSStream::synchronize`. That checkpoint exists on disk but is **not used by anything on the
active path**. This half of the repo is dormant.

## Statistical methods that are NOT neural networks

Everything in `cp_monitor.py`, `subphases.py`, `verify_spec.py`:

- **Split conformal prediction** — sorting and an order statistic. No parameters, no gradients,
  no fitting.
- **Gripper-transition segmentation** — a sign change and a ±5 window.
- **Mann-Whitney AUC**, **two-sample KS test**, **Bonferroni correction**.

> **If you remember one thing:** the diffusion policy is a neural network someone else trained.
> The conformal monitor is arithmetic on sorted numbers. They share no code and no training.

---

# 5. DATASETS

| | LIBERO demonstration data | Our rollout data |
|---|---|---|
| **path** | `third_party/LIBERO/libero/datasets/libero_10/*.hdf5` | `rollouts/task{0,8}.hdf5` |
| **size** | 13.7 GB, 10 files | 6.1 MB + 5.4 MB |
| **contents** | 50 human teleop demos/task; 128×128 dual camera, actions, states | 20 + 30 policy rollouts; 11 arrays/step, no images |
| **generated by** | LIBERO authors (human teleoperation) | **us**, `collect_rollouts.py`, this week |
| **trains the policy?** | **Yes — indirectly.** The LeRobot conversion `yzembodied/libero_10_image_task_N` (re-rendered 256×256) is what the checkpoints were trained on, 80k steps @ batch 32 | **No** |
| **used for calibration?** | No | **Yes** — successful rollouts |
| **used for evaluation?** | No | **Yes** — failures + held-out successes |
| **used at runtime?** | Only `bddl_files/` (goal predicates) and `init_files/` (start states) | — |

Also present:

| file | contents |
|---|---|
| `action_space.json` | machine-readable OSC config: `output_max`, `pos_scale_m: 0.05`, `rot_scale_rad: 0.5` |
| `verification_task0.json` | full A1–A6 outputs |
| `cp_task0.json` | thresholds + FAR/TPR/lead for the α=0.2 run |
| `pretrained_dp_task{0,1,2,3,8}.json` | success-rate summaries (pass/fail only — **no per-step data**) |

> The 13.7 GB of human demos is **not** used by the current pipeline at all, except that
> `subphases.py --demo` reads it to validate the segmenter. It matters only if training resumes.

---

# 6. A1–A6 EXPLAINED

All results from `verification_task0.json` (20 rollouts, task 0).

### A1 — Do phase labels survive real policy rollouts?

- **Question:** the segmenter was validated on tidy human demos. Do fumbling policy rollouts
  still segment consistently?
- **Why:** if segment structure varies wildly, per-phase calibration compares incomparable things.
- **Code:** `verify_spec.py:98-142`, inner `a1()`
- **Acceptance:** ≥90 % of successes give exactly 8 segments, no empty phase
- **Result:** **0.00** give exactly 8 (modal 9; range 7–14 successes, 7–34 failures).
  **0 rollouts have an empty phase.** Regrasps 2.36 (success) vs 3.67 (failure) mean.
- **Verdict:** ❌ **FAILED** — but the criterion is wrong. Human demos average 9.5 segments, so
  8 was never attainable. The underlying concern is satisfied: every rollout visits all 4 phases.
- **Decision:** do not force 8. Add regrasp count as a feature (discriminative, currently unused).

### A2 — Does the phase-MAX separate success from failure?

- **Question:** separation was shown on the per-rollout *mean*. CP uses the per-phase *max*.
- **Why:** max is noisier; mean separation does not imply max separation.
- **Code:** `verify_spec.py:194-245`
- **Acceptance:** ≥1 phase with clean/near-clean separation (AUC ≥ 0.90)
- **Result:** reach **0.88**, grasp **0.43** (below chance), transport 0.63, release 0.56
- **Verdict:** ❌ **FAILED**
- **Decision:** switch aggregation. Sweep showed reach at **p95 → AUC 1.00**, and whole-rollout
  p95 → 1.00 with successes [0.831, 0.935] vs failures [1.001, 1.015]. **Not yet implemented.**

### A3 — Is pooling the two occurrences of each stage justified?

- **Question:** K=4 pools "reach for object 1" with "reach for object 2". Legitimate?
- **Why:** K=8 would double the Bonferroni penalty and the calibration requirement.
- **Code:** `verify_spec.py:247-292`, using `occurrences()` + `ks_2samp()`
- **Acceptance:** KS p > 0.05 for every stage
- **Result:** reach 0.374, grasp 0.147, transport 0.147, release 0.675
- **Verdict:** ✅ **PASSED** — keep K = 4
- **Caveat:** n ≈ 11 pairs; this is "no evidence against", not proof. Two stages sit at 0.147.

### A4 — Is `err_norm` unstable when the robot barely moves?

- **Question:** `err_norm` divides by `cmd_norm`. Near-zero denominators produce spikes.
- **Why:** one spurious spike dominates a per-phase maximum.
- **Code:** `verify_spec.py:144-191`
- **Acceptance:** no phase max originates from a near-zero-motion step
- **Result:** `cmd_norm` 1st pct = 0.001893 (2.19 % of max). **15 of 80 phase maxima (18.8 %)
  came from steps below the 2 % floor.**
- **Verdict:** ❌ **FAILED without a floor**; passes with `cmd_norm ≥ 0.02·MAX_DELTA = 0.001732`
- **Decision:** this is the *cause* of A2's failure. Floor is applied in `verify_spec.py` only —
  **`cp_monitor.py` does not have it.**

### A5 — Is task 8 actually usable?

- **Question:** excluded at 17/20 = 85 %, but CI ±0.16 straddles the 80 % ceiling.
- **Why:** task 8 is the only candidate outside `LIVING_ROOM_SCENE2` — it decides whether the
  study has one scene or two.
- **Code:** no dedicated script; `collect_rollouts.py --task-id 8` plus manual CI
- **Acceptance:** point estimate ≤ 0.85 with CI overlapping 0.80
- **Result (partial):** 24/30 = 0.800 new; combined with the earlier 20 → **41/50 = 0.820,
  95 % CI [0.714, 0.926]**
- **Verdict:** ⏳ **PARTIAL (30/60) — but both conditions already met.** The collection process
  died after 30 rollouts.
- **Decision:** include task 8.

### A6 — Can calibration and evaluation share initial states?

- **Question:** shared starts between cal and eval would inflate the guarantee.
- **Code:** `verify_spec.py:294-323`
- **Acceptance:** zero overlap **and** the split is recorded
- **Result:** 20 rollouts, 20 distinct init states, 0 repeats, 0 success/failure overlap.
  **`split_recorded_in_hdf5: false`.**
- **Verdict:** ⚠️ **PARTIAL** — disjointness holds by construction; persistence does not.
- **Decision:** assign split by init index at collection time and store it.

---

# 7. B1 / B2 EXPLAINED

**The problem both solve:** of five tasks with usable checkpoints, three are *too good* —
task 2 at 100 %, task 3 at 95 %, task 8 at 82 %. A policy that never fails gives the detector
nothing to detect. You need roughly 40–80 % success.

### B1 — induce failures without training

Take the existing validated checkpoints and make the task harder, via either:

1. **initial object-pose perturbation** — noise on object placement at reset
2. **action noise** — Gaussian noise on the action before `env.step`

Sweep 3–4 magnitudes × 20 rollouts, target 50–70 %.

**Hard constraint:** the chosen perturbation must apply to *every* rollout of that task,
calibration and evaluation alike. Calibrating on clean rollouts and evaluating on perturbed ones
breaks exchangeability and silently voids the conformal guarantee.

**Implemented?** ❌ **No.** `collect_rollouts.py` has no perturbation argument of any kind.

### B2 — train our own policies

Only if B1 cannot produce enough in-band tasks. Would train diffusion policies through LeRobot
(not BC-Transformer) to keep one policy class and preserve the FAIL-Detect baseline.

**Prerequisite:** `train_all_tasks.py` currently keeps only LIBERO's best-success checkpoint,
which reproduces the exact saturation problem. All eval checkpoints must be retained so success
rate is selectable per task.

**Implemented?** ❌ **No** — training scaffolding exists but targets BC-Transformer, and the
retention fix is not made.

**What decides B2:** if B1 yields ≥4 in-band tasks, B2 never happens. Currently 0 (in band) + 1
(task 8) + 2 (tasks 0, 1) = 3 confirmed; B1 needs to recover one more.

---

# 8. CURRENT PROJECT STAGE

```
DONE  ████████████████████████  ~90% of infrastructure
  ├── LIBERO on macOS, 7 upstream bugs patched
  ├── 13.7 GB libero_10 demos downloaded
  ├── action space confirmed 3 ways (config, osc.py source, live err_norm ≈ 0.76)
  ├── OSC compliance gain measured (0.251 ± 0.006)
  ├── policy adapter working (normalization + image flip)
  ├── 5 tasks characterised for success rate
  ├── collection pipeline (incremental, hashed, resumable)
  ├── phase segmentation
  ├── conformal monitor (quantile verified against empirical coverage)
  └── A1, A2, A3, A4, A6 executed and reported
        ↓
CURRENTLY WORKING  ██████░░░░░░  ~50%
  ├── A5 stalled at 30/60 (process died ~14 h ago) — verdict already met
  └── two method fixes identified but NOT written: p95 aggregation, A4 floor
        ↓
NEXT  ░░░░░░░░░░░░  0%
  ├── finish A5 (~1.5 h, resumes at 30)
  ├── implement p95 + floor in cp_monitor.py
  ├── implement split persistence
  └── B1 action-noise sweep on tasks 2 and 3
        ↓
LATER  ░░░░░░░░░░░░  0%
  └── final task list fixed (target 4 tasks, ≥2 scenes)
        ↓
FINAL EXPERIMENT  ░░░░░░░░░░░░  0%
  ├── ~150 rollouts × 4 tasks (~16 h, unattended, no GPU)
  ├── calibrate at α = 0.1 with Bonferroni (feasible: 41 ≥ 39)
  └── FAIL-Detect baseline comparison
        ↓
PAPER  ░░░░░░░░░░░░  0%
```

**Overall: infrastructure ~90 %, science ~10 %.** No experiment of publishable size has run.

---

# 9. WHAT WE SHOULD DO NEXT

1. **Restart A5.** `collect_rollouts.py --task-id 8 --repo yzembodied/libeo10_task8_diffusion
   --n-rollouts 60` — resumes at rollout 30 automatically. ~1.5 h unattended.
2. **Add the A4 floor to `cp_monitor.py`.** Mask `cmd_norm < 0.02·MAX_DELTA` before aggregating.
   Every threshold currently in `cp_task0.json` is computed on contaminated data.
3. **Add `--agg {max,p95,mean}` to `cp_monitor.py`, default `p95`.** This is the single change
   that turns AUC 0.88 into 1.00.
4. **Re-run the monitor on task 0** with (2) and (3) and compare against `cp_task0.json`.
5. **Persist the calibration/evaluation split.** Assign by init index (even → cal, odd → eval)
   at collection time, store as a per-rollout attribute; drop the RNG in `cp_monitor.py`.
6. **Add `--action-noise SIGMA` to `collect_rollouts.py`**, stored per rollout. This is B1's
   simpler knob and unblocks the saturated tasks.
7. **Run the B1 sweep** on task 2 and task 3: σ ∈ {0.05, 0.1, 0.2, 0.3} × 20 rollouts, find the
   setting landing in 50–70 %.
8. **Fix the task list** at 4 tasks with ≥2 scenes and decide the FAIL-Detect image subset —
   `--images` must be set at collection time; retro-fitting costs a full re-run.
9. **Run the full collection**, ~150 rollouts/task.
10. **Re-run A3** on the full data (current KS has n≈11 and low power) and then the final
    calibration at α = 0.1 with Bonferroni.

**Deliberately not on this list:** renting a GPU, training anything, adding rotational error,
using `object_state`. None are needed for the current claim.

---

# 10. BIGGEST SCIENTIFIC RISKS

## What the project currently proves

On **one task**, with **20 rollouts**:

- The planned-vs-executed ratio separates successes from failures with **zero overlap** on the
  per-rollout mean (successes [0.753, 0.770], failures [0.814, 0.903]) and on p95.
- Gripper-event segmentation recovers task structure with no hand labelling.
- Per-phase conformal thresholds differ by **2.7×** — evidence a global threshold is inadequate.
- The pipeline runs end-to-end and produces alarms.

## What it does NOT prove

1. **It does not prove the guarantee holds.** Measured FAR was **0.67 against a 0.20 target**.
   The guarantee is violated in the only run performed. This is a sample-size artefact
   (5 calibration rollouts), but it is unproven, not proven.
2. **It does not prove stage-awareness helps.** No comparison against a well-calibrated *global*
   threshold has been run. Different thresholds per phase ≠ better detection. **This is the
   headline claim and it is currently untested.**
3. **It does not prove generality.** One task, one scene, one policy, one failure mode.
4. **It does not prove early detection.** Lead time is measured to *episode end*, not to failure
   onset — which is unlabelled. "480 steps early" may be an artefact of failures being timeouts.

## What could invalidate the final claim

- **Failure-mode monoculture.** Per-phase AUC: reach 0.88, grasp 0.43, transport 0.63,
  release 0.56. **Essentially all signal is in `reach`.** This policy fails by never acquiring
  the object. A drop-during-carry failure would load onto `transport`, and there is zero
  evidence the signal catches that. If all four tasks fail the same way, the paper measures one
  failure mode and calls it a method.
- **Non-causal phase labels.** §2.4: the `grasp`/`release` window extends backwards, so labels
  at time *t* depend on events at *t+5*. **A "sequential monitor" that needs the future is not a
  sequential monitor.** Either shrink the window to be causal, accept a 5-step lag, or reframe
  the contribution as offline post-hoc analysis.
- **Exchangeability under perturbation (B1).** If action noise is applied to make a task fail
  more, and calibration/evaluation regimes differ at all, the guarantee is void — silently.
- **The p95 reinterpretation.** Switching from max to p95 is statistically valid (CP holds for
  any score), but it changes the claim from "no timestep was unsafe" to "the error distribution
  shifted". That must be stated, not glossed.

## Experiments still necessary

| experiment | why | status |
|---|---|---|
| Global-threshold baseline vs stage-aware | **the central claim is untested** | not started |
| ≥39 calibration rollouts/phase, α=0.1 + Bonferroni | to test the guarantee at its target | not started |
| ≥2 scenes, ideally ≥2 failure modes | generality | task 8 partial |
| Causal (online) phase labelling | to support the "sequential" framing | not started |
| Failure-onset labelling | to make lead time meaningful | not started |
| FAIL-Detect comparison | required baseline | not started |

---

# 11. SUPERVISOR EXPLANATION (2 minutes)

> I'm building a runtime monitor for a robot manipulation policy. The robot runs a long
> two-part task and succeeds about half the time. At every control step I compare where the
> controller aimed the hand with where it actually got, normalised by how far it was asked to
> move — that normalisation matters because the compliant controller always undershoots by a
> constant 25 %, so the raw error mostly measures speed.
>
> The contribution is that I don't use one threshold. I split each episode into reach, grasp,
> transport and release using the gripper open/close signal, and calibrate a separate conformal
> threshold for each. Conformal prediction gives a distribution-free bound: at most 10 % of
> healthy runs will trigger an alarm.
>
> Status: the pipeline works end to end. On 20 rollouts of one task the signal separates
> successes from failures with no overlap, and the per-phase thresholds differ by 2.7× — which
> is the evidence that one global threshold is wrong.
>
> Three honest caveats. First, that run had only 5 calibration rollouts, so the false-alarm
> guarantee was violated at 0.67; I need about 40 per phase, which means ~150 rollouts per task.
> Second, almost all the discriminating power sits in the reach phase, because this policy fails
> by never picking the object up — I need a second failure mode to know the method generalises.
> Third, I have not yet run the comparison against a well-calibrated global threshold, and that
> is the experiment that actually tests my claim.
>
> No GPU is needed. I'm using published diffusion-policy checkpoints rather than training my own;
> the remaining compute is about 16 hours of simulation on my laptop.

---

# 12. TECHNICAL EXPLANATION

**Setup.** LIBERO-10 on robosuite 1.4.1 / MuJoCo. `OSC_POSE`, 20 Hz, `control_delta=True`,
`impedance_mode=fixed`, kp=150, `position_limits=None`. Actions are 7-dim in [−1,1]; the
controller rescales by `output_max = [0.05]*3 + [0.5]*3` and re-anchors the goal to the measured
eef pose each step (`set_goal_position(scale_action(δ), self.ee_pos)`), so `planned = p_t + a·0.05`
is the controller's exact internal setpoint.

**Policy.** Frozen LeRobot `DiffusionPolicy` (DDPM, 100 train steps, ResNet18 encoder,
`n_obs_steps=5`, `n_action_steps=8`, `horizon=16`), one checkpoint per task, trained 80k steps
on a LeRobot re-render of LIBERO's 50 demos/task. Not fine-tuned here. Weights SHA-256'd per
rollout to make a mid-collection swap detectable, since exchangeability is the load-bearing
assumption.

**Nonconformity score.** `err_norm_t = ‖p_t + 0.05·a_t[0:3] − p_{t+1}‖ / ‖0.05·a_t[0:3]‖`. The
normalisation is forced by a measured property: realised motion is 0.251 ± 0.006 of the
commanded delta across the full command range, so `err_raw` is ≈ 0.749·‖cmd‖ and is
anti-correlated with failure (successes 0.0203 m, failures 0.0130 m) because failing rollouts
creep. `err_norm` has nominal value ≈ 0.76, matching `1 − 0.251` — an independent confirmation
of the scaling.

**Segmentation.** The gripper channel `a[:,6]` is bang-bang. Phases: open ⇒ reach, closed ⇒
transport, ±5 steps around a rising edge ⇒ grasp, around a falling edge ⇒ release. **This map is
non-causal** — the backward half of each window needs a future transition.

**Conformal construction.** Per phase *k*, score `S_k = max_{t ∈ k} err_norm_t` over a rollout.
Calibrate on successful rollouts only; threshold `q_k` is the `⌈(n+1)(1−α_k)⌉`-th order
statistic, `+∞` if that rank exceeds *n* (requiring `n ≥ 1/α_k − 1`). Bonferroni `α_k = α/K`,
K=4. Alarm at the first *t* with `err_norm_t > q_{phase(t)}`; since `max_k S > q_k ⟺ ∃t: S_t > q_k`,
per-step alarming with max-calibrated thresholds is exactly the per-rollout event, and the union
bound gives `P(any alarm | nominal) ≤ α`. Empirical coverage of `conformal_quantile` verified
over 2000 uniform trials: 0.091 / 0.101 / 0.086 / 0.022 against 0.1 / 0.1 / 0.1 / 0.025.

**Measured results (n=20, task 0, α=0.2, no Bonferroni — the tightest feasible with 5
calibration rollouts).** Thresholds reach 1.110, grasp 1.351, transport 0.922, release 2.454.
FAR 0.67 (target 0.20), TPR 0.78 (7/9), median lead 480 steps. The naive per-step monitor:
FAR 1.00. Per-phase max AUC: reach 0.88, grasp 0.43, transport 0.63, release 0.56 — but at p95,
reach 1.00. 18.8 % of phase maxima originate at `cmd_norm < 2 %` of max, i.e. the max statistic
is dominated by division artefacts.

**Open methodological issues.** (i) `cp_monitor.py` still uses `max` and applies no floor;
(ii) the cal/eval split is RNG-regenerated, not persisted; (iii) lead time references episode
end rather than failure onset; (iv) no global-threshold ablation exists, so the stage-aware
claim is unfalsified rather than supported.

---

# 13. THINGS I CURRENTLY MISUNDERSTAND

Written from the actual confusions visible in this project's history.

### 1. "Are we training a diffusion model?" — **No.**
Nothing is training. `DiffusionPolicy.from_pretrained(repo)` downloads finished weights, `.eval()`
freezes them, `@torch.no_grad()` forbids gradients. The word "diffusion" refers to how the policy
generates actions at *inference* (denoising a chunk of 8 actions), not to any training you run.

### 2. Training vs inference vs calibration — three different things
- **Training** — someone else, 80k gradient steps, on HuggingFace. Needs a GPU. Already done.
- **Inference** — running the frozen policy to produce actions. What `collect_rollouts.py` does.
  Needs no GPU (231 ms/step on MPS).
- **Calibration** — sorting numbers and picking an order statistic. Not learning. No gradients,
  no parameters, no GPU. Takes milliseconds.

### 3. LIBERO demonstrations vs our rollouts — not interchangeable
The 13.7 GB of human demos **trains policies**. Our HDF5s **calibrate the monitor**. Demos are
all successes and could never calibrate a failure detector. Our rollouts have never trained
anything. They meet only indirectly: the checkpoints were trained on a re-rendered conversion of
those same demos.

### 4. "Calibration data" is not "training data"
Calibration rollouts are *successful* rollouts used to learn what "normal" looks like — but
"learn" means "take a quantile of". Add one calibration rollout and the threshold moves by one
order statistic. Nothing is fitted.

### 5. A1–A6 are not project stages
They are **six audits of the measurement setup**, from a spec document. They do not correspond to
phases of work, model stages, or anything the robot does. A1–A4 and A6 test whether assumptions
hold; A5 tests whether a task is usable.

### 6. LIBERO "task" vs our "phase/sub-phase"
A **task** is the whole chore ("put both items in the basket"), indexed 0–9. A **phase** is a
segment *within* one episode (reach / grasp / transport / release), derived from the gripper
signal. One task episode contains ~9 phase segments. Also note LIBERO's BDDL goals give only 2
milestones per task — too coarse — which is why phases come from gripper events instead.

### 7. B1 vs B2 — cheap fix vs expensive fix
Same problem (three tasks are too easy). **B1** adds noise to existing checkpoints — hours, free.
**B2** trains new policies — GPU, money. B2 happens only if B1 fails. Neither is implemented.

### 8. Why a GPU is (not) required
Training needs one. Inference does not — the sim runs at 80–200 Hz on the Mac and the policy at
~230 ms/step. Since we are not training, **no GPU is needed anywhere on the current path.** The
only thing that would change that is deciding to cover all 10 tasks, which forces B2.

### 9. The monitor does not stop the robot
`first_alarm()` returns an index used for metrics. Nothing intervenes. Calling this a "safety
monitor" describes the intended application, not the implemented behaviour.

### 10. `err_raw` is not a worse version of `err_norm` — it is the *opposite*
On real data `err_raw` is **higher for successes** (0.0203 m) than failures (0.0130 m). A monitor
built on it would rank the classes backwards. This is not a scale issue; it is a sign issue,
caused by the OSC compliance gain.

---

# APPENDIX — Code health

### Authoritative vs superseded

| script | status |
|---|---|
| `collect_rollouts.py`, `dp_policy.py`, `subphases.py`, `cp_monitor.py`, `verify_spec.py`, `show_rollout.py` | **authoritative**, active path |
| `eval_pretrained_dp.py` | **superseded** by `collect_rollouts.py` (which reports success rate too, and keeps the data) |
| `train_*.py`, `eval_checkpoint.py`, `common.py`, `profile_train_step.py`, `watch_training.py` | **dormant** — BC-Transformer path, `.venv` only |
| `check_action_space.py`, `check_tracking_gain.py`, `check_timing.py` | **one-shot**, results captured in `action_space.json` and the report |
| `inspect_dataset.py` | utility for LIBERO demos |
| `bootstrap.sh` | required |
| `runpod_setup.sh` | speculative — for a GPU box not currently in use |

### Duplication, dead code, inconsistencies

1. **Duplicated policy adapter.** `eval_pretrained_dp.py:29-131` reimplements the whole
   normalization/flip pipeline that `dp_policy.py` provides. Two copies of the code that must
   not drift. `dp_policy.py` was factored out *after* `eval_pretrained_dp.py` was written; the
   latter was never refactored.
2. **Two different checkpoint hashes.** `common.py:sha256_file` hashes a *file*;
   `dp_policy.py:fingerprint` hashes *state-dict tensors*. Different algorithms — hashes from
   the training path and the collection path are not comparable.
3. **`phase_max` implemented twice, differently.** `cp_monitor.py:63` (no floor) vs
   `verify_spec.py:187` (with floor). **They produce different numbers on the same data.**
4. **Dead variable.** `subphases.py:80` computes `nseg` and never uses it.
5. **Write-only data.** `object_state` is stored in every rollout but read by nothing.
6. **Unused constant path.** `ROT_SCALE`/`rot_scale` is defined, stored, and displayed, but no
   rotational error is ever computed.
7. **Spec/implementation mismatch.** The prose describes p95 aggregation and an A4 floor as the
   conclusions; `cp_monitor.py` implements neither.
8. **A5 has no script.** It is a `collect_rollouts.py` invocation plus a hand-computed CI, unlike
   A1–A4/A6 which live in `verify_spec.py`.

### Bugs and methodological problems

| # | issue | severity |
|---|---|---|
| 1 | **Non-causal phase labels** — `grasp`/`release` windows extend backwards; verified 5/11 steps mislabelled online. Undermines the "sequential monitoring" claim. | **high** |
| 2 | **`cp_monitor.py` lacks the A4 floor** — every published threshold is computed on 19 %-contaminated maxima. | **high** |
| 3 | **Lead time referenced to episode end**, not failure onset (which is unlabelled). Overstates earliness. | **high** |
| 4 | **Cal/eval split not persisted** — regenerated from `RandomState(0)`; appending rollouts silently reshuffles the split. | medium |
| 5 | **No global-threshold ablation** — the central research claim has no control condition. | **high (scientific)** |
| 6 | `--score err_raw` is offered as a CLI choice although it is known to rank classes backwards. No warning. | low |
| 7 | `cp_monitor.py` reports FAR against `args.alpha` even when `--correction none` makes the per-phase level `α` and the union-bound level `Kα`. The printed "target" is then optimistic. | medium |
| 8 | `eval_pretrained_dp.py` discarded per-step data for ~100 rollouts (~2 h compute) before `collect_rollouts.py` existed. | historical |
