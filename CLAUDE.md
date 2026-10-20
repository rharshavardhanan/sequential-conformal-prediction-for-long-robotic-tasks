# PROJECT_1 — sequential conformal prediction for early failure detection

Stage-aware conformal thresholds on the planned-vs-executed tracking error of a
long-horizon manipulation policy, so a failing rollout can be flagged mid-episode
with a bounded false-alarm rate. Simulator is LIBERO (`libero_10` suite) on
robosuite 1.4.1 / MuJoCo.

## Layout

```
scripts/     code — the only thing you edit
  probes/      one-shot measurements whose answers are now frozen below
  setup/       bootstrap.sh, runpod_setup.sh, libero-macos.patch
data/        every generated artifact — the only thing scripts write to
  rollouts/    task{0,1,8}_v10.hdf5 + session_log.json
  init_states/ extended init pools (task{id}_init_160.npy)
  results/     cp_*.json, verification_*.json, action_space.json
  logs/
docs/        technical-report.md, verification-report.md, status.md, superpowers/
third_party/ LIBERO clone (gitignored, rebuilt by bootstrap)
```

`data/` is one `.gitignore` line by design: a new output needs no new rule.

## Two environments — get this wrong and nothing works

**`uv run` is NOT how you run the pipeline.** There are two venvs and they are
not interchangeable:

| env | built from | holds | used by |
|---|---|---|---|
| `.venv` | `uv sync` (pyproject + uv.lock) | LIBERO, robosuite, numpy **<1.24** | `scripts/probes/*` only |
| `.venv-lerobot` | `requirements-lerobot.txt` | + lerobot 0.4.4, numpy **2.2.6**, torch 2.10 | **the entire live pipeline** |

They conflict on numpy and cannot be merged: robosuite 1.4 predates the numpy 2
ABI break, lerobot requires numpy 2.x. So every pipeline command is spelled
`.venv-lerobot/bin/python -m scripts.X`, never `uv run`.

`requirements-lerobot.txt` was frozen 2026-09-08 from the env that produced all
450 rollouts. `scripts/fingerprint.py` aborts collection when the environment
drifts from the recorded fingerprint — that manifest is what makes such an abort
recoverable rather than terminal.

## Setup

```bash
./scripts/setup/bootstrap.sh   # clones LIBERO, patches it, writes ~/.libero/config.yaml, uv sync
uv venv --python 3.10 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python -r requirements-lerobot.txt
uv pip install --python .venv-lerobot/bin/python -e third_party/LIBERO
```

Upstream gotchas the bootstrap handles — all of these will bite again on the GPU
box, so run `bootstrap.sh` there too rather than installing by hand:

- LIBERO ships no `libero/__init__.py`, so a modern PEP 660 editable install maps
  zero packages and `import libero` fails with no error at install time.
- `libero/libero/__init__.py` calls `input()` on first import unless
  `~/.libero/config.yaml` already exists. Hangs any non-interactive run.
- `benchmark_scripts/download_libero_datasets.py` cannot fetch libero_10:
  `--datasets libero_10` is not an accepted choice, and `--datasets libero_100
  --use-huggingface` globs `libero_100/*`, which matches nothing in the HF repo
  (it stores `libero_10/` and `libero_90/`) — it downloads zero files silently.
  Bootstrap calls `snapshot_download` directly: 13.7 GB instead of 80 GB.
- `scripts/setup/libero-macos.patch` carries five source fixes; see the header
  comment in `bootstrap.sh` for what each one is and why.

## Policy — downloaded, not trained

**Nothing trains.** The policy is a frozen pretrained LeRobot `DiffusionPolicy`
(DDPM, ResNet18 encoder, `n_obs_steps=5`, `n_action_steps=8`), one checkpoint per
task, pulled from the Hub:

| repo | tasks |
|---|---|
| `yzembodied/libero10_task{0,1,2}_diffusion` | 0, 1, 2 |
| `yzembodied/libeo10_task{3,8}_diffusion` | 3, 8 — note the upstream `libeo` typo |

`from_pretrained()` downloads finished weights, `.eval()` freezes them,
`@torch.no_grad()` forbids gradients.

Measured success rates sit in the required **40-80%** band for tasks 0 and 1
(0.687, 0.727) — below the band there is too little nominal behaviour to
calibrate on, above it too few failures to detect. Task 8 is **0.853, out of
band**; see `docs/status.md`.

**Normalization is applied by hand on purpose** (`scripts/dp_policy.py`). These
checkpoints store stats as `normalize_inputs.*` / `unnormalize_outputs.*` inside
`model.safetensors` (pre-0.4 LeRobot layout). lerobot 0.4.4 moved to external
processor files and **discards those keys on load**, leaving the policy with no
stats — a naive `from_pretrained()` returns plausible-looking 7-vectors that are
pure garbage.

**Freeze the checkpoint for the whole collection.** CP's guarantee assumes one
data-generating process; a mid-collection swap silently mixes two robots into one
calibration set. `scripts/fingerprint.py:file_sha256` stamps the checkpoint hash
into every rollout so a swap is detectable after the fact.

## Commands

```bash
# --- live pipeline (.venv-lerobot ONLY) ---
.venv-lerobot/bin/python -m scripts.collect_rollouts --task-id 0 --n-rollouts 150
.venv-lerobot/bin/python -m scripts.verify_spec  --rollouts data/rollouts/task0_v10.hdf5
.venv-lerobot/bin/python -m scripts.cp_monitor   --rollouts data/rollouts/task0_v10.hdf5 --alpha 0.1
.venv-lerobot/bin/python -m scripts.show_rollout data/rollouts/task0_v10.hdf5 -r 0
.venv-lerobot/bin/python -m scripts.gen_init_states --task-id 0 --n 160

# --- probes: answers already frozen below, kept as provenance (.venv) ---
uv run python scripts/probes/check_action_space.py   # controller config + scaling
uv run python scripts/probes/check_tracking_gain.py  # OSC compliance gain
uv run python scripts/probes/check_timing.py         # sim throughput, Mac vs cloud
```

## Action space — confirmed, do not re-derive

`OSC_POSE`, 20 Hz, `control_delta=True`, `impedance_mode=fixed`, 7-dim action
(6 arm + 1 gripper), all components normalised to [-1, 1].

The controller rescales before the sim sees it, so the planned setpoint is:

```python
planned_pos = obs["robot0_eef_pos"] + action[:3] * 0.05      # metres
planned_rot = ...                    action[3:6] * 0.5       # radians (axis-angle)
```

`current + action[:3]` is wrong by 20x. `position_limits` is None, so the
setpoint is unclipped. The goal is re-anchored to the *measured* eef pose every
control step, not integrated from the previous setpoint.

**OSC is compliant**: realised motion is a near-constant 0.25 of the commanded
delta (gain 0.251, sd 0.006, over the full command range). So a raw
`||planned - executed||` is ~0.75x the commanded delta — mostly a readout of
commanded speed, not of anomaly. Normalise by `||commanded delta||` or keep
speed roughly constant within each calibration bucket.

## Init states — 50 or 100, it is a choice

Each task ships **two** init-state files, and they differ:

| file | states per task | used by |
|---|---|---|
| `*.init` | **100** | nothing by default |
| `*.pruned_init` | **50** | LIBERO's `benchmark`, hence `evaluate_one_task_success` |

`task.init_states_file` resolves to the **pruned** one, so LIBERO's own eval has
only 50 distinct starts. `evaluate_one_task_success` indexes with
`% init_states.shape[0]`, so asking for 100 rollouts silently replays each start
twice.

Collecting 150 rollouts from 100 states with `i % 100` reuses states 0-49 twice —
a *structured* reuse pattern, which is not exchangeable and breaks CP's
assumption. `scripts/gen_init_states.py` extends each pool to 160 validated
states so every rollout gets its own; `collect_rollouts.py` prefers
`data/init_states/task{id}_init_160.npy` and falls back to the shipped file.

## Dormant: the BC-Transformer path

The project originally trained its own BC-Transformer. That path was removed on
2026-09-08 (`scripts/train_*.py`, `eval_checkpoint.py`, `common.py`,
`profile_train_step.py`, `watch_training.py`, and the `train` dependency group) —
recoverable from git history if training ever returns. Kept here because the
sizing numbers still govern any decision to resume:

BC-Transformer is 5.4M params but encodes 10 frames x 2 cameras per sample, so it
is **compute-bound, not I/O-bound** — 673 ms/batch compute vs 65 ms data at batch
32. More DataLoader workers will not help. Per epoch 5.65 min (459 batches,
14,700 seqs from 50 demos); 50 epochs x 1 task = 4.7 h; **x 10 tasks = 47 h**.

Sim itself is *not* the bottleneck: MuJoCo runs at 200 Hz state-only and 80 Hz
with 2x128px cameras, so rollout collection is a couple of hours.
