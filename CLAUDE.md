# PROJECT_1 — sequential conformal prediction for early failure detection

Stage-aware conformal thresholds on the planned-vs-executed tracking error of a
long-horizon manipulation policy, so a failing rollout can be flagged mid-episode
with a bounded false-alarm rate. Simulator is LIBERO (`libero_10` suite) on
robosuite 1.4.1 / MuJoCo.

## Setup

```bash
./scripts/bootstrap.sh      # clones LIBERO, patches it, writes ~/.libero/config.yaml, uv sync
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
- `patches/libero-macos.patch` carries five source fixes; see the header comment
  in `bootstrap.sh` for what each one is and why.

Deps are a trimmed manifest (LIBERO's own `setup.py` declares
`install_requires=[]`). The `lifelong` training stack is in a separate group
(`uv sync --group train`); rollout collection never imports it. `egl-probe`
(a robomimic dep) is confined to Linux via `override-dependencies` — it
cmake-fails on Apple Silicon because EGL does not exist on macOS, and nothing
on a Mac calls it.

## Policy

No usable pretrained checkpoint exists: LIBERO ships none, and its zoo is
BC-RNN / BC-Transformer / BC-ViLT (no diffusion policy). Public libero_10
checkpoints on HF are 7B VLAs or use incompatible obs/action conventions. So we
train our own BC-Transformer, which satisfies the 7 x [-1,1] delta action format
and the 128x128 two-camera obs spec *by construction* — verified: the demo files
report `ac_dim=7`.

Success rate must land in **40-80%**: below it there is too little nominal
behaviour to calibrate on, above it too few failures to detect. Training our own
makes this a dial (pick the epoch) rather than luck.

**Freeze the checkpoint for the whole collection.** CP's guarantee assumes one
data-generating process; a mid-collection swap silently mixes two robots into one
calibration set. `scripts/common.py:sha256_file` stamps the checkpoint hash into
every rollout so a swap is detectable after the fact.

## Checks

```bash
uv run python scripts/check_action_space.py    # controller config + scaling, writes action_space.json
uv run python -m scripts.profile_train_step    # is training data- or compute-bound?
uv run python -m scripts.train_single_task --task-id 0 --n-epochs 50 --eval-every 5 --n-eval 20
uv run python -m scripts.eval_checkpoint --checkpoint <path> --task-id 0 --n-eval 20
uv run python scripts/check_tracking_gain.py   # OSC compliance gain
uv run python scripts/check_timing.py          # sim throughput, Mac vs cloud sizing
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

## Training cost, measured on this Mac (M-series, MPS)

BC-Transformer is 5.4M params but encodes 10 frames x 2 cameras per sample, so
it is **compute-bound, not I/O-bound** — 673 ms/batch compute vs 65 ms data at
batch 32. More DataLoader workers will not help.

| | |
|---|---|
| per epoch (459 batches, 14,700 seqs from 50 demos) | 5.65 min |
| 50 epochs x 1 task | 4.7 h |
| 50 epochs x 10 tasks | **47 h** |

Sim itself is *not* the bottleneck: MuJoCo runs at 200 Hz state-only and 80 Hz
with 2x128px cameras, so rollout collection is a couple of hours. Training is
what belongs on the GPU box.

## Init states — 50 or 100, it is a choice

Each task ships **two** init-state files, and they differ:

| file | states per task | used by |
|---|---|---|
| `*.init` | **100** | nothing by default |
| `*.pruned_init` | **50** | LIBERO's `benchmark`, hence `evaluate_one_task_success` |

`task.init_states_file` resolves to the **pruned** one, so LIBERO's own eval has
only 50 distinct starts. `evaluate_one_task_success` indexes with
`% init_states.shape[0]`, so asking for 100 rollouts silently replays each start
twice. For the 100-rollout-per-task rule to give 100 *distinct* initial
conditions, load the `.init` file explicitly. (Upstream pruned to 50 without
documenting why — verify the extra 50 are actually solvable before relying on
them.)

## Action space — confirmed, do not re-derive

(see the earlier section; the key line is `action[:3] * 0.05`)

**OSC is compliant**: realised motion is a near-constant 0.25 of the commanded
delta (gain 0.251, sd 0.006, over the full command range). So a raw
`||planned - executed||` is ~0.75x the commanded delta — mostly a readout of
commanded speed, not of anomaly. Normalise by `||commanded delta||` or keep
speed roughly constant within each calibration bucket.
