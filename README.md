# Sequential conformal prediction for early failure detection

Stage-aware conformal thresholds on the planned-vs-executed tracking error of a
long-horizon manipulation policy, so a failing rollout can be flagged
**mid-episode** with a bounded false-alarm rate.

Simulator is [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
(`libero_10` suite) on robosuite 1.4.1 / MuJoCo. The policy is a **frozen
pretrained** LeRobot `DiffusionPolicy` — nothing in this repo trains a model.

## The idea

Split conformal prediction calibrated on nominal rollouts gives a threshold with
a distribution-free guarantee. Two things this gets right that a naive
implementation does not:

1. **The guarantee is per-rollout, not per-step.** Checking a 600-step episode
   step-by-step is 600 tests; even at `alpha=0.01` a nominal rollout alarms with
   probability ~1. Calibrating the per-rollout *maximum* score instead gives
   `P(any alarm during a nominal rollout) <= alpha`.
2. **The threshold is stage-aware.** Reach, grasp, carry and place have very
   different nominal tracking error. One global threshold is either deaf during
   grasp or trigger-happy during reach. Stages are segmented from gripper
   transitions, which are bang-bang and therefore clean.

## Quickstart

```bash
./scripts/setup/bootstrap.sh                       # LIBERO + datasets (13.7 GB)
uv venv --python 3.10 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python -r requirements-lerobot.txt
uv pip install --python .venv-lerobot/bin/python -e third_party/LIBERO

.venv-lerobot/bin/python -m scripts.collect_rollouts --task-id 0 --n-rollouts 150
.venv-lerobot/bin/python -m scripts.cp_monitor --rollouts data/rollouts/task0_v10.hdf5 --alpha 0.1
```

> **Two venvs, not one.** `.venv` (numpy <1.24, for robosuite) and
> `.venv-lerobot` (numpy 2.x, for lerobot) conflict and cannot be merged. The
> pipeline runs in `.venv-lerobot`; `uv run` gives you the wrong one. See
> `CLAUDE.md`.

## Layout

| path | what |
|---|---|
| `scripts/` | the pipeline — collect, segment, monitor, verify |
| `scripts/probes/` | one-shot measurements whose answers are frozen in `CLAUDE.md` |
| `scripts/setup/` | bootstrap, GPU-box setup, the LIBERO patch |
| `data/` | every generated artifact (gitignored) |
| `docs/` | reports and plans |

## Status

450 rollouts collected across tasks 0, 1 and 8. The measurement instrument is
built and validated; the experiment testing the project's actual claim has not
run yet. Current numbers, open problems and the recommended order of work:
[`docs/status.md`](docs/status.md).

- [`docs/technical-report.md`](docs/technical-report.md) — full pipeline trace
- [`docs/verification-report.md`](docs/verification-report.md) — the A1-A6 verifications
