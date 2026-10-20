# Repo reorganization — design

**Date:** 2026-09-08 · **Status:** approved

## Problem

The repo is organized around a policy it no longer uses. The pipeline pivoted
from *train our own BC-Transformer* to *frozen pretrained LeRobot diffusion
policy*; the filesystem never followed. `TECHNICAL_REPORT.md:988` already calls
the training stack "dormant".

Three consequences, in order of severity:

1. **The live environment is unreproducible.** `pyproject.toml` pins
   `numpy<1.24` and describes `.venv` (the dormant BC-Transformer env). Every
   live script runs under `.venv-lerobot` — lerobot 0.4.4, numpy 2.2.6, torch
   2.10 — which exists in no manifest at all. All 450 rollouts were produced by
   an environment that cannot be rebuilt. `fingerprint.py` aborts on env drift,
   so this is a live threat to the CP guarantee, not a hygiene issue.
2. **Generated output lands in seven different places** — `rollouts/`, `logs/`,
   `init_states_v10/`, `probe_v8/`, `probe_v10/`, `sweep_task3/`,
   `dataset_preview/`, plus eight loose JSON files at the repo root. This is why
   the root grew to 27 visible entries.
3. **`CLAUDE.md` is stale in a way that misdirects.** It attributes checkpoint
   hashing to `scripts/common.py:sha256_file`; the live path actually uses
   `scripts/fingerprint.py:file_sha256`. `common.py` is imported only by dormant
   code. The section "Action space — confirmed, do not re-derive" appears twice.

## Target

```
PROJECT_1/
├── scripts/     code
│   ├── probes/  check_* one-shot measurements (provenance)
│   └── setup/   bootstrap.sh, runpod_setup.sh, libero-macos.patch
├── data/        ALL generated output (one .gitignore line)
├── docs/        reports and plans
├── third_party/ LIBERO clone
└── .trash/      staged deletions, gitignored and hidden
```

Root: 27 visible entries / 11 directories -> 4 directories + 6 files.

**The invariant that keeps it clean:** `scripts/` is the only thing edited,
`data/` is the only thing written to, and `data/` is one `.gitignore` line.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Code directory name | keep `scripts/` | `python -m scripts.X` is in ~15 docstrings and the user's shell history. A rename buys convention and costs muscle memory. These are scripts, not an installable package. |
| Deletion policy | stage into `.trash/` | `probe_*`, `sweep_task3`, `experiments/` are gitignored and were never committed — `rm` is irreversible and they cost hours of compute. `.trash/` is hidden, so the root still reads as 4 folders. |
| `check_*.py` probes | keep, move to `scripts/probes/` | They are the evidence for constants `CLAUDE.md` says never to re-derive. Deleting the evidence for a frozen constant is bad science. Out of the main path, not gone. |
| `.venv` (1.9 GB) | **keep** | The audit proposed deleting it. Wrong: the `check_*` probes need it and it is the env `pyproject.toml` describes. Real reclaim is ~620 MB. |
| The three reports | move, do not merge | Merging 1,556 lines of overlapping prose is an editorial task that can lose information. Out of scope here; relocation is mechanical and safe. |

## Non-goals

- Merging `REPORT.md` / `PROGRESS.md` / `TECHNICAL_REPORT.md`.
- Any change to pipeline behaviour, numerics, or the CP method.
- Rewriting `.venv-lerobot`. It is frozen to a manifest, not rebuilt.

## Risk and its control

The only real risk is breaking the live collection pipeline. The path surface is
small and was enumerated before any move: five hardcoded paths across four live
files (`collect_rollouts.py` x2, `gen_init_states.py`, `verify_spec.py` x2,
`inspect_dataset.py`). Everything else is an explicit CLI argument.

Control: `git mv` for tracked files so history follows; the environment manifest
is frozen *before* anything moves; and the pipeline is executed end to end
against the relocated data as the final acceptance gate.
