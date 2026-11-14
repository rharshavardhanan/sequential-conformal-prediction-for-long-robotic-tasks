# Repo reorganization — implementation plan

Design: `docs/superpowers/specs/2026-09-08-repo-reorganization-design.md`

Order matters. Task 1 is a safety net and runs first. Task 3 must land in the
same breath as task 2 or the collector breaks. Task 10 is the acceptance gate.

---

## Task 1 — Freeze the live environment manifest

- **Goal:** `.venv-lerobot` is rebuildable from a committed file.
- **Why:** the env that produced all 450 rollouts exists in no manifest.
  `fingerprint.py` aborts collection on env drift, and nothing could rebuild the
  env it demands. This is the highest-value, lowest-risk change; if the session
  died after this task alone it would still have been worth running.
- **Files:** `requirements-lerobot.txt` (new)
- **Steps:** `pip freeze` from `.venv-lerobot`; drop the editable `libero` line
  (bootstrap supplies it); add a header saying which interpreter and why.
- **Done when:** `grep -c '^lerobot==0.4.4' requirements-lerobot.txt` -> `1` and
  `grep -c '^numpy==2' requirements-lerobot.txt` -> `1`.

## Task 2 — Create `data/` and move every generated artifact into it

- **Goal:** all output lives under `data/`; none at the repo root.
- **Why:** output currently lands in seven places. This is the single change
  that takes the root from 27 entries to 4 directories.
- **Files:** `data/{rollouts,init_states,results,logs}/`; moves from
  `rollouts/`, `init_states_v10/`, `logs/`, and 8 root `*.json`.
- **Steps:** mkdir the tree; `git mv` tracked files, plain `mv` for ignored ones;
  preserve every filename exactly.
- **Done when:** `ls data/rollouts/*.hdf5 | wc -l` -> `3`; `ls *.json 2>/dev/null | wc -l` -> `0`.

## Task 3 — Repoint the five hardcoded paths

- **Goal:** every live script reads and writes under `data/` with no new flags.
- **Why:** task 2 breaks `collect_rollouts.py` on its own — the init-state path
  is a hardcoded f-string, not an argument. Without this the collector dies.
- **Files:** `scripts/collect_rollouts.py` (L58, L87),
  `scripts/gen_init_states.py` (L73), `scripts/verify_spec.py` (L57, L64),
  `scripts/inspect_dataset.py` (L39)
- **Steps:** edit each literal; fix `verify_spec.py`'s already-stale
  `rollouts/task0.hdf5` default to the real `task0_v10.hdf5`.
- **Done when:** `grep -rn 'init_states_v10\|"rollouts"' scripts/*.py` -> no hits.

## Task 4 — Stage the dead BC-Transformer stack into `.trash/`

- **Goal:** `scripts/` contains only code the live pipeline uses.
- **Why:** 726 lines across 6 files that the project's own report calls dormant,
  plus 478 MB of artifacts only they produced.
- **Files:** -> `.trash/`: `train_single_task.py`, `train_all_tasks.py`,
  `eval_checkpoint.py`, `profile_train_step.py`, `watch_training.py`,
  `common.py`, `eval_pretrained_dp.py`, `demo.sh`, `bert/`, `experiments/`
- **Steps:** mkdir `.trash/`; `git mv` tracked, `mv` ignored.
- **Done when:** `grep -rn 'scripts.common' scripts/` -> no hits.

## Task 5 — Stage superseded data into `.trash/`

- **Goal:** no throwaway probe output at the repo root.
- **Why:** `probe_v8/`, `probe_v10/`, `sweep_task3/`, `rollouts/archive/`,
  `dataset_preview/` are referenced by zero scripts and zero docs.
- **Files:** those five, plus `init_states_v10/task0_init_100.npy` (superseded;
  only `_160` is read).
- **Steps:** `mv` each into `.trash/`.
- **Done when:** root shows exactly 4 visible directories.

## Task 6 — Drop the `train` dependency group

- **Goal:** the manifest declares only what the live pipeline imports.
- **Why:** 10 deps (robomimic, wandb, transformers, hydra, thop...) that nothing
  reachable imports any more.
- **Files:** `pyproject.toml`, `uv.lock`
- **Steps:** delete the `[dependency-groups]` block and its comment; `uv lock`.
- **Done when:** `uv lock --check` exits 0 and `grep -c robomimic pyproject.toml` -> `0`.
- **Rollback:** if `uv lock` fails to re-resolve, `git checkout pyproject.toml uv.lock` and skip this task; it is independent of every other one.

## Task 7 — Subdivide `scripts/` into `probes/` and `setup/`

- **Goal:** the live pipeline is the first thing visible in `scripts/`.
- **Why:** 9 live files currently sit beside 3 settled one-shot probes and 2
  setup shell scripts.
- **Files:** `scripts/probes/` <- `check_action_space.py`, `check_timing.py`,
  `check_tracking_gain.py`; `scripts/setup/` <- `bootstrap.sh`,
  `runpod_setup.sh`, `patches/libero-macos.patch`
- **Steps:** `git mv`; fix `bootstrap.sh`'s `cd "$(dirname "$0")/.."` (now two
  levels) and its `../../patches/` patch path.
- **Done when:** `bash -n scripts/setup/bootstrap.sh` exits 0 and the patch path
  it references resolves (`test -f`).

## Task 8 — Rewrite `.gitignore` around the new tree

- **Goal:** one ignore line per concept instead of fourteen.
- **Why:** the current file lists individual output filenames
  (`pretrained_dp_task*.json`, `checkpoints.json`); every new output needs a new
  line. `.ruff_cache` is only ignored by the machine's global gitignore, so it
  reappears untracked on the GPU box.
- **Files:** `.gitignore`
- **Done when:** `git status --porcelain` lists no untracked artifact.

## Task 9 — Correct `CLAUDE.md`

- **Goal:** the file describes the repo that now exists.
- **Why:** it names `common.py:sha256_file` (dormant) instead of
  `fingerprint.py:file_sha256` (live), duplicates the action-space section
  verbatim, and every command path it lists has moved.
- **Files:** `CLAUDE.md`, `README.md` (currently 0 bytes)
- **Done when:** `grep -c '^## Action space' CLAUDE.md` -> `1`;
  `grep -c 'common.py' CLAUDE.md` -> `0`; `test -s README.md`.

## Task 10 — Acceptance: run the live pipeline against the moved data

- **Goal:** proof the reorganization changed no behaviour.
- **Why:** every prior task is a path edit. Only execution proves they are right.
- **Steps:** compile every live script; run `verify_spec` and `cp_monitor`
  against `data/rollouts/task0_v10.hdf5` under `.venv-lerobot`.
- **Done when:** `cp_monitor` prints its per-phase thresholds and exits 0, and
  the numbers match `REPORT.md`'s recorded values for task 0.
