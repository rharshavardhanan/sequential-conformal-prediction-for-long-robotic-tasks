# Mentor demo script — one-command presentation of PROJECT_1

**Date:** 2026-08-18

## Brainstorm outcome (thin, and why)

The ask was well specified: one `.sh` that shows the project's real results in the
terminal, fast, for a mentor presentation. The one unstated constraint worth
surfacing: the request asked to "fake running" but show "real actual output".

Investigation resolved that tension rather than trading off against it —
`cp_monitor.py` recomputes every headline number from `rollouts/task0.hdf5` in
**0.39 s** and reproduces the committed `cp_task0.json` exactly. So no number is
fabricated. Only rollout *collection* (31.9 min wall) cannot re-run live, and a
real recorded log of it exists at `logs/collect_task0.log`.

Design rule adopted: **every panel is labelled `[LIVE]` or `[REPLAY]`.** The
mentor can tell from the screen which is which, and can ask for any live panel to
be re-run with different arguments.

## Task

**Goal** — `./scripts/demo.sh` presents the project end to end in one command,
with every displayed number computed from real data or replayed from a real log.

**Why** — the results currently live in four JSON files, two markdown reports and
an HDF5 file. There is no way to show them in a meeting without opening files by
hand, and the honest headline (the max-based score misses its false-alarm target)
is buried.

**Files** — `scripts/demo.sh` (new).

**Steps**
1. Preflight: check `.venv-lerobot`, `rollouts/*.hdf5`, `logs/*.log` exist; fail loudly.
2. Framing panels: problem, signal definition, phase definition, CP pitfalls.
3. `[REPLAY]` collection from `logs/collect_task0.log`, streamed.
4. `[LIVE]` `subphases.py`, `cp_monitor.py` (task0 + task8, incl. `--per-step`), `verify_spec.py`.
5. Honest scorecard: what met acceptance, what did not, and the identified fix.
6. Flags: `--fast`, `--pause`, `--section N`, `--no-color`, `--help`.

**Done when** — `./scripts/demo.sh --fast` exits 0 and its CP panel prints
`false alarm rate (held-out nominal): 0.67` and `detection rate: 0.78 (7/9)`,
matching `cp_task0.json`; and `--section` runs each section standalone.

## Non-goals

No fabricated numbers. No re-running collection. No edits to the analysis scripts
— the demo orchestrates them, it does not reimplement them.
