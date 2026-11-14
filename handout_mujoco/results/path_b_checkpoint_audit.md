# Path B checkpoint audit — Reach→Push arbitration artifacts

**Date:** 2026-09-14 · branch `mujoco-arbitration` · nothing modified, nothing downloaded.

## Verdict

**PATH B BLOCKED ON CHECKPOINT RECOVERY.**

The cheap MLP, the Reach TQC actor and the Push TQC actor are not present on this
machine in any form. No adapter was written, no weights were invented, no substitute
model was used.

## Everywhere searched

| Location / method | Result |
|---|---|
| `git branch -a` | `main`, `mujoco-arbitration`, `remotes/origin/main` — no policy branch |
| `git log --all` | 2 commits (`194426e`, `ac43271`); neither touches panda-gym |
| `git log --all --diff-filter=D` (files ever deleted) | none |
| `git rev-list --all --objects` filtered for `.zip/.pt/.pth/.ckpt/.pkl/tqc/sac/scp` | no matching blob in history |
| `git stash list` | empty |
| `find` over `~/PROJECTS ~/Downloads ~/Desktop ~/Documents ~/Library/CloudStorage` for `.zip .pt .pth .ckpt .pkl .joblib` | 39 files, none panda-gym (list below) |
| `find` for dirs `composed`, `results_push`, `panda_gym_demo`, `scp`, `*arbitr*`, `*panda*` | only `third_party/mujoco_menagerie/franka_emika_panda` (the robot model) |
| content grep for `TQC`, `VecNormalize`, `stable_baselines`, `sb3`, `alpha_esc`, `q_esc`, `qhat` | only this project's own files + unrelated JS lockfiles |
| `~/Downloads/handout_mujoco` (second copy) | byte-identical scope — data + replay only |
| `PROJECT_1.zip` (24 GB full backup) | **0** matches for `tqc|panda_gym|composed/|part45` — predates this work |
| `bcrnn_ckpts.zip` | LIBERO BC-RNN task0 checkpoints (5 × 76 MB) — different project |

## Candidate artifacts found, and why each is unusable

| Candidate | Why not |
|---|---|
| `roboturk/roboturk_act*.pt`, `roboturk_bcrnn_state_only.pt` | RoboTurk track: 37-dim state, 8-dim joint-target action. Different task, observation and action space. |
| `data/tower/{dart,rel,tgt}/tower_{bcrnn,act}.pt` | Tower Creation track: 32/41/50-dim obs, 8-dim joint targets. Different task. |
| `data/checkpoints/bcrnn_task0_ep*.pth`, `checkpoints_bct/` | LIBERO BC-RNN / BC-Transformer. Different task. |
| `~/Downloads/bcrnn_ckpts.zip` | Same LIBERO checkpoints, zipped. |
| `.trash/experiments/LIBERO_10/.../task0_model.pth` | LIBERO BC-Transformer, deleted run. Different task. |
| `certain-road/*.pt`, `buffalo_l.zip` | YOLO / face models. Unrelated projects. |

None of these is a 5.9k-parameter panda-gym MLP or a TQC actor. Substituting any of
them would produce a demo that is not the benchmarked system.

## Why they were never here

`handout_mujoco/requirements.txt` lists **`mujoco, numpy, imageio[ffmpeg], pillow`** —
no torch, no stable-baselines3, no panda-gym. The handout was deliberately cut as a
**replay-only** package so it could be shared without the research repo. Its README
states the provenance: `panda_gym_demo/composed/part45_calib_bench.py`, a path that
does not exist on this machine.

## Exactly what must be recovered

From the machine that ran `panda_gym_demo/composed/part45_calib_bench.py`:

**Required (blocking):**
1. `cheap` MLP checkpoint — 5,891 parameters (count from `results/efficiency_summary.json`)
2. Reach TQC actor — 5,382 parameters
3. Push TQC actor — 541,190 parameters (fine-tuned from `enaitzb/TQC-PandaPush-v3`, run 2 @350k)
4. Any `VecNormalize` / observation-normalisation statistics saved alongside them
5. The observation builder — the panda-gym env wrapper or feature function the policies were trained against

**Required to confirm the arbiter (non-blocking, see ARBITER_INTERFACE.md):**
6. `scp/arbitration.py`, `scp/scoring.py`, `scp/calibrate.py`
7. `composed/part45_calib_bench.py`

**Useful:** `composed/results_push/` (the full results dir the handout was cut from),
`train_push.py`, `results/push_finetune_log_run{1,2}.json`.

SB3 saves policies as `.zip`; the cheap MLP may be `.pt`/`.pth` or a plain state dict.
Search the source machine for `*.zip` under the TQC training output directory.
