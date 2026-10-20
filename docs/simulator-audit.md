# Simulator audit — toward a model-agnostic single-window experiment interface

**Date:** 2026-09-10
**Scope:** full repository audit. **No invasive changes made.** Every number below was
printed from running code, not read from a document.

---

## A. Repository architecture

```
scripts/
  ── environment / rollout ───────────────────────────────────────────
  collect_rollouts.py    LIBERO env + policy -> HDF5. Owns the score formula.
  gen_init_states.py     extends each init pool to 160 validated states
  fingerprint.py         env/checkpoint hashing; aborts collection on drift
  ── policy ──────────────────────────────────────────────────────────
  dp_policy.py           DiffusionPolicyRunner (lerobot). Manual normalization.
  train_bc_rnn.py        BC-RNN / BC-Transformer trainer (LIBERO lifelong)
  eval_bcrnn.py          success-rate eval for robomimic-family checkpoints
  ── CP science ──────────────────────────────────────────────────────
  subphases.py           OFFLINE phase segmentation from the gripper channel
  cp_monitor.py          calibration: q_safe, q_esc, dwell limits + artefact
  monitor.py             CausalLabeller + SafetyMonitor -> StatePacket  [live core]
  verify_spec.py         Part-A spec verifications against collected rollouts
  ablation_global.py     flat-vs-subtask threshold ablation
  ── viewer / inspection ─────────────────────────────────────────────
  demo_monitor.py        frame sources (replay/live) + single-window viewer
  show_rollout.py        per-rollout dump
  inspect_dataset.py     dataset inspection
tests/test_monitor.py    18 tests, pass under BOTH venvs
```

**Layer separation as it stands today:**

| layer | where it lives | verdict |
|---|---|---|
| LIBERO — tasks, goals, init states | `libero.libero.benchmark`, `gen_init_states.py` | present |
| robosuite — robot, cameras, controller | `OffScreenRenderEnv` | present |
| MuJoCo — physics | under robosuite | present |
| policy | `dp_policy.py` only | **single-family** |
| monitor — score + phase + verdict | `monitor.py` | present, clean |
| CP — thresholds | `cp_monitor.py` (offline) + artefact JSON | present, clean |
| arbiter | — | **does not exist in this repo** |
| viewer | `demo_monitor.py` | present, single window |

The monitor/viewer boundary is already correct: the viewer consumes `StatePacket` and
computes nothing. `grep` confirms no scoring, phase or threshold logic in the render path.

---

## B. Verified simulation model — Task 2

Printed from a live `OffScreenRenderEnv` on task 0:

| quantity | assumed in docs | **actual** | match |
|---|---|---|---|
| robot model | Franka Panda | `OnTheGroundPanda` | yes (LIBERO's floor-mounted Panda) |
| controller | OSC_POSE | `OSC_POSE` | yes |
| action dim | 7 | `7` | yes |
| control frequency | 20 Hz | `20 Hz` | yes |
| physics timestep | 0.002 s | `0.002 s` | yes |
| physics frequency | 500 Hz | `500 Hz` | yes |
| physics steps per action | 25 | `25` | yes |
| `control_delta` | True | `True` | yes |
| `impedance_mode` | fixed | `fixed` | yes |
| `position_limits` | None | `None` | yes |
| position scale | 0.05 m | `output_max[:3] = [0.05 0.05 0.05]` | yes |
| rotation scale | 0.5 rad | `output_max[3:6] = [0.5 0.5 0.5]` | yes |
| max episode length | 600 | `600` (caller-enforced, not env-enforced) | yes |
| goal predicates | 2 | `[['in','alphabet_soup_1','basket_1_contain_region'], ['in','tomato_sauce_1','basket_1_contain_region']]` | yes |
| init pool | 160 | `data/init_states/task0_init_160.npy` shape `(160, 123)` | yes |

**No discrepancies.** Every value in the research documents is correct. `planned = eef_before
+ action[:3] * 0.05` is confirmed by `output_max` and must not be changed.

One nuance worth recording: `max_steps=600` is enforced by the *collection loop*, not by the
environment. The env will keep stepping past 600 if asked.

---

## C. Data flow for one timestep — Task 3

```
X(t) = obs dict from robosuite  (45 keys total; each policy consumes a subset)
   │
   ├── DiffusionPolicy consumes 5 keys:
   │      agentview_image            (256,256,3) uint8   -> per-channel normalized
   │      robot0_eye_in_hand_image   (256,256,3) uint8   -> per-channel normalized
   │      robot0_eef_pos             (3,)   ┐
   │      robot0_eef_quat            (4,)   ├─ -> quat2axisangle -> state(8) -> minmax
   │      robot0_gripper_qpos        (2,)   ┘
   │      robot0_joint_pos           (7,)        -> separate minmax-normalized input
   │      history: n_obs_steps = 5, held INSIDE lerobot's queues
   │
   └── BC-Transformer / BC-RNN consume 4 keys (via cfg.data.obs_key_mapping):
          agentview_rgb    <- agentview_image
          eye_in_hand_rgb  <- robot0_eye_in_hand_image
          gripper_states   <- robot0_gripper_qpos
          joint_states     <- robot0_joint_pos
          history: seq_len = 10, held by the caller
          ALSO consumes a task embedding (BERT, models--bert-base-cased in HF cache)
   │
   ▼
Y_pred(t) = action[7] in [-1,1]
   action[0:3]  position delta, x0.05 m
   action[3:6]  axis-angle delta, x0.5 rad
   action[6]    gripper, bang-bang -1 open / +1 closed
   │
   ▼
OSC_POSE  -> joint torques        (setpoint re-anchored to MEASURED eef each step)
   │
   ▼
MuJoCo x 25 physics steps @ 0.002 s
   │
   ▼
Y_actual(t) = obs(t+1)["robot0_eef_pos"]
   │
   ▼
TRACKING   planned = eef_before + action[:3]*0.05
           err_raw = ||planned - executed||
           cmd_norm = ||action[:3]*0.05||
           err_norm = err_raw / max(cmd_norm, 1e-8)
           motion_mask = cmd_norm >= 0.0017320508075688774
   │
   ▼
CP MONITOR  phase (causal, 5-step lag)  ->  q_safe[phase], q_esc[phase], d_hat[phase]
   │
   ▼
DECISION    SAFE / ESCALATE / VIOLATION / STUCK_TIMEOUT   (independent latches)
   │
   ▼
ARBITRATION  ** NOT IMPLEMENTED **
```

**Critical detail preserved:** the OSC goal is re-anchored to the *measured* eef pose every
control step, not integrated from the previous setpoint. `current + action[:3]` without the
0.05 factor is wrong by 20x.

---

## D. Policy support — Task 5, measured not assumed

A policy counts as supported only if it loads, receives its observation, produces a 7-vector,
drives the same env, completes a rollout, and logs data.

| | Diffusion Policy | BC-RNN | BC-Transformer | ACT |
|---|---|---|---|---|
| library | lerobot 0.4.4 | LIBERO + robomimic 0.2.0 | LIBERO + robomimic 0.2.0 | lerobot 0.4.4 |
| **runs in** | `.venv-lerobot` | `.venv` | `.venv` | `.venv-lerobot` |
| checkpoint | HF repo, `model.safetensors` | `data/checkpoints/bcrnn_task0_ep*.pth` | `data/checkpoints_bct/bct_task0_ep010.pth` | **none** |
| checkpoint format | safetensors | `{state_dict, cfg, epoch, train_loss, monitor}` | `{state_dict, cfg, previous_masks}` | — |
| loading | `DiffusionPolicy.from_pretrained` | `get_algo_class(...).policy.load_state_dict` | same | `ACTPolicy.from_pretrained` |
| normalization | **manual**, from in-model `normalize_*` keys | inside LIBERO policy | inside LIBERO policy | ? |
| image preprocessing | none — raw 256px into ResNet18 | LIBERO's own transform | LIBERO's own transform | ? |
| history | 5 obs steps, internal queue | seq_len 10, caller-held | seq_len 10, caller-held | internal |
| output | **action chunk**, `n_action_steps=8` | single action | single action | **action chunk** |
| task embedding | no | **yes** (BERT) | **yes** (BERT) | no |
| **measured success (task 0)** | **0.687** (n=150) | **0.00–0.05** (n=20) | **0.44** (n=50) | — |
| **status** | **SUPPORTED** | loads + runs, but **useless** | loads + runs, **not yet wired to this simulator** | **NOT SUPPORTED** |

`data/checkpoints_bct/bct_task0_ep010.pth` was saved by LIBERO's own trainer (it carries
`previous_masks` and has no `epoch`/`monitor`), so its loader differs slightly from the
BC-RNN checkpoints that `train_bc_rnn.py` wrote. Both are `torch.load` + `load_state_dict`.

### The environment split — the single biggest constraint

```
.venv            numpy 1.23.5  torch 2.13.0  robomimic 0.2.0  lerobot  ABSENT
.venv-lerobot    numpy 2.2.6   torch 2.10.0  robomimic ABSENT  lerobot 0.4.4
```

Both carry robosuite 1.4.1, mujoco 3.2.7, libero 0.1.0, h5py, matplotlib. So the
**simulator, monitor and viewer already run in either environment** — verified: all 18 tests
pass under both. Only the *policy library* differs.

Today that means **no single process can run a DiffusionPolicy and a BC-Transformer
together**, which blocks cross-family arbitration.

**But the split looks bridgeable, contrary to `CLAUDE.md`.** Two pieces of evidence:

1. `uv pip install --dry-run robomimic==0.2.0` into `.venv-lerobot` resolves to **8 purely
   additive packages** (`robomimic, egl-probe, tensorboard, tensorboardX, grpcio, markdown,
   werkzeug, tensorboard-data-server`). It would **not touch numpy, torch, mujoco, robosuite,
   lerobot or python** — the exact six keys `fingerprint.build()` records. Fingerprint impact:
   **none**.
2. `data/checkpoints/history_task0.json` records that BC-RNN was trained with
   **robomimic 0.2.0 under numpy 2.1.3** on Colab. robomimic already runs on numpy 2.

Residual risk: robomimic 0.2.0 contains one `np.bool` (in `dataset.py`, the training path,
not inference). Unverified until actually installed. **Reversible** with `uv pip uninstall`.

---

## E. Scientific inconsistencies found

1. **`scp/` does not exist in this repository.** `docs/status.md` §5 describes a package
   (`io, phases, scoring, calibrate, methods, evaluate`) that is on the teammate's side. All
   calibration here runs through `cp_monitor.py`.

2. **Thresholds do not reproduce across the two implementations.** At the pre-registered
   settings this repo gives reach 1.230 / grasp 1.321 / transport 1.072 / release 1.922
   against the report's 1.40 / 1.32 / 1.29 / 1.68. Only `grasp` agrees. This is
   `status.md` §8 item 1, still unresolved. **Not touched by this audit.**

3. **`CLAUDE.md` overstates the environment conflict.** It says the two venvs "cannot be
   merged". The dry run and the training history above both suggest otherwise for inference.
   Reported, not acted on.

4. **BC-RNN is not a viable cheap arm** (0–5%), so the arbitration design in `status.md` §4
   — cheap BC-RNN, expensive ACT — needs its cheap arm replaced by BC-Transformer (44%).
   Already documented in `docs/bc-rnn-vs-bc-transformer.md`.

5. **Rollout logging stores no monitor output.** `err_norm` and `motion_mask` are stored, but
   phase, CP score comparison, `q_safe`, `q_esc`, dwell and verdict are all recomputed
   offline. Not a bug — they are *derived* quantities, and storing derivations invites
   divergence — but it does mean a rollout file alone cannot reproduce a monitor decision
   without also naming its calibration artefact.

---

## F. Current implementation status — Task 18

| feature | status | what is missing |
|---|---|---|
| MuJoCo environment | DONE | — |
| LIBERO task 0 | DONE | — |
| Diffusion Policy | DONE | — |
| BC-RNN | PARTIAL | loads and runs, but 0–5% success — unusable as a policy arm |
| BC-Transformer | PARTIAL | 44% success measured; **no adapter wiring it into the simulator** |
| ACT | MISSING | library present in lerobot 0.4.4; no checkpoint, no training code |
| common policy adapter | MISSING | `demo_monitor.live_source` hardcodes `DiffusionPolicyRunner` |
| single-window viewer | DONE | — |
| live mode | DONE | — |
| replay mode | DONE | — |
| rollout logging | PARTIAL | live mode logs nothing; only `collect_rollouts.py` writes HDF5 |
| CP scoring | DONE | — |
| phase detection | DONE | — |
| causal phase detection | DONE | 5-step lag, byte-identical to offline on all 150 rollouts |
| q_safe | DONE | — |
| q_esc | DONE | — |
| stall detection | DONE | — |
| arbitration | MISSING | no arbiter code in this repo |
| policy switching | MISSING | blocked on adapters + the env split |
| 5-run experiment mode | MISSING | — |
| checkpoint fingerprint guard | DONE | `SafetyMonitor.check_policy` refuses mismatches |
| tests | PARTIAL | 18 tests cover the monitor; none cover env construction, policy loading, chunking, logging |

---

## G. Dataset inventory

| dataset | location | size | contents |
|---|---|---|---|
| collected rollouts | `data/rollouts/task{0,1,8}_v10.hdf5` | 4.4 GB | 150 complete each; task 0 103/47, task 1 109/41, task 8 128/22 |
| init-state pools | `data/init_states/task{0,1,8}_init_160.npy` | 372 KB | (160, 123) each |
| LIBERO human demos | `third_party/LIBERO/libero/datasets/libero_10/` | 13 GB | 10 files, 50 demos/task. Keys: `agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori, ee_states, gripper_states, joint_states` |
| BC-RNN checkpoints | `data/checkpoints/` | 365 MB | 5 epochs + `history_task0.json` |
| BC-Transformer ckpt | `data/checkpoints_bct/` | 21 MB | 1 checkpoint, ep010, sha `dfb20c44` |
| results | `data/results/` | 6.9 MB | CP artefacts, eval JSONs, verification |
| HF policy cache | `~/.cache/huggingface/hub` | — | 5 diffusion + 12 DiT + smolvla + bert-base-cased |

**Per-step HDF5 schema (18 datasets):** `actions, action_chunk, chunk_len, chunk_is_new,
eef_pos, eef_quat, gripper_qpos, joint_pos, planned_pos, executed_pos, err_raw, err_norm,
cmd_norm, motion_mask, object_state, predicates, images/agentview, images/eye_in_hand`.

**Per-rollout attrs (19)** include `success, steps, split, seed, init_state_index,
policy_sha256, fingerprint, git_revision` and library versions.

---

## H. Recommended architecture

```
scripts/
  policies/
    base.py        PolicyAdapter ABC: reset, act, fingerprint, name,
                   action_horizon, observation_spec  -> PolicyOutput
    diffusion.py   wraps the EXISTING dp_policy.DiffusionPolicyRunner
    libero_bc.py   BC-Transformer + BC-RNN (same loader, different policy_type)
    act.py         when a checkpoint exists
    registry.py    name -> adapter. The ONLY place a policy name is branched on.
  chunking.py      PolicyOutput + ChunkBuffer: chunk -> one action per step
  arbiter.py       ArbitrationDecision, pure function of (score, phase, thresholds)
  experiment.py    N-run driver + aggregate summary
  monitor.py       UNCHANGED
  cp_monitor.py    UNCHANGED
  subphases.py     UNCHANGED
  demo_monitor.py  viewer only; sources call the registry
```

`PolicyOutput` carries `action` (the one executable 7-vector), `chunk` (optional), `chunk_len`,
`is_new`. `collect_rollouts.py` already records exactly these — the abstraction matches what
the pipeline stores, so chunk information is not lost.

---

## I. Implementation order

Each step is independently testable and independently rejectable.

1. **Verify the env merge.** Install robomimic into `.venv-lerobot`, confirm
   `import libero.lifelong.models.bc_transformer_policy` works under numpy 2.2.6, re-run all
   18 tests, re-run `fingerprint.build()` and diff against a stored rollout's fingerprint to
   prove it is unchanged. **If this fails, everything below stays single-family and
   arbitration needs an out-of-process policy server.** Reversible.
2. **`PolicyAdapter` + `DiffusionAdapter`.** Pure refactor, wrapping existing code. Prove it
   by reproducing `rollout_0083` bit-for-bit through the adapter.
3. **`LiberoBCAdapter`** for BC-Transformer. Prove: loads, one full 600-step rollout, 7-vector
   actions, HDF5 written.
4. **Rollout logging from live mode**, reusing `collect_rollouts.py`'s writer rather than a
   second one.
5. **`--runs N` experiment mode** with the per-run table and aggregate.
6. **Arbiter**, as a pure function with its own tests, wired to nothing.
7. **Policy switching**, only once 2, 3 and 6 are green.
8. **ACT** — needs training first; out of scope until a checkpoint exists.

## Tests to add

env construction and the Task-2 constants · action dimension · observation extraction per
policy · adapter loading for each family · output shape · action scaling (`0.05`) · chunk
buffer yields one action per step and preserves chunk contents · adapter round-trip
reproduces a recorded rollout · HDF5 written from live mode matches the collection schema ·
arbitration decision table · switch counting · fingerprint guard per adapter · viewer startup
under `MPLBACKEND=Agg`.

Existing regression tests (18) already cover: streamed phase labels equal offline labels on
all 150 rollouts, score equals recorded `err_norm`, streaming violation index equals batch,
motion mask matches, known success stays SAFE, known stall triggers STUCK_TIMEOUT, policy
mismatch refused.

---

## What I did NOT change

Nothing. No file in `scripts/`, `tests/` or `data/` was modified during this audit. The two
calibration artefacts for tasks 1 and 8 were generated in a prior session, not this one.
