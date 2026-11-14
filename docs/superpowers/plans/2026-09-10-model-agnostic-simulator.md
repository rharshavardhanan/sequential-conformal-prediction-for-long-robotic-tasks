# Model-Agnostic Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** turn the single-window viewer into a model-agnostic experiment interface where any
policy family drives the same MuJoCo env, monitor and GUI, with arbitration and N-run mode.

**Architecture:** a `PolicyAdapter` ABC returning `PolicyOutput`; a registry as the only place
policy names branch; a pure `arbiter`; the viewer still renders `StatePacket` and computes
nothing.

**Tech Stack:** `.venv-lerobot` only (robomimic now installed there, see Step 0), matplotlib,
pytest.

## Global Constraints

- **`.venv-lerobot/bin/python` for everything.** Never `uv run`.
- **Never modify** `monitor.py`, `cp_monitor.py`, `subphases.py`, `fingerprint.py`,
  `collect_rollouts.py`'s score formula, or the HDF5 schema.
- **`planned = eef_before + action[:3] * 0.05`.** Never change.
- **`q_esc < q_safe` always** (alpha_esc > alpha). The GUI must never render them inverted.
- **No invented quantities.** No confidence %, no failure probability. ACT shows
  `NOT AVAILABLE` until a checkpoint exists and is measured.
- **Image sizes differ per family:** DiffusionPolicy wants 256x256; BC-Transformer wants
  128x128 (`shape_meta.all_shapes`). The env renders 256; BCT downsamples `[::2, ::2]`,
  exactly as `collect_rollouts.py` does for storage.
- **Do not commit** unless asked.

---

### Step 0 — Environment merge  [DONE 2026-09-10]

`uv pip install --python .venv-lerobot/bin/python --no-deps robomimic==0.2.0`
(`--no-deps` because `egl-probe` fails to cmake-build on macOS; `.venv` has robomimic without
it too). Verified: fingerprint vs `task0_v10.hdf5` **unchanged**, 18 tests pass,
DiffusionPolicy still loads, and BCTransformer + Diffusion + ACT all import in one process.

Task embeddings precomputed to `data/task_embs_libero10_bert.npz` (10, 768) so the lerobot env
never needs `transformers` or `hydra`.

---

### Task 1: PolicyAdapter, PolicyOutput, ChunkBuffer

**Goal** — one interface every policy family satisfies, and a buffer that turns a chunk into
exactly one executable action per control step.

**Why** — without it every new policy means another `if` in the simulator, and chunk-emitting
policies (Diffusion, ACT) cannot share a loop with single-action ones (BC-T, BC-RNN).

**Files:** Create `scripts/policies/__init__.py`, `scripts/policies/base.py`;
Test `tests/test_policies.py`

**Interfaces produced:**
- `PolicyOutput(action: np.ndarray[7], chunk: np.ndarray|None, chunk_len: int, is_new: bool)`
- `PolicyAdapter` ABC: `reset()`, `act(obs) -> PolicyOutput`, `fingerprint() -> str`,
  `name -> str`, `action_horizon -> int`, `observation_spec -> dict`, `compute_class -> str`
- `ChunkBuffer(horizon)`: `push(chunk)`, `pop() -> np.ndarray[7]`, `__len__`

**Done when** — `pytest tests/test_policies.py -k "output or buffer"` passes, including a test
that a chunk of 8 yields 8 single actions in order and that chunk contents are preserved.

---

### Task 2: DiffusionAdapter

**Goal** — the existing `DiffusionPolicyRunner` behind the new interface, bit-identical.

**Why** — proves the abstraction is lossless before anything depends on it.

**Files:** Create `scripts/policies/diffusion.py`; Modify `tests/test_policies.py`

**Done when** — feeding `rollout_0083`'s recorded actions is unnecessary; instead the adapter's
`fingerprint()` equals `3105133c...` and a 25-step live rollout through the adapter produces
`err_norm` values matching a direct `DiffusionPolicyRunner` run to 1e-9.

---

### Task 3: LiberoBCAdapter (BC-Transformer, BC-RNN)

**Goal** — LIBERO/robomimic checkpoints run in the same loop as lerobot ones.

**Why** — this is the model-agnostic claim. BC-Transformer (44%) is the viable cheap arm.

**Files:** Create `scripts/policies/libero_bc.py`; Modify `tests/test_policies.py`

**Notes for the implementer:** the checkpoint carries `cfg` (EasyDict) and `cfg.shape_meta`,
so no hydra is needed. Obs mapping from `cfg.data.obs_key_mapping`:
`agentview_rgb <- agentview_image`, `eye_in_hand_rgb <- robot0_eye_in_hand_image`,
`gripper_states <- robot0_gripper_qpos`, `joint_states <- robot0_joint_pos`. Images are
`[::2, ::2]` downsampled to 128, transposed to CHW, float/255. History `seq_len=10`.
Task embedding from `data/task_embs_libero10_bert.npz`. Policy has `get_action`.

**Done when** — `pytest tests/test_policies.py -k libero` passes: loads
`data/checkpoints_bct/bct_task0_ep010.pth`, `act()` returns shape `(7,)` in `[-1,1]`, and one
full 600-step MuJoCo rollout completes.

---

### Task 4: Registry

**Goal** — one table mapping a policy name to its adapter and its declared metadata.

**Why** — the audit's requirement that policy names branch in exactly one place.

**Files:** Create `scripts/policies/registry.py`; Modify `tests/test_policies.py`

Each entry declares `compute_class` (cheap/expensive), `measured_success` (or `None` — never
invented), `available` (checkpoint present?), and a factory.

**Done when** — `list_policies()` reports `diffusion_task0` available with success 0.687,
`bct_task0` available with 0.44, `act_task0` **unavailable with success None**, and
`pytest -k registry` passes.

---

### Task 5: Arbiter

**Goal** — a pure function from monitor state to an `ArbitrationDecision`.

**Why** — arbitration must be testable without a simulator and must not live in the GUI.

**Files:** Create `scripts/arbiter.py`; Test `tests/test_arbiter.py`

`ArbitrationDecision(selected_policy, reason, escalated, violated, stalled, switched, event)`.
Immediate escalation; hysteretic de-escalation (N calm steps); optional chunk-boundary
constraint; switch counting; per-policy step counts.

**Done when** — `pytest tests/test_arbiter.py` passes: escalates when score crosses q_esc,
does not de-escalate before the hysteresis window, counts switches correctly, and returns the
cheap policy when no expensive arm is available.

---

### Task 6: GUI — policy strip, arbitration panel, run controls, summary

**Goal** — the single window shows policy stack, arbitration decision and reason, the CP plot
with correctly-ordered thresholds, an event log of transitions only, and a run summary.

**Files:** Modify `scripts/demo_monitor.py`

**Done when** — three mp4s re-render with unchanged verdicts, and a frame shows the policy
strip with ACT marked NOT AVAILABLE.

---

### Task 7: `--runs N` experiment mode

**Files:** Create `scripts/experiment.py`; Modify `scripts/demo_monitor.py`

**Done when** — `--runs 5` prints a per-run table and an aggregate with the empirical success
rate, explicitly labelled as a sanity check rather than a benchmark.

---

### Task 8: Instructions

**Files:** Create `docs/running-the-simulator.md`

**Done when** — a reader can run replay, live, a different policy, and a 5-run experiment by
copy-paste.
