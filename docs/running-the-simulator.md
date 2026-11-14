# Running the simulator

Everything runs in **`.venv-lerobot`**. Never `uv run`. Prefix live commands with
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

```bash
cd ~/PROJECTS/PROJECT_1
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## 1. See what policies exist

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor --list-policies
```

```
name                 display           class      avail  measured success
diffusion_task0      Diffusion Policy  expensive  True   0.687 (n=150)
bct_task0            BC-Transformer    cheap      True   0.440 (n=50)
bcrnn_task0          BC-RNN            cheap      True   0.050 (n=20)   unusable
act_task0            ACT               expensive  False  not measured   NOT TRAINED
```

`measured success` is only ever a number that was actually measured. ACT shows
`not measured` because nothing has been trained — the GUI will never invent one.

---

## 2. Replay a recorded rollout (always works, deterministic)

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --replay data/rollouts/task0_v10.hdf5 -r 83 \
  --artefact data/results/demo_task0.json
```

The three cases worth showing:

| rollout | what happens |
|---|---|
| `-r 41` | stays SAFE the whole way — the monitor is not trigger-happy |
| `-r 83` | VIOLATION at step 40 of 600, latched red |
| `-r 76` | STUCK_TIMEOUT at 128 with a LOW score — why dwell is separate |

Pick cases by index rather than by hope:

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --replay data/rollouts/task0_v10.hdf5 --list \
  --artefact data/results/demo_task0.json
```

---

## 3. Run it live

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --live --task-id 0 --init 83 \
  --policy diffusion_task0 --camera frontview --res 512 \
  --artefact data/results/demo_task0.json
```

~300 ms/step, so a 600-step episode takes about 3 minutes. `--max-steps 120` gives a
40-second version. Close the window at any time and you still get the run summary.

`--camera` accepts `frontview` (default — shows the whole arm), `agentview` (what the
policy sees; crops the arm), `sideview`, `birdview`. Only the *display* camera changes;
the policy always receives agentview + wrist at 256 px, because it was calibrated there.

---

## 4. Run a different policy

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --live --task-id 0 --init 83 --policy bct_task0 \
  --artefact data/results/demo_task0.json
```

This **refuses to run**, on purpose:

```
[abort] bct_task0 (sha dfb20c44) was not the policy this artefact was
        calibrated on (sha 3105133c).
        Conformal thresholds do not transfer across policies.
```

Conformal prediction assumes one data-generating process. A different policy is a
different process, and nothing in the stack would otherwise raise — the numbers would
just quietly stop meaning what they claim. Two ways forward:

**(a) Watch it drive with no safety claim.** The thresholds are hidden and the header
is marked `[UNCALIBRATED]`:

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --live --task-id 0 --init 83 --policy bct_task0 --uncalibrated \
  --artefact data/results/demo_task0.json
```

**(b) Calibrate for it properly.** Two commands, roughly 2 hours for the first:

```bash
.venv-lerobot/bin/python -m scripts.collect_rollouts \
    --task-id 0 --n-rollouts 150 --suffix _bct

.venv-lerobot/bin/python -m scripts.cp_monitor \
    --rollouts data/rollouts/task0_bct.hdf5 \
    --alpha 0.1 --cal-frac 0.8 --alpha-esc 0.3 --beta 0.1 \
    --out data/results/demo_bct_task0.json
```

Then point `--artefact` at the new file. There is no shortcut where a new policy
borrows an old policy's thresholds.

---

## 5. Run it N times

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --live --task-id 0 --init 83 --runs 5 --max-steps 300 \
  --artefact data/results/demo_task0.json
```

Each run uses a different init state (`--init` + run index) and prints a row; the
aggregate reports the empirical success rate and says out loud that 5 runs is a sanity
check, not a benchmark. The per-task rates this project reports come from 150 rollouts.

The summary separates **policy outcome** (did the task succeed) from **monitor outcome**
(SAFE / ESCALATE / VIOLATION / STUCK_TIMEOUT). They are different things: a run can
succeed and still be flagged, or fail without being caught.

---

## 6. Arbitration

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --live --task-id 0 --init 83 --policy bct_task0 --uncalibrated \
  --arbitrate --calm-steps 20 \
  --artefact data/results/demo_task0.json
```

Without `--expensive`, escalation is shown as a **counterfactual** — "an arbiter would
escalate here" — because there is no second arm to switch to. Naming an untrained policy
as the expensive arm is refused rather than drawn:

```
[abort] --expensive act_task0 has no checkpoint. Naming an untrained policy as the
        expensive arm would show a switch that cannot happen.
```

---

## 7. Save a video instead of watching

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --replay data/rollouts/task0_v10.hdf5 -r 83 \
  --artefact data/results/demo_task0.json \
  --save data/results/demo_r83.mp4
```

With `--runs N`, each run is saved as `..._run1.mp4`, `..._run2.mp4`, and so on.

---

## 8. Reading the screen

| region | shows |
|---|---|
| header | mode (`[LIVE]`/`[REPLAY]`), task, init state, policy, UNCALIBRATED warning |
| robot view | the MuJoCo scene, with the current phase named above it |
| verdict banner | SAFE / ESCALATE / VIOLATION / STUCK_TIMEOUT, latched once fired |
| left column | phase, policy + hash, step, s(t), q_esc, q_safe, the inequality, dwell, all four thresholds |
| TRACKING | commanded vs realised displacement in mm — the compliance gain, visible |
| MOTION | command magnitude and whether the step is above the motion floor |
| ARBITRATION | selected policy, compute class, reason |
| POLICY STACK | every policy for this task with its **measured** success and availability |
| plot | s(t) solid, q_safe dashed, q_esc dotted, escalation band shaded between them, phase bands behind, masked steps greyed, violation latched with a red rule |
| event log | only transitions — phase changes, CP decisions, arbiter switches |

`q_esc` is always **below** `q_safe`. Escalation is the looser budget
(`alpha_esc` 0.30 > `alpha` 0.10), so it fires first, by construction.

---

## 9. Tests

```bash
.venv-lerobot/bin/python -m pytest tests/ -q      # 35 tests
```

Covers: causal phase labels equal offline segmentation on all 150 task-0 rollouts, the
5-step lag, streaming score equals recorded `err_norm`, streaming violation index equals
the batch index, motion mask, known success stays SAFE, known stall fires STUCK_TIMEOUT,
policy-hash mismatch refused, chunk buffering, registry honesty, and nine arbiter cases.

---

## 10. Environment note

`robomimic==0.2.0` was installed into `.venv-lerobot` with `--no-deps` (its `egl-probe`
dependency fails to cmake-build on macOS, and `.venv` does not have it either). This is
what lets BC-Transformer and lerobot policies run in **one process**.

Verified afterwards: the environment fingerprint is byte-identical to the one stored in
`task0_v10.hdf5`, so rollout collection can still resume. Nothing the fingerprint records
(torch, numpy, mujoco, robosuite, lerobot, python) changed.

To undo: `uv pip uninstall --python .venv-lerobot/bin/python robomimic`.
