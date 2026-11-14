# Live Monitor Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** A single-screen experiment viewer that runs the calibrated sequential-CP safety
monitor step-by-step over a LIBERO-Long rollout — from recorded HDF5 or from live MuJoCo —
so the whole decision chain is legible in a few seconds.

**Architecture:** Three layers with a hard boundary. A frame source yields
`(frame, action, eef_before, eef_after)`; `SafetyMonitor.step()` turns that into a
`StatePacket`; the viewer renders packets and nothing else. The monitor never imports
matplotlib or MuJoCo; the viewer never computes a score.

**Tech Stack:** Python 3.10 in `.venv-lerobot`, numpy 2.2.6, h5py, matplotlib 3.10.9
(`macosx` backend), pytest 9.1.1. No new dependencies.

**Design spec:** `docs/superpowers/specs/2026-09-09-live-monitor-demo-design.md`

## Global Constraints

- **Run everything with `.venv-lerobot/bin/python`, never `uv run`.** `.venv` has numpy
  <1.24 and no lerobot; the pipeline env is `.venv-lerobot`. Tests:
  `.venv-lerobot/bin/python -m pytest`.
- **No new dependencies.** matplotlib 3.10.9 and pytest 9.1.1 are already installed. Do not
  add PySide6, pyqtgraph, or anything else to `.venv-lerobot` — it is frozen against
  `requirements-lerobot.txt` and guarded by `scripts/fingerprint.py`.
- **Do not modify** `scripts/collect_rollouts.py`, `scripts/dp_policy.py`,
  `scripts/fingerprint.py`, or `scripts/subphases.py`.
- **The 0.05 scale factor is mandatory.** `planned = eef_before + action[:3] * 0.05`.
  Omitting it is wrong by 20x. `ROT_SCALE = 0.5` is not used by this monitor.
- **Motion floor is `0.02 * 0.05 * sqrt(3) = 0.0017320508075688774`.** Read it from the
  artefact, never hardcode it in the viewer.
- **PHASES order is fixed:** `["reach", "grasp", "transport", "release"]`, from
  `scripts.subphases.PHASES`. Index with `scripts.subphases.PHASE_ID`.
- **Never display a quantity the algorithm does not consume.** No confidence percentages, no
  "failure probability", no `BC-RNN`/`ACT` policy row — neither policy exists.
- **Do not commit unless explicitly asked.** Committing moves HEAD, and
  `fingerprint.compare()` diffs `git_revision`, so any future resume of
  `task{0,1,8}_v10.hdf5` would abort. Steps below stage files; the commit step is listed but
  runs only on the user's say-so.

---

### Task 1: Causal phase labeller

**Goal** — a streaming labeller whose emitted labels are byte-identical to
`subphases.segment()`, at a constant 5-step lag.

**Why** — this is what makes the demo a monitor rather than a replay of offline analysis. If
the labels differ from the ones calibration used, the threshold on screen is not the
calibrated threshold and the guarantee is detached from the display.

**Files:**
- Create: `scripts/monitor.py`
- Create: `tests/test_monitor.py`

**Interfaces:**
- Consumes: `scripts.subphases.PHASE_ID`, `scripts.subphases.PHASES`,
  `scripts.subphases.segment`
- Produces: `CausalLabeller(window: int = 5)` with `push(gripper_cmd: float) -> list[tuple[int, int]]`
  returning `(index, phase_id)` pairs finalised by this step, and `flush() -> list[tuple[int, int]]`
  emitting the tail. `CausalLabeller.LAG == 5`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_monitor.py
import h5py
import numpy as np
import pytest

from scripts.monitor import CausalLabeller
from scripts.subphases import PHASES, segment

ROLLOUTS = "data/rollouts/task0_v10.hdf5"


def _stream(gripper_cmd):
    """Run the labeller over a full gripper channel; return labels and lags."""
    lab = CausalLabeller()
    out = np.full(len(gripper_cmd), -1, np.int8)
    lags = set()
    for t, v in enumerate(gripper_cmd):
        for i, p in lab.push(v):
            out[i] = p
            lags.add(t - i)
    for i, p in lab.flush():
        out[i] = p
    return out, lags


def test_causal_labels_match_offline_on_every_rollout():
    """The streamed labels must equal segment() exactly, on all 150 rollouts.

    This is the load-bearing test: calibration used segment(), so any divergence
    means the displayed threshold is not the calibrated one.
    """
    with h5py.File(ROLLOUTS, "r") as f:
        keys = sorted(k for k in f if k.startswith("rollout_")
                      and f[k].attrs.get("complete"))
        assert len(keys) == 150
        for k in keys:
            g = f[k]["actions"][:, 6]
            want, _ = segment(g)
            got, _ = _stream(g)
            assert np.array_equal(got, want), (
                f"{k}: first mismatch at "
                f"{int(np.flatnonzero(got != want)[0])}"
            )


def test_lag_is_exactly_five():
    """Every label finalised during streaming is finalised exactly 5 steps late."""
    with h5py.File(ROLLOUTS, "r") as f:
        g = f["rollout_0000"]["actions"][:, 6]
    _, lags = _stream(g)
    assert lags == {5}
    assert CausalLabeller.LAG == 5


def test_forward_window_reaches_steps_not_yet_seen():
    """Regression: a transition relabels 5 steps AFTER it, which do not exist yet.

    A labeller that only rewrites already-buffered steps passes nothing but the
    most trivial input. Here the gripper closes at index 10; segment() labels
    indices 5..14 as grasp, so indices 10..14 must be grasp even though they are
    appended after the transition is detected.
    """
    g = np.array([-1.0] * 10 + [1.0] * 15)
    want, _ = segment(g)
    got, _ = _stream(g)
    assert np.array_equal(got, want)
    assert PHASES[got[12]] == "grasp"
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.monitor'`

- [x] **Step 3: Write minimal implementation**

```python
# scripts/monitor.py
"""Causal streaming monitor for the sequential-CP safety demo.

The offline pipeline segments a completed trajectory. A live monitor cannot.
`subphases.segment()` relabels indices d-4..d+5 around a transition at diff-index
d, and d is only observable at time d+1 -- so the label for index i is final only
once time i+5 has been seen. This module emits at that lag, which makes its labels
provably identical to the ones calibration used, at a cost of 250 ms at 20 Hz.
"""

from scripts.subphases import PHASE_ID


class CausalLabeller:
    """Streaming phase labels, identical to subphases.segment() at a 5-step lag.

    The backward half of the relabel window (d-4..d) is rewritten in place. The
    forward half (d+1..d+5) lands on steps that do not exist yet, so it is carried
    as an open window and applied on append -- the single trap in this component.
    """

    LAG = 5

    def __init__(self, window=5):
        self.w = window
        self.g, self.lab, self.emitted = [], [], 0
        self.open_until, self.open_ph = -1, None

    def push(self, gripper_cmd):
        """Feed one gripper command; return [(index, phase_id)] finalised now."""
        closed = bool(gripper_cmd > 0)
        self.g.append(closed)
        t = len(self.g) - 1
        base = PHASE_ID["transport"] if closed else PHASE_ID["reach"]
        self.lab.append(self.open_ph if t <= self.open_until else base)

        if t >= 1 and self.g[t] != self.g[t - 1]:
            d = t - 1  # diff-index, as in segment()
            ph = PHASE_ID["grasp"] if closed else PHASE_ID["release"]
            for i in range(max(0, d - self.w + 1), min(d + self.w + 1, len(self.lab))):
                self.lab[i] = ph
            self.open_until, self.open_ph = d + self.w, ph

        out = []
        while self.emitted <= t - self.w:
            out.append((self.emitted, self.lab[self.emitted]))
            self.emitted += 1
        return out

    def flush(self):
        """Emit the final `window` steps, which no later observation can revise."""
        out = []
        while self.emitted < len(self.lab):
            out.append((self.emitted, self.lab[self.emitted]))
            self.emitted += 1
        return out
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v`
Expected: 3 passed

- [x] **Step 5: Stage (commit only if asked)**

```bash
git add scripts/monitor.py tests/test_monitor.py
# git commit -m "feat: causal phase labeller matching offline segmentation"
```

**Done when** — `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v` reports
`3 passed`, including `test_causal_labels_match_offline_on_every_rollout` over all 150
task-0 rollouts.

---

### Task 2: Calibration artefact with escalation and dwell constants

**Goal** — `cp_monitor.py --out` emits a complete, self-describing artefact: four `q_safe`,
four `q_esc`, four `d_hat`, every parameter that produced them, and the checkpoint hash.

**Why** — the viewer loads constants read-only and must never recalibrate at demo time.
Without `q_esc` there is no escalation band; without `d_hat` there is no STUCK_TIMEOUT; and
without the checkpoint hash the demo cannot refuse to run on a mismatched policy.

**Files:**
- Modify: `scripts/cp_monitor.py` (add `--alpha-esc`, `--beta`; extend the `--out` payload;
  fix `first_alarm` masking)
- Modify: `tests/test_monitor.py` (append)

**Interfaces:**
- Consumes: `scripts.cp_monitor.conformal_quantile(scores, alpha) -> (q, rank, feasible)`,
  `scripts.cp_monitor.phase_agg(r, how, use_mask) -> np.ndarray[4]`,
  `scripts.cp_monitor.load(path, score_key) -> (rollouts, meta)`
- Produces: `scripts.cp_monitor.dwell_maxima(r) -> np.ndarray[4]` returning the per-rollout
  maximum contiguous dwell per phase, `nan` where the phase is absent. The `--out` JSON gains
  keys `escalation_thresholds`, `dwell_limits`, `alpha_esc`, `beta`, `cal_frac`,
  `phase_window`, `motion_floor_abs`, `seed`, `command`.

- [x] **Step 1: Write the failing test**

```python
# append to tests/test_monitor.py
import json
import subprocess

from scripts.cp_monitor import dwell_maxima, load
from scripts.subphases import PHASES


def test_dwell_maxima_takes_longest_contiguous_segment_per_phase():
    """Per-rollout max dwell, not pooled segments -- the brief's version fires
    on 47% of successes because a rollout holds several segments per phase."""
    rollouts, _ = load(ROLLOUTS, "err_norm")
    r = next(x for x in rollouts if x["name"] == "rollout_0076")
    d = dwell_maxima(r)
    # 0076's gripper never closes: 600 steps of reach, nothing else.
    assert d[PHASES.index("reach")] == 600
    assert np.isnan(d[PHASES.index("grasp")])
    assert np.isnan(d[PHASES.index("transport")])


def test_artefact_has_every_constant_the_viewer_needs(tmp_path):
    out = tmp_path / "demo_task0.json"
    subprocess.run(
        [".venv-lerobot/bin/python", "-m", "scripts.cp_monitor",
         "--rollouts", ROLLOUTS, "--alpha", "0.1", "--cal-frac", "0.8",
         "--alpha-esc", "0.3", "--beta", "0.1", "--out", str(out)],
        check=True, capture_output=True,
    )
    a = json.loads(out.read_text())
    for key in ("thresholds", "escalation_thresholds", "dwell_limits", "alpha",
                "alpha_esc", "beta", "cal_frac", "agg", "phase_window",
                "motion_floor_abs", "score", "policy_sha256", "seed", "command"):
        assert key in a, f"artefact missing {key}"
    for key in ("thresholds", "escalation_thresholds", "dwell_limits"):
        assert set(a[key]) == set(PHASES), f"{key} must cover all four phases"
    # alpha_esc > alpha implies a strictly looser quantile, so q_esc <= q_safe.
    for p in PHASES:
        assert a["escalation_thresholds"][p] <= a["thresholds"][p] + 1e-12
    assert a["motion_floor_abs"] == pytest.approx(0.0017320508075688774)
    assert a["phase_window"] == 5


def test_reference_values_reproduce():
    """Locks the numbers the demo displays. Regenerate deliberately, never silently."""
    a = json.loads(open("data/results/demo_task0.json").read())
    assert a["thresholds"]["reach"] == pytest.approx(1.2303, abs=1e-3)
    assert a["thresholds"]["grasp"] == pytest.approx(1.3213, abs=1e-3)
    assert a["thresholds"]["transport"] == pytest.approx(1.0721, abs=1e-3)
    assert a["thresholds"]["release"] == pytest.approx(1.9220, abs=1e-3)
    assert a["dwell_limits"]["reach"] == pytest.approx(128.0)
    assert a["dwell_limits"]["transport"] == pytest.approx(80.0)
    assert a["policy_sha256"].startswith("3105133c")
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -k "dwell or artefact or reference" -v`
Expected: FAIL — `ImportError: cannot import name 'dwell_maxima'`

- [x] **Step 3: Add `dwell_maxima` and the new arguments**

Add to `scripts/cp_monitor.py`, after `phase_agg`:

```python
def dwell_maxima(r):
    """Longest contiguous run of each phase in one rollout; nan where absent.

    Calibrated the same way the deviation score is: the per-ROLLOUT maximum, not
    pooled per-segment durations. A rollout holds several segments per phase, so a
    per-segment quantile is not a per-rollout bound -- measured, it fires on 47%
    of successful rollouts at beta=0.10.
    """
    lab = np.asarray(r["labels"])
    out = np.full(len(PHASES), np.nan)
    if len(lab) == 0:
        return out
    edges = np.flatnonzero(np.diff(lab)) + 1
    for start, end in zip(np.r_[0, edges], np.r_[edges, len(lab)]):
        i = int(lab[start])
        out[i] = end - start if np.isnan(out[i]) else max(out[i], end - start)
    return out
```

Add the two arguments alongside the existing ones in `main()`:

```python
    ap.add_argument("--alpha-esc", type=float, default=0.3,
                    help="risk budget for the escalation threshold; must exceed "
                         "--alpha so q_esc < q_safe and escalation precedes violation")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="stall risk budget, separate from alpha (D.10)")
```

- [x] **Step 4: Compute the escalation and dwell constants**

In `main()`, immediately after the `thresholds` loop that prints the per-phase table:

```python
    if args.alpha_esc <= args.alpha:
        raise SystemExit(f"[abort] --alpha-esc ({args.alpha_esc}) must exceed --alpha "
                         f"({args.alpha}); otherwise q_esc >= q_safe and escalation "
                         f"cannot precede violation")

    alpha_esc_p = (args.alpha_esc / len(PHASES)
                   if args.correction == "bonferroni" else args.alpha_esc)
    beta_p = args.beta / len(PHASES) if args.correction == "bonferroni" else args.beta

    esc = np.full(len(PHASES), np.inf)
    for p, i in PHASE_ID.items():
        s = cal_max[:, i]
        esc[i] = conformal_quantile(s[~np.isnan(s)], alpha_esc_p)[0]

    cal_dwell = np.array([dwell_maxima(r) for r in cal])
    dwell = np.full(len(PHASES), np.inf)
    for p, i in PHASE_ID.items():
        s = cal_dwell[:, i]
        dwell[i] = conformal_quantile(s[~np.isnan(s)], beta_p)[0]

    print(f"\n{'phase':11s} {'q_esc':>11s} {'q_safe':>11s} {'d_hat':>9s}")
    for p, i in PHASE_ID.items():
        print(f"{p:11s} {esc[i]:11.4f} {thresholds[i]:11.4f} {dwell[i]:9.1f}")
    print("note: grasp/release dwell is structurally 10 (the +/-5 window), so those")
    print("      two phases can never fire STUCK_TIMEOUT -- reach/transport only.")
```

- [x] **Step 5: Fix the masking asymmetry in `first_alarm`**

Replace `first_alarm` in `scripts/cp_monitor.py`:

```python
    def first_alarm(r):
        """First scorable timestep exceeding this phase's threshold.

        The motion mask is applied HERE as well as in calibration. Calibrating
        with the mask and evaluating without it lets a low-motion step (mean
        err_norm 0.99, max 2.78 -- the extreme tail) raise an alarm calibration
        never saw.
        """
        mask = r.get("motion_mask")
        for t in range(len(r["score"])):
            if mask is not None and not mask[t]:
                continue
            if r["score"][t] > thresholds[r["labels"][t]]:
                return t
        return None
```

- [x] **Step 6: Extend the `--out` payload**

Replace the `res = {...}` dict's tail so it carries every constant plus provenance:

Add `import sys` to the imports at the top of `scripts/cp_monitor.py`, then:

```python
    res.update({
        "escalation_thresholds": dict(zip(PHASES, esc.tolist())),
        "dwell_limits": dict(zip(PHASES, dwell.tolist())),
        "alpha_esc": args.alpha_esc, "beta": args.beta,
        "cal_frac": args.cal_frac, "seed": args.seed,
        "phase_window": 5,
        "motion_floor_abs": float(meta.get("motion_floor_abs", 0.0017320508075688774)),
        "n_cal": len(cal),
        "command": " ".join([".venv-lerobot/bin/python", "-m", "scripts.cp_monitor"]
                            + sys.argv[1:]),
    })
```

- [x] **Step 7: Generate the artefact the demo loads**

```bash
.venv-lerobot/bin/python -m scripts.cp_monitor \
  --rollouts data/rollouts/task0_v10.hdf5 \
  --alpha 0.1 --cal-frac 0.8 --alpha-esc 0.3 --beta 0.1 \
  --out data/results/demo_task0.json
```

- [x] **Step 8: Run tests to verify they pass**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v`
Expected: 6 passed

- [x] **Step 9: Record the number change**

The masking fix moves this repo's published task-0 figures on the seed-0 split:
FAR 0.238 -> 0.190, TPR 0.617 -> 0.447. Append one line to `docs/status.md` §6.4 noting
the fix, the direction, and that `max`-aggregation thresholds are unchanged.

- [x] **Step 10: Stage (commit only if asked)**

```bash
git add scripts/cp_monitor.py tests/test_monitor.py data/results/demo_task0.json docs/status.md
# git commit -m "feat: emit full calibration artefact; mask at alarm time"
```

**Done when** — `data/results/demo_task0.json` exists with
`thresholds.reach == 1.2303 +/- 1e-3`, `dwell_limits.reach == 128.0`,
`escalation_thresholds.transport < thresholds.transport`, and
`.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v` reports `6 passed`.

---

### Task 3: SafetyMonitor — the streaming decision core

**Goal** — `SafetyMonitor.step()` turns one `(action, eef_before, eef_after)` into a
`StatePacket` whose score matches the recorded `err_norm` and whose violation step matches
the batch pipeline exactly.

**Why** — this is the experiment. If the viewer computed any of this itself, the screen
would show a second implementation rather than the pipeline's own answer.

**Files:**
- Modify: `scripts/monitor.py`
- Modify: `tests/test_monitor.py` (append)

**Interfaces:**
- Consumes: `CausalLabeller` (Task 1); the artefact JSON from Task 2
- Produces: `StatePacket` (a `dataclass` with fields `t, phase, score, cmd_norm, scorable,
  q_safe, q_esc, dwell, dwell_limit, realised, violated, escalated, stalled, verdict,
  event`), and
  `SafetyMonitor(artefact: dict)` with `step(action, eef_before, eef_after) -> list[StatePacket]`
  and `flush() -> list[StatePacket]`. `SafetyMonitor.from_file(path) -> SafetyMonitor`.

- [x] **Step 1: Write the failing test**

```python
# append to tests/test_monitor.py
from scripts.monitor import SafetyMonitor

ARTEFACT = "data/results/demo_task0.json"


def _run(rollout_group, monitor):
    """Feed a stored rollout through the monitor; return all packets in order."""
    act = rollout_group["actions"][:]
    eef = rollout_group["eef_pos"][:]
    ex = rollout_group["executed_pos"][:]
    packets = []
    for t in range(len(act)):
        packets += monitor.step(act[t], eef[t], ex[t])
    packets += monitor.flush()
    return packets


def test_streaming_score_matches_recorded_err_norm():
    """The monitor must reproduce collect_rollouts.py's score exactly, or the
    thresholds on screen are being compared against a different quantity."""
    with h5py.File(ROLLOUTS, "r") as f:
        g = f["rollout_0083"]
        packets = _run(g, SafetyMonitor.from_file(ARTEFACT))
        want = g["err_norm"][:]
    assert len(packets) == len(want)
    got = np.array([p.score for p in packets])
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_scorable_flag_matches_recorded_motion_mask():
    with h5py.File(ROLLOUTS, "r") as f:
        g = f["rollout_0075"]  # has 5 masked steps
        packets = _run(g, SafetyMonitor.from_file(ARTEFACT))
        want = g["motion_mask"][:]
    got = np.array([p.scorable for p in packets])
    assert np.array_equal(got, want)


def test_violation_step_matches_batch_first_alarm():
    """Streaming and batch must flag the same INDEX. The streaming monitor
    reports it 5 steps later in wall time; that lag is asserted separately."""
    import json

    from scripts.cp_monitor import load
    a = json.loads(open(ARTEFACT).read())
    th = np.array([a["thresholds"][p] for p in PHASES])
    rollouts, _ = load(ROLLOUTS, "err_norm")
    with h5py.File(ROLLOUTS, "r") as f:
        for name in ("rollout_0083", "rollout_0041", "rollout_0094"):
            r = next(x for x in rollouts if x["name"] == name)
            hit = (r["score"] > th[r["labels"]]) & r["motion_mask"]
            want = int(np.argmax(hit)) if hit.any() else None
            packets = _run(f[name], SafetyMonitor.from_file(ARTEFACT))
            fired = [p.t for p in packets if p.violated]
            got = fired[0] if fired else None
            assert got == want, f"{name}: streaming {got} vs batch {want}"


def test_stall_latches_on_the_known_stall_rollout():
    with h5py.File(ROLLOUTS, "r") as f:
        packets = _run(f["rollout_0076"], SafetyMonitor.from_file(ARTEFACT))
    assert any(p.stalled for p in packets)
    first = next(p for p in packets if p.stalled)
    assert first.phase == "reach"
    assert first.t == pytest.approx(128, abs=2)


def test_realised_motion_tracks_the_osc_compliance_gain():
    """OSC realises ~0.251 of each commanded delta (data/results/action_space.json).
    If this ratio drifts, the score's denominator no longer means what it claims."""
    with h5py.File(ROLLOUTS, "r") as f:
        packets = _run(f["rollout_0041"], SafetyMonitor.from_file(ARTEFACT))
    moving = [p for p in packets if p.scorable]
    gain = np.mean([p.realised / p.cmd_norm for p in moving])
    assert 0.20 < gain < 0.32, f"compliance gain {gain:.3f} outside the measured band"


def test_success_rollout_stays_safe():
    with h5py.File(ROLLOUTS, "r") as f:
        packets = _run(f["rollout_0041"], SafetyMonitor.from_file(ARTEFACT))
    assert not any(p.violated for p in packets)
    assert packets[-1].verdict == "SAFE"


def test_latches_never_clear_and_verdict_ranks_by_severity():
    with h5py.File(ROLLOUTS, "r") as f:
        packets = _run(f["rollout_0083"], SafetyMonitor.from_file(ARTEFACT))
    seen = False
    for p in packets:
        seen = seen or p.violated
        assert p.violated == seen, "violation latch cleared"
        if p.violated:
            assert p.verdict == "VIOLATION"
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -k "streaming or scorable or violation_step or stall or success_rollout or latches" -v`
Expected: FAIL — `ImportError: cannot import name 'SafetyMonitor'`

- [x] **Step 3: Write the implementation**

Append to `scripts/monitor.py`:

```python
import json
from dataclasses import dataclass

import numpy as np

from scripts.subphases import PHASES

POS_SCALE = 0.05  # OSC_POSE output_max; omitting it is wrong by 20x
EPS = 1e-8


@dataclass
class StatePacket:
    """One finalised timestep. Every field is a quantity the algorithm consumes."""

    t: int
    phase: str
    score: float
    cmd_norm: float       # ||action[:3] * 0.05||, metres -- what was commanded
    realised: float       # ||eef_after - eef_before||, metres -- what actually moved
    scorable: bool
    q_safe: float
    q_esc: float
    dwell: int
    dwell_limit: float
    violated: bool
    escalated: bool
    stalled: bool
    verdict: str
    event: str | None


class SafetyMonitor:
    """Streaming sequential-CP monitor. Loads constants; never recalibrates."""

    def __init__(self, artefact):
        self.a = artefact
        self.q_safe = {p: artefact["thresholds"][p] for p in PHASES}
        self.q_esc = {p: artefact["escalation_thresholds"][p] for p in PHASES}
        self.d_hat = {p: artefact["dwell_limits"][p] for p in PHASES}
        self.floor = artefact["motion_floor_abs"]
        self.lab = CausalLabeller(artefact["phase_window"])
        self.scores, self.cmds, self.realised = [], [], []
        self.violated = self.escalated = self.stalled = False
        self.prev_phase, self.dwell = None, 0

    @classmethod
    def from_file(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def check_policy(self, policy_sha256):
        """Refuse to run on a checkpoint the thresholds were not calibrated on.

        A mismatch raises nothing in the stack -- the numbers simply stop meaning
        what they claim -- so it is checked explicitly.
        """
        if policy_sha256 != self.a["policy_sha256"]:
            raise SystemExit(
                f"[abort] policy checkpoint mismatch\n"
                f"  artefact : {self.a['policy_sha256']}\n"
                f"  loaded   : {policy_sha256}\n"
                f"  the coverage guarantee does not transfer across checkpoints."
            )

    def step(self, action, eef_before, eef_after):
        """Feed one control step; return the packets finalised by it."""
        cmd = np.asarray(action[:3], float) * POS_SCALE
        planned = np.asarray(eef_before, float) + cmd
        cmd_norm = float(np.linalg.norm(cmd))
        err = float(np.linalg.norm(planned - np.asarray(eef_after, float)))
        self.scores.append(err / max(cmd_norm, EPS))
        self.cmds.append(cmd_norm)
        # realised motion, for the commanded-vs-actual comparison on screen.
        # OSC is compliant: this lands near 0.251 x cmd_norm (probes/check_tracking_gain).
        self.realised.append(float(np.linalg.norm(
            np.asarray(eef_after, float) - np.asarray(eef_before, float))))
        return [self._decide(i, p) for i, p in self.lab.push(action[6])]

    def flush(self):
        return [self._decide(i, p) for i, p in self.lab.flush()]

    def _decide(self, i, phase_id):
        phase = PHASES[phase_id]
        self.dwell = self.dwell + 1 if phase == self.prev_phase else 1
        self.prev_phase = phase

        score, cmd_norm, realised = self.scores[i], self.cmds[i], self.realised[i]
        scorable = cmd_norm >= self.floor
        q_safe, q_esc, d_hat = self.q_safe[phase], self.q_esc[phase], self.d_hat[phase]

        event = None
        if scorable and score > q_safe and not self.violated:
            self.violated = True
            event = f"{i:4d}  VIOLATION   {phase}  s={score:.3f} > q_safe={q_safe:.3f}"
        elif scorable and score > q_esc and not self.escalated:
            self.escalated = True
            event = (f"{i:4d}  ESCALATE    {phase}  s={score:.3f} > q_esc={q_esc:.3f}"
                     f"  [counterfactual - single-arm rollout]")
        if self.dwell > d_hat and not self.stalled:
            self.stalled = True
            event = f"{i:4d}  STUCK       {phase}  dwell={self.dwell} > d_hat={d_hat:.0f}"

        verdict = ("VIOLATION" if self.violated else
                   "STUCK_TIMEOUT" if self.stalled else
                   "ESCALATE" if self.escalated else "SAFE")
        return StatePacket(i, phase, score, cmd_norm, realised, scorable, q_safe,
                           q_esc, self.dwell, d_hat, self.violated, self.escalated,
                           self.stalled, verdict, event)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v`
Expected: 13 passed

- [x] **Step 5: Stage (commit only if asked)**

```bash
git add scripts/monitor.py tests/test_monitor.py
# git commit -m "feat: streaming SafetyMonitor matching the batch pipeline"
```

**Done when** — `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v` reports
`13 passed`, including `test_violation_step_matches_batch_first_alarm` (streaming and batch
agree on the flagged index for three named rollouts) and
`test_streaming_score_matches_recorded_err_norm` (score matches `err_norm` to 1e-9).

---

### Task 4: Frame sources — replay and live

**Goal** — two generators yielding the identical 4-tuple, so the viewer cannot tell them
apart, and a `--list` mode that names the demo cases by index.

**Why** — replay is the demo that always works and always shows the intended case; live is
the proof it is not a video. Sharing one tuple shape is what stops them drifting into two
different demos.

**Files:**
- Create: `scripts/demo_monitor.py`
- Modify: `tests/test_monitor.py` (append)

**Interfaces:**
- Consumes: `SafetyMonitor` (Task 3)
- Produces: `replay_source(path, rollout_index) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]`
  yielding `(frame_hwc_uint8, action_7, eef_before_3, eef_after_3)`;
  `live_source(task_id, init_index, max_steps, device, monitor) -> Iterator[...]` with the
  same shape, verifying the checkpoint hash when `monitor` is given;
  `list_cases(path, artefact_path) -> list[dict]` with keys `name, success, steps, alarm_step, phases`.

- [x] **Step 1: Write the failing test**

```python
# append to tests/test_monitor.py
from scripts.demo_monitor import list_cases, replay_source


def test_replay_source_yields_the_shapes_the_monitor_expects():
    frames = list(replay_source(ROLLOUTS, 83))
    assert len(frames) == 600
    frame, action, before, after = frames[0]
    assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8
    assert action.shape == (7,)
    assert before.shape == (3,) and after.shape == (3,)


def test_replay_source_reproduces_the_recorded_score_end_to_end():
    """Source -> monitor must equal the stored err_norm. This is what proves the
    replay path feeds the monitor the same experiment collection recorded."""
    mon = SafetyMonitor.from_file(ARTEFACT)
    packets = []
    for frame, action, before, after in replay_source(ROLLOUTS, 83):
        packets += mon.step(action, before, after)
    packets += mon.flush()
    with h5py.File(ROLLOUTS, "r") as f:
        want = f["rollout_0083"]["err_norm"][:]
    np.testing.assert_allclose([p.score for p in packets], want, atol=1e-9)


def test_list_cases_finds_the_three_demo_rollouts():
    cases = {c["name"]: c for c in list_cases(ROLLOUTS, ARTEFACT)}
    assert cases["rollout_0041"]["success"] is True
    assert cases["rollout_0041"]["alarm_step"] is None
    assert cases["rollout_0083"]["success"] is False
    assert cases["rollout_0083"]["alarm_step"] == 40
    assert cases["rollout_0076"]["phases"] == ["reach"]
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -k "replay or list_cases" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.demo_monitor'`

- [x] **Step 3: Write the sources**

```python
# scripts/demo_monitor.py
"""Single-screen experiment viewer for the sequential-CP safety monitor.

  # replay a recorded rollout (always works, deterministic, labelled [REPLAY])
  .venv-lerobot/bin/python -m scripts.demo_monitor \
      --replay data/rollouts/task0_v10.hdf5 -r 83 \
      --artefact data/results/demo_task0.json

  # name the demo cases by index
  .venv-lerobot/bin/python -m scripts.demo_monitor \
      --replay data/rollouts/task0_v10.hdf5 --list \
      --artefact data/results/demo_task0.json

  # drive it from the live simulator (~4.4x slower than real time, stochastic)
  .venv-lerobot/bin/python -m scripts.demo_monitor --live --task-id 0 --init 83 \
      --artefact data/results/demo_task0.json

Both sources yield the identical tuple, so the viewer cannot tell them apart and
the two paths cannot drift into two different demos.
"""

import argparse
import json

import h5py
import numpy as np

from scripts.monitor import SafetyMonitor
from scripts.subphases import PHASES, segment


def replay_source(path, rollout_index):
    """Yield (frame, action, eef_before, eef_after) from a recorded rollout."""
    name = f"rollout_{rollout_index:04d}"
    with h5py.File(path, "r") as f:
        if name not in f:
            raise SystemExit(f"[abort] {name} not in {path}")
        g = f[name]
        frames = g["images"]["agentview"][:]
        actions, eef, ex = g["actions"][:], g["eef_pos"][:], g["executed_pos"][:]
    for t in range(len(actions)):
        yield frames[t], actions[t], eef[t], ex[t]


def live_source(task_id, init_index, max_steps=600, device="mps", monitor=None):
    """Yield the same tuple from a live MuJoCo rollout under the frozen policy.

    ~217 ms/step on MPS -- about 4.4x slower than the 20 Hz control rate -- and
    stochastic: a rollout that failed during collection may not fail again.
    """
    import os

    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from scripts.dp_policy import DiffusionPolicyRunner

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

    pool = np.load(f"data/init_states/task{task_id}_init_160.npy")
    runner = DiffusionPolicyRunner(
        f"yzembodied/libero10_task{task_id}_diffusion", device=device)
    if monitor is not None:
        # Same guard the replay path applies. A mismatched checkpoint raises
        # nothing on its own -- the thresholds just stop meaning what they claim.
        monitor.check_policy(runner.fingerprint())
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(pool[init_index])
        for _ in range(5):  # settle, as in collection
            obs, _, _, _ = env.step(np.zeros(7))
        runner.reset()
        torch.manual_seed(20260805 + task_id * 100000 + init_index)

        done, steps = False, 0
        while steps < max_steps and not done:
            before = obs["robot0_eef_pos"].copy()
            frame = obs["agentview_image"][::-1][::2, ::2].copy()
            action = runner.act(obs)
            obs, _, done, _ = env.step(action)
            yield frame, action, before, obs["robot0_eef_pos"].copy()
            steps += 1
    finally:
        env.close()


def list_cases(path, artefact_path):
    """Name candidate demo rollouts with the step each would first alarm."""
    mon_a = json.load(open(artefact_path))
    th = np.array([mon_a["thresholds"][p] for p in PHASES])
    out = []
    with h5py.File(path, "r") as f:
        for k in sorted(x for x in f if x.startswith("rollout_")
                        and f[x].attrs.get("complete")):
            g = f[k]
            labels, segs = segment(g["actions"][:, 6])
            score, mask = g["err_norm"][:], g["motion_mask"][:]
            hit = (score > th[labels]) & mask
            out.append({
                "name": k,
                "success": bool(g.attrs["success"]),
                "steps": int(g.attrs["steps"]),
                "alarm_step": int(np.argmax(hit)) if hit.any() else None,
                "phases": sorted({p for _, _, p in segs}, key=PHASES.index),
            })
    return out
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v`
Expected: 16 passed

- [x] **Step 5: Verify the live source imports without running a rollout**

Run:
```bash
.venv-lerobot/bin/python -c "
from scripts.demo_monitor import live_source, replay_source, list_cases
print('sources import cleanly')"
```
Expected: `sources import cleanly`

- [x] **Step 6: Stage (commit only if asked)**

```bash
git add scripts/demo_monitor.py tests/test_monitor.py
# git commit -m "feat: replay and live frame sources with a shared tuple shape"
```

**Done when** — `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v` reports
`16 passed`, and `list_cases` returns `alarm_step == 40` for `rollout_0083` and `None` for
`rollout_0041`.

---

### Task 5: The viewer

**Goal** — a single matplotlib window showing the robot, the current decision, the tracking
comparison, the motion/dwell bar, the live signal plot with a shaded escalation band and
phase bands, and a scrolling event log.

**Why** — this is the deliverable. Everything before it is machinery that makes the numbers
on this screen trustworthy.

**Files:**
- Modify: `scripts/demo_monitor.py`

**Interfaces:**
- Consumes: `StatePacket` (Task 3), `replay_source` / `live_source` / `list_cases` (Task 4)
- Produces: `run_viewer(source, monitor, title, meta, save=None, speed=1.0) -> dict` where
  `meta` is `{"policy_sha256": str, "max_steps": int}`, returning
  the run summary; `main()` wired to `--replay/-r`, `--live/--task-id/--init`, `--artefact`,
  `--list`, `--save`, `--speed`.

- [x] **Step 1: Build the figure skeleton**

Append to `scripts/demo_monitor.py`:

```python
def _build_figure(artefact, title):
    """Four regions: robot, decision panel, live signal, event log."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title("Sequential CP safety monitor")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 2, 1],
                  width_ratios=[1, 1], hspace=0.28, wspace=0.12)

    ax_robot = fig.add_subplot(gs[0, 0]); ax_robot.axis("off")
    ax_panel = fig.add_subplot(gs[0, 1]); ax_panel.axis("off")
    ax_sig = fig.add_subplot(gs[1, :])
    ax_log = fig.add_subplot(gs[2, :]); ax_log.axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold", ha="left", x=0.02)
    ax_sig.set_xlabel("timestep"); ax_sig.set_ylabel("err_norm")
    ax_sig.grid(alpha=0.25, linewidth=0.5)
    return fig, ax_robot, ax_panel, ax_sig, ax_log
```

- [x] **Step 2: Draw the phase bands and the escalation band**

```python
PHASE_COLOR = {"reach": "#e8f0fe", "grasp": "#fde8e8",
               "transport": "#e8f6ec", "release": "#fdf3e0"}


def _redraw_signal(ax, packets, artefact, xmax):
    """Score, stepped thresholds, shaded escalation band, phase bands, mask gaps.

    The band matters: q_esc sits only 3-5% below q_safe for reach/grasp/transport,
    so two dashed lines would visually coincide. Shading turns the comparison into
    three legible regions -- below / inside / above.
    """
    ax.clear()
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_xlabel("timestep"); ax.set_ylabel("err_norm")
    ax.set_xlim(0, xmax)
    if not packets:
        return

    t = np.array([p.t for p in packets])
    safe = np.array([p.q_safe for p in packets])
    esc = np.array([p.q_esc for p in packets])

    # phase bands
    start = 0
    for i in range(1, len(packets) + 1):
        if i == len(packets) or packets[i].phase != packets[start].phase:
            ax.axvspan(packets[start].t, packets[i - 1].t + 1,
                       color=PHASE_COLOR[packets[start].phase], zorder=0)
            start = i

    ax.fill_between(t, esc, safe, step="post", color="#f0a202", alpha=0.30,
                    zorder=1, label="escalation band")
    ax.step(t, safe, where="post", color="#c62828", linestyle="--", linewidth=1.8,
            zorder=3, label=r"$\hat{q}_{safe}$ (per phase)")
    ax.step(t, esc, where="post", color="#ef6c00", linestyle=":", linewidth=1.5,
            zorder=3, label=r"$\hat{q}_{esc}$")

    # score: masked steps greyed, never silently plotted as scorable
    score = np.array([p.score for p in packets])
    scorable = np.array([p.scorable for p in packets])
    shown = np.where(scorable, score, np.nan)
    ax.plot(t, shown, color="#1a237e", linewidth=1.8, zorder=4, label="s(t)")
    if (~scorable).any():
        ax.scatter(t[~scorable], score[~scorable], s=10, color="#9e9e9e",
                   zorder=4, label="below motion floor (not scored)")

    fired = [p for p in packets if p.violated]
    if fired:
        ax.axvline(fired[0].t, color="#c62828", linewidth=2, zorder=5)
        ax.scatter([fired[0].t], [fired[0].score], s=70, color="#c62828", zorder=6)
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)
```

- [x] **Step 3: Draw the decision panel, the tracking bars and the motion bar**

```python
VERDICT_COLOR = {"SAFE": "#2e7d32", "ESCALATE": "#ef6c00",
                 "VIOLATION": "#c62828", "STUCK_TIMEOUT": "#6a1b9a"}


def _redraw_panel(ax, p, artefact, meta):
    """Only quantities the algorithm consumes. No confidence, no probabilities."""
    ax.clear(); ax.axis("off")
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0.0, 0.86), 1.0, 0.13, transform=ax.transAxes,
                           color=VERDICT_COLOR[p.verdict], zorder=1))
    ax.text(0.5, 0.925, p.verdict, transform=ax.transAxes, ha="center",
            va="center", fontsize=22, fontweight="bold", color="white", zorder=2)

    rows = [
        ("phase", p.phase),
        ("policy", f"DiffusionPolicy {meta['policy_sha256'][:8]}"),
        ("step", f"{p.t} / {meta['max_steps']}"),
        ("", ""),
        ("s(t)", f"{p.score:.3f}" + ("" if p.scorable else "   (below motion floor)")),
        ("q_esc", f"{p.q_esc:.3f}"),
        ("q_safe", f"{p.q_safe:.3f}"),
        ("", f"{p.score:.3f} {'<' if p.score < p.q_esc else '>'} {p.q_esc:.3f} "
             f"{'<' if p.score < p.q_safe else '>'} {p.q_safe:.3f}"),
        ("", ""),
        ("dwell", f"{p.dwell} / {p.dwell_limit:.0f} steps"),
    ]
    y = 0.78
    for k, v in rows:
        if k or v:
            ax.text(0.02, y, k, transform=ax.transAxes, fontsize=11, color="#555")
            ax.text(0.42, y, v, transform=ax.transAxes, fontsize=11,
                    fontweight="bold" if k in ("phase", "s(t)") else "normal",
                    family="monospace")
        y -= 0.062

    y -= 0.02
    ax.text(0.02, y, "all thresholds", transform=ax.transAxes, fontsize=10,
            color="#555"); y -= 0.05
    for ph in PHASES:
        marker = "> " if ph == p.phase else "  "
        ax.text(0.02, y, f"{marker}{ph:10s} {artefact['thresholds'][ph]:.3f}",
                transform=ax.transAxes, fontsize=10, family="monospace",
                fontweight="bold" if ph == p.phase else "normal",
                color="#000" if ph == p.phase else "#777")
        y -= 0.05


MAX_DELTA = 0.05 * np.sqrt(3)  # largest commandable step, metres


def _draw_tracking(ax, p, y0):
    """Commanded vs realised displacement as two bars on a shared scale.

    This is what makes the score legible without the formula: the policy asked
    for this much motion, the arm delivered this much, and the residual is
    normalised by the request. Both bars are metres; neither is invented.
    """
    ax.text(0.02, y0, "TRACKING", transform=ax.transAxes, fontsize=10,
            color="#555", fontweight="bold")
    for j, (label, val, color) in enumerate(
            [("commanded", p.cmd_norm, "#1a237e"), ("actual", p.realised, "#00897b")]):
        y = y0 - 0.055 * (j + 1)
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=10, color="#555")
        ax.barh(y, 0.42 * min(val / MAX_DELTA, 1.0), height=0.032, left=0.34,
                transform=ax.transAxes, color=color, zorder=3)
        ax.text(0.80, y, f"{val * 1000:6.2f} mm", transform=ax.transAxes,
                fontsize=10, family="monospace")
    ax.text(0.02, y0 - 0.055 * 3, "residual", transform=ax.transAxes, fontsize=10,
            color="#555")
    ax.text(0.34, y0 - 0.055 * 3, f"{p.score:.3f}", transform=ax.transAxes,
            fontsize=10, family="monospace", fontweight="bold")


def _draw_motion(ax, p, y0):
    """Motion floor as a filled bar. A stationary arm has a LOW deviation score,
    which is exactly why stall detection is a separate mechanism."""
    frac = min(p.cmd_norm / MAX_DELTA, 1.0)
    filled = int(round(frac * 16))
    ax.text(0.02, y0, "MOTION", transform=ax.transAxes, fontsize=10,
            color="#555", fontweight="bold")
    ax.text(0.02, y0 - 0.055, "|" + "#" * filled + "." * (16 - filled) + "|",
            transform=ax.transAxes, fontsize=11, family="monospace")
    state = "MOVING" if p.scorable else "BELOW FLOOR"
    ax.text(0.62, y0 - 0.055, state, transform=ax.transAxes, fontsize=11,
            family="monospace", fontweight="bold",
            color="#2e7d32" if p.scorable else "#9e9e9e")
    if p.stalled:
        ax.text(0.62, y0 - 0.110, "STALLED", transform=ax.transAxes, fontsize=11,
                family="monospace", fontweight="bold", color="#6a1b9a")
```

- [x] **Step 4: Wire the animation loop and the run summary**

```python
def run_viewer(source, monitor, title, meta, save=None, speed=1.0):
    """Render packets as they are finalised. Returns the run summary."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    artefact = monitor.a
    fig, ax_robot, ax_panel, ax_sig, ax_log = _build_figure(artefact, title)
    im = ax_robot.imshow(np.zeros((128, 128, 3), np.uint8))
    phase_txt = ax_robot.set_title("", fontsize=18, fontweight="bold")
    packets, log, frames = [], [], []

    def update(item):
        frame, action, before, after = item
        frames.append(frame)
        new = monitor.step(action, before, after)
        if not new:
            im.set_data(frame)
            return
        packets.extend(new)
        p = packets[-1]
        im.set_data(frame)
        phase_txt.set_text(p.phase.upper())
        _redraw_panel(ax_panel, p, artefact, meta)
        _draw_tracking(ax_panel, p, 0.34)
        _draw_motion(ax_panel, p, 0.13)
        _redraw_signal(ax_sig, packets, artefact, meta["max_steps"])
        for q in new:
            if q.event:
                log.append(q.event)
        ax_log.clear(); ax_log.axis("off")
        for i, line in enumerate(log[-5:]):
            ax_log.text(0.01, 0.85 - i * 0.19, line, transform=ax_log.transAxes,
                        fontsize=10, family="monospace")

    anim = FuncAnimation(fig, update, frames=source, interval=max(1, int(50 / speed)),
                         repeat=False, cache_frame_data=False)
    if save:
        anim.save(save, writer="ffmpeg", fps=max(1, int(20 * speed)), dpi=110)
        print(f"wrote {save}")
    else:
        plt.show()

    for q in monitor.flush():
        packets.append(q)
    return _summary(packets, meta)


def _summary(packets, meta):
    """Reports what exists. Expensive-policy fraction is OMITTED, not zeroed --
    there is no second arm to measure."""
    if not packets:
        return {}
    score = np.array([p.score for p in packets])
    scorable = np.array([p.scorable for p in packets])
    viol = [p for p in packets if p.violated]
    esc = [p for p in packets if p.escalated]
    stall = [p for p in packets if p.stalled]
    s = {
        "steps": len(packets),
        "verdict": packets[-1].verdict,
        "violation_step": viol[0].t if viol else None,
        "escalation_step": esc[0].t if esc else None,
        "stall_step": stall[0].t if stall else None,
        "steps_above_q_safe": float(np.mean(
            [p.score > p.q_safe for p in packets if p.scorable])),
        "steps_above_q_esc": float(np.mean(
            [p.score > p.q_esc for p in packets if p.scorable])),
        "steps_masked": int((~scorable).sum()),
        "max_err_norm": float(score[scorable].max()) if scorable.any() else None,
        "policy": f"DiffusionPolicy {meta['policy_sha256'][:8]} "
                  f"(single arm - no arbitration in this rollout)",
    }
    print("\n" + "=" * 62)
    print("  RUN SUMMARY")
    print("=" * 62)
    for k, v in s.items():
        print(f"  {k:22s} {v}")
    if s["escalation_step"] is not None:
        print("  escalation is COUNTERFACTUAL: no second policy arm exists yet")
    print("=" * 62)
    return s
```

- [x] **Step 5: Wire `main()`**

```python
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artefact", default="data/results/demo_task0.json")
    ap.add_argument("--replay", default=None, help="path to a collected task*.hdf5")
    ap.add_argument("-r", "--rollout", type=int, default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="name candidate demo cases")
    ap.add_argument("--save", default=None, help="write an mp4 instead of showing")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    monitor = SafetyMonitor.from_file(args.artefact)

    if args.list:
        if not args.replay:
            raise SystemExit("[abort] --list needs --replay <hdf5>")
        rows = list_cases(args.replay, args.artefact)
        print(f"{'rollout':>14} {'outcome':>8} {'steps':>6} {'alarm':>6}  phases")
        for c in rows:
            print(f"{c['name']:>14} {'SUCCESS' if c['success'] else 'fail':>8} "
                  f"{c['steps']:>6} {str(c['alarm_step']):>6}  {','.join(c['phases'])}")
        return

    if args.live:
        meta = {"policy_sha256": monitor.a["policy_sha256"], "max_steps": 600}
        src = live_source(args.task_id, args.init, monitor=monitor)
        title = f"[LIVE] task {args.task_id} - init {args.init}"
    else:
        if args.replay is None or args.rollout is None:
            raise SystemExit("[abort] need --replay <hdf5> -r <index>, or --live")
        with h5py.File(args.replay, "r") as f:
            sha = str(f.attrs["policy_sha256"])
            steps = int(f[f"rollout_{args.rollout:04d}"].attrs["steps"])
        monitor.check_policy(sha)
        meta = {"policy_sha256": sha, "max_steps": steps}
        src = replay_source(args.replay, args.rollout)
        title = f"[REPLAY] task 0 - rollout_{args.rollout:04d}"

    print(f"{title}\nartefact: {args.artefact}")
    print(f"produced by: {monitor.a['command']}")
    run_viewer(src, monitor, title, meta, save=args.save, speed=args.speed)


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Render each demo case to mp4 (the recorded backup)**

```bash
for R in 41 83 76; do
  .venv-lerobot/bin/python -m scripts.demo_monitor \
    --replay data/rollouts/task0_v10.hdf5 -r $R \
    --artefact data/results/demo_task0.json \
    --save data/results/demo_r$R.mp4
done
ls -la data/results/demo_r*.mp4
```

Expected: three non-empty mp4 files. `demo_r83.mp4` prints
`violation_step 40`; `demo_r41.mp4` prints `verdict SAFE`; `demo_r76.mp4` prints a non-null
`stall_step`.

- [x] **Step 7: Show it on screen once**

```bash
.venv-lerobot/bin/python -m scripts.demo_monitor \
  --replay data/rollouts/task0_v10.hdf5 -r 83 \
  --artefact data/results/demo_task0.json
```

Expected: a window opens; the dashed `q_safe` line visibly **steps** at phase boundaries; the
escalation band is visible as a shaded strip; the red rule appears at step 40 and stays.

- [x] **Step 8: Stage (commit only if asked)**

```bash
git add scripts/demo_monitor.py
# git commit -m "feat: single-screen experiment viewer"
```

**Done when** — the three mp4s exist and their printed summaries carry
`violation_step 40` (r83), `verdict SAFE` (r41), and a non-null `stall_step` (r76); and the
on-screen run shows the threshold line stepping at phase boundaries.

---

### Task 6: Live-mode smoke run

**Goal** — one real live rollout completes end to end and its summary is consistent with the
replay path.

**Why** — the live path shares the monitor but not the frame source, so nothing before this
task exercises `OffScreenRenderEnv` + `DiffusionPolicyRunner` under the viewer. It is the
only way to know the `[LIVE]` mode works before it is needed in front of someone.

**Files:**
- Modify: `scripts/demo_monitor.py` (only if the smoke run reveals a defect)

- [x] **Step 1: Confirm the runner API is what `live_source` assumes**

`scripts/dp_policy.py` defines `act(obs) -> np.ndarray[7]` (line 120) and `fingerprint() -> str`
(line 139); `live_source` uses both. Confirm they are still there:

```bash
.venv-lerobot/bin/python -c "
from scripts.dp_policy import DiffusionPolicyRunner as R
assert hasattr(R, 'act') and hasattr(R, 'fingerprint')
print('runner API ok')"
```
Expected: `runner API ok`

- [x] **Step 2: Run one live rollout to mp4**

```bash
time .venv-lerobot/bin/python -m scripts.demo_monitor \
  --live --task-id 0 --init 83 \
  --artefact data/results/demo_task0.json \
  --save data/results/demo_live.mp4
```
Expected: completes in roughly 2-5 minutes (~217 ms/step); prints a run summary; writes a
non-empty `data/results/demo_live.mp4`.

- [x] **Step 3: Confirm the policy-mismatch guard actually refuses**

```bash
.venv-lerobot/bin/python -c "
import json
from scripts.monitor import SafetyMonitor
m = SafetyMonitor.from_file('data/results/demo_task0.json')
try:
    m.check_policy('deadbeef' * 8)
    print('FAIL: mismatch was accepted')
except SystemExit as e:
    print('OK: refused ->', str(e).splitlines()[0])"
```
Expected: `OK: refused -> [abort] policy checkpoint mismatch`

- [x] **Step 4: Full test suite**

Run: `.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v`
Expected: 16 passed

- [x] **Step 5: Stage (commit only if asked)**

```bash
git add -A scripts/ tests/ data/results/demo_task0.json
# git commit -m "test: live-mode smoke run"
```

**Done when** — `data/results/demo_live.mp4` exists and is non-empty, the mismatch guard
prints `OK: refused`, and the full suite reports `15 passed`.

---

## Deferred, deliberately

These are in the brief but not in this plan. Each needs something that does not exist yet.

| item | blocked on |
|---|---|
| Real BC-RNN / ACT policy row and switch animation | neither policy is trained; no arbiter code in this repo |
| Expensive-policy fraction, switch count in the summary | same — omitted rather than zeroed |
| Flat-CP side-by-side (brief §5.4) | needs a second calibration arm this repo lacks |
| GPIO / Jetson LED output | hardware unconfirmed (`status.md` §4) |
| Resolving the threshold discrepancy with `scp/` | `status.md` §8 item 1; out of scope here |


---

## Execution record — 2026-09-09

All six tasks executed inline. Final state: **17 tests passing** (the plan projected 16;
one was added during execution, below), artefact at `data/results/demo_task0.json`, four
mp4s at `data/results/demo_r{41,83,76}.mp4` and `data/results/demo_live.mp4`.

### Deviations from the plan

**1. A defect the plan did not anticipate: escalation reported after violation.**
`_decide` used `if violated ... elif escalated ...`. Because `q_esc < q_safe`, a step
crossing `q_safe` necessarily crosses `q_esc`, but the `elif` left `escalated` False on
that step; it then latched on some later, milder step. `rollout_0076` reported violation
at 536 and escalation at **537** — semantically impossible. Fixed by making the two
independent latches (the violation event still supersedes the escalate line when both
cross on the same step). Regression test added:
`test_escalation_never_reported_after_violation`, which is the 17th test.

After the fix, `escalation_step == violation_step` on both failing demo cases — which
matches the design spec's measured finding that escalation precedes violation by 0 steps
in 21/21 paired cases.

**2. Three legibility defects, found by looking at the rendered output.**
All 16 tests passed while the screen was wrong:
- tracking bars scaled to `MAX_DELTA` (86.6 mm), a value measured commands never approach
  (p50 18.9 mm, p90 40.5 mm) — a typical command drew a 22% sliver. Rescaled to a 40 mm
  full scale, clipping 10.8% of steps, with the numeric value always printed.
- `MOVING` and `STALLED` rendered side by side, reading as a contradiction. They describe
  different things — the current step versus a latch from an earlier one — so the latch
  moved to its own line carrying the dwell figures that fired it.
- the signal legend sat on top of the trace; moved above the axes.

**3. `git_revision` in the fingerprint was left alone.** Nothing was committed, per the
global constraint. The working tree carries the changes.

### Verification evidence

```
.venv-lerobot/bin/python -m pytest tests/test_monitor.py -v   ->  17 passed

thresholds             {reach 1.2303, grasp 1.3213, transport 1.0721, release 1.9220}
escalation_thresholds  {reach 1.1387, grasp 1.1850, transport 1.0369, release 1.4575}
dwell_limits           {reach 128.0,  grasp 10.0,   transport 80.0,   release 10.0}

demo_r41  verdict SAFE       violation None  stall None
demo_r83  verdict VIOLATION  violation 40    stall 293
demo_r76  verdict VIOLATION  violation 536   stall 128
demo_live verdict VIOLATION  violation 40    stall 293   (3:00 wall, 600 steps)
```

The live run reproduced `rollout_0083` to 13 significant figures (`max_err_norm`
1.278087500260789 live vs 1.278087500260813 replay), which is the strongest available
evidence that the live and replay paths are the same experiment rather than two.
