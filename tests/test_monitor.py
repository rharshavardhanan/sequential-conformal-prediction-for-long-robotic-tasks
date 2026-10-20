"""Tests for the causal streaming monitor behind the live demo.

Run: .venv-lerobot/bin/python -m pytest tests/test_monitor.py -v
"""

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


import json
import subprocess

from scripts.cp_monitor import dwell_maxima, load

ARTEFACT = "data/results/demo_task0.json"


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
    a = json.loads(open(ARTEFACT).read())
    assert a["thresholds"]["reach"] == pytest.approx(1.2303, abs=1e-3)
    assert a["thresholds"]["grasp"] == pytest.approx(1.3213, abs=1e-3)
    assert a["thresholds"]["transport"] == pytest.approx(1.0721, abs=1e-3)
    assert a["thresholds"]["release"] == pytest.approx(1.9220, abs=1e-3)
    assert a["dwell_limits"]["reach"] == pytest.approx(128.0)
    assert a["dwell_limits"]["transport"] == pytest.approx(80.0)
    assert a["policy_sha256"].startswith("3105133c")


from scripts.monitor import SafetyMonitor


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


def test_escalation_never_reported_after_violation():
    """q_esc < q_safe, so crossing q_safe implies crossing q_esc. The two latches
    are independent: an `elif` lets a violating step leave `escalated` False,
    which then latches on a later milder step and reports escalation AFTER
    violation -- which is semantically impossible.
    """
    with h5py.File(ROLLOUTS, "r") as f:
        for name in ("rollout_0083", "rollout_0076", "rollout_0094"):
            packets = _run(f[name], SafetyMonitor.from_file(ARTEFACT))
            viol = [p.t for p in packets if p.violated]
            esc = [p.t for p in packets if p.escalated]
            if viol:
                assert esc, f"{name}: violated but never escalated"
                assert esc[0] <= viol[0], (
                    f"{name}: escalation at {esc[0]} after violation at {viol[0]}")
            for p in packets:
                assert not (p.violated and not p.escalated), (
                    f"{name} step {p.t}: violated without escalating")


def test_task_repo_encodes_the_upstream_typo():
    """Tasks 3 and 8 live under 'libeo10', not 'libero10'. The obvious pattern
    404s on exactly those two, which is why the map exists."""
    from scripts.demo_monitor import task_repo
    assert task_repo(0) == "yzembodied/libero10_task0_diffusion"
    assert task_repo(1) == "yzembodied/libero10_task1_diffusion"
    assert task_repo(3) == "yzembodied/libeo10_task3_diffusion"
    assert task_repo(8) == "yzembodied/libeo10_task8_diffusion"
