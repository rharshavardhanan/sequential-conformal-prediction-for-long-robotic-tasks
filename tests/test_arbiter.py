"""Arbiter tests. Pure -- no simulator, no policy, no rendering."""

from dataclasses import dataclass

import pytest

from scripts.arbiter import Arbiter


@dataclass
class Pkt:
    """Minimal stand-in for monitor.StatePacket."""
    score: float
    q_esc: float = 1.0
    q_safe: float = 1.2
    scorable: bool = True
    escalated: bool = False
    violated: bool = False
    stalled: bool = False


def test_starts_on_the_cheap_arm():
    a = Arbiter("cheap", "exp")
    d = a.step(Pkt(0.7))
    assert d.selected_policy == "cheap" and d.compute_class == "cheap"
    assert not d.switched and a.switches == 0


def test_escalates_the_step_the_score_crosses_q_esc():
    a = Arbiter("cheap", "exp")
    a.step(Pkt(0.7))
    d = a.step(Pkt(1.05))                      # q_esc = 1.0
    assert d.switched and d.selected_policy == "exp"
    assert d.compute_class == "expensive"
    assert "q_esc" in d.reason and d.event


def test_does_not_de_escalate_before_the_hysteresis_window():
    a = Arbiter("cheap", "exp", calm_steps=3)
    a.step(Pkt(1.05))
    for _ in range(2):
        d = a.step(Pkt(0.5))
        assert d.selected_policy == "exp", "de-escalated too early"
    d = a.step(Pkt(0.5))
    assert d.switched and d.selected_policy == "cheap"
    assert a.switches == 2


def test_latched_escalation_does_not_pin_the_arbiter_forever():
    """Regression: driving switching from the monitor's LATCHED escalated flag
    makes de-escalation impossible, because the latch never clears."""
    a = Arbiter("cheap", "exp", calm_steps=2)
    a.step(Pkt(1.05, escalated=True))
    a.step(Pkt(0.5, escalated=True))           # latch still set, score in band
    d = a.step(Pkt(0.5, escalated=True))
    assert d.selected_policy == "cheap", "latch pinned the arbiter to expensive"


def test_unscorable_steps_never_escalate():
    """Below the motion floor the score is not comparable to a threshold."""
    a = Arbiter("cheap", "exp")
    d = a.step(Pkt(99.0, scorable=False))
    assert not d.switched and d.selected_policy == "cheap"


def test_stall_escalates_even_with_a_low_score():
    """A stalled arm has a LOW deviation score -- that is why dwell is separate."""
    a = Arbiter("cheap", "exp")
    d = a.step(Pkt(0.3, stalled=True))
    assert d.switched and d.selected_policy == "exp"
    assert "stall" in d.reason


def test_no_expensive_arm_reports_a_counterfactual_and_never_switches():
    a = Arbiter("cheap", expensive_policy=None)
    d = a.step(Pkt(1.05))
    assert d.selected_policy == "cheap" and not d.switched
    assert "no expensive arm" in d.reason
    assert a.summary()["expensive_fraction"] is None


def test_chunk_boundary_defers_the_switch():
    a = Arbiter("cheap", "exp", chunk_boundary_only=True)
    d = a.step(Pkt(1.05), at_chunk_boundary=False)
    assert not d.switched and "deferred" in d.reason
    d = a.step(Pkt(1.05), at_chunk_boundary=True)
    assert d.switched


def test_summary_counts_switches_and_compute_fraction():
    a = Arbiter("cheap", "exp", calm_steps=2)
    for p in [Pkt(0.5), Pkt(1.05), Pkt(0.5), Pkt(0.5), Pkt(0.5)]:
        a.step(p)
    s = a.summary()
    assert s["switches"] == 2
    assert sum(s["steps_on"].values()) == 5
    assert 0.0 < s["expensive_fraction"] < 1.0
    assert s["fraction_on"]["cheap"] + s["fraction_on"]["exp"] == pytest.approx(1.0)
