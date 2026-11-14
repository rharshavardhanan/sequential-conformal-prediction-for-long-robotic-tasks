"""Policy arbitration: choose the cheap or the expensive arm, per control step.

Pure. Takes numbers, returns a decision. No simulator, no policy objects, no
rendering -- which is what makes it testable without MuJoCo and keeps it out of
the GUI.

What the guarantee actually is (docs/status.md section 4): the CP coverage
guarantee does NOT transfer to an arbitrated trajectory, because arbitration
changes which policy generates the data. What transfers is a bounded
SPURIOUS-ESCALATION rate: q_esc is drawn from the same calibration set at a
larger budget alpha_esc > alpha, so q_esc < q_safe and escalation happens
strictly before a safety violation would. This module therefore never labels an
arbitrated run "CP-certified safe".

A naive "escalate at a fraction of q_safe" rule was tested and rejected -- it
escalates below the calibrated quantile, so no bound covers it (0.9x gave 56.7%
spurious escalation against a 10% budget). Only the dual-threshold rule is
implemented here.
"""

from dataclasses import dataclass, field

CHEAP, EXPENSIVE = "cheap", "expensive"


@dataclass
class ArbitrationDecision:
    """What the arbiter chose this step, and why."""

    selected_policy: str
    compute_class: str
    reason: str
    escalated: bool          # latched: has escalation ever fired this episode
    violated: bool
    stalled: bool
    switched: bool           # did the selected policy change THIS step
    event: "str | None" = None


@dataclass
class Arbiter:
    """Immediate escalation, hysteretic de-escalation.

    Asymmetric on purpose: escalate the instant the score leaves the calibrated
    band, but return to the cheap arm only after `calm_steps` consecutive
    in-band steps. Symmetric switching thrashes -- measured at 6.7 transitions
    per failing episode, cut to 1.5 by hysteresis.
    """

    cheap_policy: str
    expensive_policy: "str | None" = None      # None => nothing to escalate TO
    calm_steps: int = 20
    chunk_boundary_only: bool = False

    current: str = field(default=None)
    escalated: bool = field(default=False)
    _calm: int = field(default=0)
    switches: int = field(default=0)
    steps_on: dict = field(default_factory=dict)

    def __post_init__(self):
        self.current = self.cheap_policy
        self.steps_on = {self.cheap_policy: 0}
        if self.expensive_policy:
            self.steps_on[self.expensive_policy] = 0

    def step(self, packet, at_chunk_boundary=True):
        """Decide from one monitor StatePacket. Returns ArbitrationDecision."""
        want, reason = self.current, "steady"

        # INSTANTANEOUS band membership, not the monitor's latch. packet.escalated
        # is latched for the whole episode by design (the safety claim is about
        # the rollout, not the instant), so driving switching from it would pin
        # the arbiter to the expensive arm forever and de-escalation could never
        # fire. The raw comparison is the live signal.
        out_of_band = packet.scorable and packet.score > packet.q_esc
        in_band = not out_of_band
        if out_of_band:
            self.escalated = True

        if self.expensive_policy is None:
            # Nothing to escalate to. Report the counterfactual honestly rather
            # than pretending a switch happened.
            reason = ("would escalate (no expensive arm available)"
                      if out_of_band else "steady")
        else:
            if out_of_band or packet.stalled:
                if self.current != self.expensive_policy:
                    want = self.expensive_policy
                    reason = ("stall detected" if packet.stalled and not out_of_band
                              else "score crossed q_esc")
                self._calm = 0
            elif self.current == self.expensive_policy:
                self._calm = self._calm + 1 if in_band else 0
                if self._calm >= self.calm_steps:
                    want = self.cheap_policy
                    reason = f"{self.calm_steps} calm steps"

        blocked = self.chunk_boundary_only and not at_chunk_boundary
        switched = want != self.current and not blocked
        if blocked and want != self.current:
            reason += " (deferred to chunk boundary)"
        if switched:
            prev = self.current
            self.current = want
            self.switches += 1
            self._calm = 0
            event = f"{prev} -> {want}  ({reason})"
        else:
            event = None

        self.steps_on[self.current] = self.steps_on.get(self.current, 0) + 1
        cls = EXPENSIVE if self.current == self.expensive_policy else CHEAP
        return ArbitrationDecision(
            selected_policy=self.current, compute_class=cls, reason=reason,
            escalated=self.escalated, violated=packet.violated,
            stalled=packet.stalled, switched=switched, event=event)

    def summary(self):
        total = sum(self.steps_on.values()) or 1
        return {
            "switches": self.switches,
            "steps_on": dict(self.steps_on),
            "fraction_on": {k: v / total for k, v in self.steps_on.items()},
            "expensive_fraction": (
                self.steps_on.get(self.expensive_policy, 0) / total
                if self.expensive_policy else None),
        }
