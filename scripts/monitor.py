"""Causal streaming monitor for the sequential-CP safety demo.

The offline pipeline segments a completed trajectory. A live monitor cannot.
`subphases.segment()` relabels indices d-4..d+5 around a transition at diff-index
d, and d is only observable at time d+1 -- so the label for index i is final only
once time i+5 has been seen. This module emits at that lag, which makes its labels
provably identical to the ones calibration used, at a cost of 250 ms at 20 Hz.
"""

import json
from dataclasses import dataclass

import numpy as np

from scripts.subphases import PHASE_ID, PHASES

POS_SCALE = 0.05  # OSC_POSE output_max; omitting it is wrong by 20x
EPS = 1e-8


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
    event: "str | None"


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

        # q_esc < q_safe by construction, so crossing q_safe implies crossing
        # q_esc. These are independent latches, NOT alternatives -- an `elif`
        # here lets a violating step leave `escalated` False, which then latches
        # on some later, milder step and reports escalation AFTER violation.
        event = None
        if scorable and score > q_esc and not self.escalated:
            self.escalated = True
            event = (f"{i:4d}  q_esc CROSSED  {phase}  "
                     f"s={score:.3f} > q_esc={q_esc:.3f}"
                     f"   (an arbiter would escalate here)")
        if scorable and score > q_safe and not self.violated:
            self.violated = True
            # supersedes the escalate line when both cross on the same step
            event = f"{i:4d}  VIOLATION   {phase}  s={score:.3f} > q_safe={q_safe:.3f}"
        if self.dwell > d_hat and not self.stalled:
            self.stalled = True
            event = f"{i:4d}  STUCK       {phase}  dwell={self.dwell} > d_hat={d_hat:.0f}"

        verdict = ("VIOLATION" if self.violated else
                   "STUCK_TIMEOUT" if self.stalled else
                   "ESCALATE" if self.escalated else "SAFE")
        return StatePacket(i, phase, score, cmd_norm, realised, scorable, q_safe,
                           q_esc, self.dwell, d_hat, self.violated, self.escalated,
                           self.stalled, verdict, event)
