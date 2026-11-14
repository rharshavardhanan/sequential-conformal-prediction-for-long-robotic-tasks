"""Cheap<->expert arbitration driven by a conformal escalation threshold.

CHEAP by default. Escalate the instant the score exceeds q_esc for the CURRENT phase.
Hand back only after 8 consecutive in-band steps, so a single quiet step does not bounce
control back and forth. chunk_boundary_only = False: escalation may happen at any step.

The score is ensemble disagreement, selected by Gate 6 (held-out push AUC@8 = 0.813). It
is computed from the observation BEFORE the action is taken, so the decision is causal --
the arbiter never sees the consequence of the step it is deciding about.
"""

import numpy as np
from composed import PUSH, REACH

HANDBACK_STEPS = 8
CHEAP, EXPERT = 0, 1


class Arbiter:
    def __init__(self, cheap, ensemble, expert, q_esc, handback=HANDBACK_STEPS,
                 random_rate=None, rng=None):
        """q_esc maps phase -> threshold. `random_rate` replaces the score test with a
        Bernoulli draw at that per-step rate: the matched-rate control arm."""
        self.cheap, self.ens, self.expert = cheap, ensemble, expert
        self.q_esc = q_esc
        self.handback = handback
        self.random_rate = random_rate
        self.rng = rng or np.random.default_rng(0)
        self.reset_stats()

    def reset_stats(self):
        self.mode = CHEAP
        self.in_band = 0
        self.expert_steps = 0
        self.expert_calls = 0          # contiguous expert spans, i.e. hand-offs to expert
        self.switches = 0
        self.first_esc_step = None
        self.first_esc_phase = None
        self.n_steps = 0
        self.n_cheap_steps = 0
        self.n_trigger_evals = 0
        self.n_trigger_fires = 0
        self.trace = []

    def score(self, obs):
        return float(np.stack([m.act(obs) for m in self.ens]).std(axis=0).mean())

    def act(self, env):
        obs = env.obs26()
        s = self.score(obs)
        phase = env.phase
        q = self.q_esc[phase]
        self.n_steps += 1

        if self.mode == CHEAP:
            self.n_cheap_steps += 1
            self.n_trigger_evals += 1
            fire = (self.rng.random() < self.random_rate) if self.random_rate is not None \
                else (s > q)
            if fire:
                self.n_trigger_fires += 1
                self.mode = EXPERT
                self.in_band = 0
                self.expert_calls += 1
                self.switches += 1
                if self.first_esc_step is None:
                    self.first_esc_step = self.n_steps
                    self.first_esc_phase = int(phase)
        else:
            # hand back only after a sustained quiet stretch
            self.in_band = self.in_band + 1 if s <= q else 0
            if self.in_band >= self.handback:
                self.mode = CHEAP
                self.in_band = 0
                self.switches += 1

        if self.mode == EXPERT:
            self.expert_steps += 1
            a, tag = self.expert.act(env)
        else:
            a, tag = self.cheap.act(obs), "cheap"

        self.trace.append({"step": self.n_steps, "phase": int(phase), "score": s,
                           "q_esc": float(q), "mode": int(self.mode), "tag": tag})
        return a, tag

    def stats(self):
        return {
            "expert_calls": self.expert_calls,
            "expert_steps": self.expert_steps,
            "steps": self.n_steps,
            "expert_step_frac": self.expert_steps / max(self.n_steps, 1),
            "switches": self.switches,
            "first_escalation_step": self.first_esc_step,
            "first_escalation_phase": self.first_esc_phase,
            "trigger_fire_rate_among_cheap_steps":
                self.n_trigger_fires / max(self.n_trigger_evals, 1),
            "reason": "score > q_esc[phase]" if self.random_rate is None
                      else f"bernoulli({self.random_rate:.4f}) matched-rate control",
        }


def phase_q(cal_json, alpha_esc, K=2, score="b_ensemble_disagreement"):
    """Pull q_esc[phase] out of calibration.json for a given alpha_esc."""
    t = cal_json["thresholds"][score]
    if K == 2:
        return {REACH: t["K2"]["reach"]["q_esc"][f"{alpha_esc:.2f}"]["q"],
                PUSH: t["K2"]["push"]["q_esc"][f"{alpha_esc:.2f}"]["q"]}
    q = t["K1"]["pooled"]["q_esc"][f"{alpha_esc:.2f}"]["q"]
    return {REACH: q, PUSH: q}
