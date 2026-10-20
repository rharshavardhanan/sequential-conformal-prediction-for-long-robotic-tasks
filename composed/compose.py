"""Composed two-phase task: Reach -> Slide (or Reach -> Push with COMPOSED_TASK=push), in one scene.

Phase A (reach): reach-goal sampled from PandaReach-v3's own goal box
  ([-0.15,0.15] x [-0.15,0.15] x [0,0.3], robot base at [-0.6,0,0] in both envs), with a
  per-seed RNG.  Policy A sees exactly PandaReach's observation: ee pos+vel (6), goal (3).
  Success: ||ee - reach_goal|| < 0.05 (Reach's threshold).  Budget 50 steps.
  Phase-A failure ENDS the episode as a failure (compounding premise).
Phase B (slide): starts from wherever phase A left the arm AND the puck.  Policy B sees
  PandaSlide's native observation.  Success: Slide's own is_success.  Budget 50 steps.
Episode success = both phases succeed.  Horizon <= 100 steps.  Each phase ends at its
own success, matching the native v3 success-termination the sub-skills were verified under.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
from verify_subskills import Policy  # noqa: E402

import gymnasium as gym  # noqa: E402
import panda_gym  # noqa: E402,F401

MODELS = os.path.join(HERE, "models")
BUDGET = {0: 50, 1: 50}
HORIZON = BUDGET[0] + BUDGET[1]
REACH_LOW = np.array([-0.15, -0.15, 0.0])
REACH_HIGH = np.array([0.15, 0.15, 0.3])
REACH_THR = 0.05
EXPERT_A = ("PandaReach-v3", "chencliu/tqc-PandaReach-v3", "tqc-PandaReach-v3.zip", "tqc")
CHEAP, EXPENSIVE = C.CHEAP, C.EXPENSIVE

# Phase-B task.  COMPOSED_TASK=slide (default; first composition, gate failed) or
# COMPOSED_TASK=push (recommended follow-up; fine-tuned Push expert).  Each writes its own results dir.
TASK = os.environ.get("COMPOSED_TASK", "slide")
if TASK == "slide":
    RESULTS = os.path.join(HERE, "results")
    SCENE = "PandaSlide-v3"
    REST_Z = 0.015                     # puck: cylinder height 0.03
    EXPERT_B = ("PandaSlide-v3", "chencliu/tqc-PandaSlide-v3", "tqc-PandaSlide-v3.zip", "tqc")
elif TASK == "push":
    import json as _json
    RESULTS = os.path.join(HERE, "results_push")
    SCENE = "PandaPush-v3"
    REST_Z = 0.02                      # cube: size 0.04
    _v = _json.load(open(os.path.join(HERE, "results", "push_ft_verify.json")))
    _ft, _d = _v["chosen"], os.path.join(MODELS, _v["chosen_dir"])
    EXPERT_B = ("PandaPush-v3", "local:" + os.path.join(_d, _ft + "_vecnorm.pkl"), os.path.join(_d, _ft + ".zip"), "tqc")
else:
    raise ValueError(TASK)
PHASES = ("reach", TASK)
os.makedirs(RESULTS, exist_ok=True)


# PyBullet carries hidden contact/solver state across env.reset(): on the same seed, a shared env
# flipped 7/40 cheap-policy outcomes (36/40 trajectories differed) depending on episode order, while
# a fresh env per episode was bit-identical (0/40).  DETERMINISTIC_RESET=1 rebuilds the env every
# episode.  Default ON for push (all Reach->Push results from the determinism fix onward), OFF for
# slide (keeps the earlier Reach->Slide artefacts reproducible).
FRESH = os.environ.get("DETERMINISTIC_RESET", "1" if TASK == "push" else "0") == "1"


class Composed:
    def __init__(self, render=False):
        self.render = render
        self._make()

    def _make(self):
        self.env = gym.make(SCENE, render_mode="rgb_array") if self.render else gym.make(SCENE)
        self.base = self.env.unwrapped

    def reset(self, seed):
        if FRESH:
            self.env.close()
            self._make()
        obs, _ = self.env.reset(seed=seed)
        self.reach_goal = np.random.default_rng(seed + 7_000_000).uniform(REACH_LOW, REACH_HIGH)
        self.phase, self.t, self.t_phase = 0, 0, 0
        self.slide_obs = obs
        self.puck0 = obs["observation"][6:9].copy()
        self.slide_goal = obs["desired_goal"].copy()
        return self.obs_for_phase()

    def obs_for_phase(self):
        o = self.slide_obs
        if self.phase == 0:
            return {"observation": o["observation"][:6].copy(), "achieved_goal": o["observation"][:3].copy(),
                    "desired_goal": self.reach_goal.copy()}
        return {k: v.copy() for k, v in o.items()}

    def flat(self):
        """Cheap-policy input: slide obs (18) + current phase goal (3) + phase one-hot (2)."""
        o = self.slide_obs["observation"]
        g = self.reach_goal if self.phase == 0 else self.slide_obs["desired_goal"]
        return np.concatenate([o, g, np.eye(2)[self.phase]]).astype(np.float32)

    def step(self, action):
        """Returns (event, info): event in {'continue', 'phase_switch', 'success', 'fail'}."""
        obs, _, _, _, info = self.base.step(action)
        self.slide_obs = obs
        self.t += 1
        self.t_phase += 1
        if self.phase == 0:
            if np.linalg.norm(obs["observation"][:3] - self.reach_goal) < REACH_THR:
                self.boundary = self.boundary_state()
                self.phase, self.t_phase = 1, 0
                return "phase_switch", info
            return ("fail" if self.t_phase >= BUDGET[0] else "continue"), info
        if info["is_success"]:
            return "success", info
        return ("fail" if self.t_phase >= BUDGET[1] else "continue"), info

    def boundary_state(self):
        o = self.slide_obs["observation"]
        return {"t": self.t, "ee": o[:3].tolist(), "puck": o[6:9].tolist(),
                "puck_displacement_in_phaseA": float(np.linalg.norm(o[6:8] - self.puck0[:2])),
                "puck_z": float(o[8]), "puck_speed": float(np.linalg.norm(o[12:15])),
                "puck_goal_dist": float(np.linalg.norm(o[6:9] - self.slide_goal)),
                "puck_goal_dist_initial": float(np.linalg.norm(self.puck0 - self.slide_goal))}


class Experts:
    def __init__(self):
        self.a = Policy(*EXPERT_A)
        self.b = Policy(*EXPERT_B)

    def __call__(self, env: Composed):
        return (self.a if env.phase == 0 else self.b)(env.obs_for_phase())


class Cheap:
    def __init__(self, path):
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.mu, self.sd = blob["mu"], blob["sd"]
        self.net = C.build_mlp(len(self.mu), 3, blob["hidden"])
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self.torch = torch

    def __call__(self, env: Composed):
        x = (env.flat() - self.mu) / self.sd
        with self.torch.no_grad():
            return self.net(self.torch.from_numpy(x[None]))[0].numpy()

    def params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


def run_episode(env: Composed, seed, cheap, expert, rule, kw=None, rng=None, record=False):
    """rule: cheap | expert | handover(kw t) | arb(kw q_esc, hold) | random(kw rate, hold)."""
    env.reset(seed)
    kw = kw or {}
    arbiter = None
    if rule in ("arb", "random"):
        arbiter = C.OnlineArbiter({"closed": kw.get("q_esc", {}).get("reach", np.inf),
                                   "open": kw.get("q_esc", {}).get(PHASES[1], np.inf)}, 1,
                                  kw.get("hold", C.arb.DEFAULT_HOLD))
    mode = EXPENSIVE if rule == "expert" or (rule == "handover" and kw["t"] == 0) else CHEAP
    modes, scores, masks, labels, flags, acts = [], [], [], [], [], []
    boundary, event, t_switch = None, "continue", None
    t_inf = {CHEAP: 0.0, EXPENSIVE: 0.0}
    traj = {"qpos": [], "obj_pos": [], "obj_quat": [], "ee": []} if record else None

    def snap():                      # replay data for kinematic animation (e.g. in MuJoCo)
        sim, rob = env.base.sim, env.base.robot
        traj["qpos"].append([rob.get_joint_angle(j) for j in rob.joint_indices])
        traj["obj_pos"].append(sim.get_base_position("object"))
        traj["obj_quat"].append(sim.get_base_orientation("object"))
        traj["ee"].append(rob.get_ee_position())
    if record:
        snap()
    while True:
        ph = env.phase
        ee0 = env.slide_obs["observation"][:3].copy()
        t0 = time.perf_counter()
        a = (cheap if mode == CHEAP else expert)(env)
        t_inf[mode] += time.perf_counter() - t0
        event, _ = env.step(a)
        if record:
            snap()
        s, _, ok = C.step_score(np.concatenate([a, [0.0]]), ee0, env.slide_obs["observation"][:3])
        modes.append(mode); scores.append(s); masks.append(ok); labels.append(ph); acts.append(a)
        fl = False
        if rule == "arb":
            fl = arbiter.flag(s, ph, ok)          # ph index 0 = reach, 1 = slide
            mode = arbiter.step_flag(fl, ok)
        elif rule == "random":
            fl = bool(ok and rng.random() < kw["rate"])
            mode = arbiter.step_flag(fl, ok)
        elif rule == "handover":
            mode = EXPENSIVE if len(modes) >= kw["t"] else CHEAP
        flags.append(fl)
        if event == "phase_switch":
            boundary, t_switch = env.boundary, env.t
        if event in ("success", "fail"):
            break
    modes = np.array(modes, np.int8)
    rec = {"seed": seed, "rule": rule, "success": event == "success", "T": len(modes),
           "phaseA_success": t_switch is not None, "t_switch": t_switch,
           "fail_phase": None if event == "success" else ("reach" if t_switch is None else PHASES[1]),
           "boundary": boundary, "modes": modes, "score": np.array(scores), "mask": np.array(masks, bool),
           "labels": np.array(labels, np.int8), "flags": np.array(flags, bool), "actions": np.array(acts),
           "frac_expensive": float(modes.mean()), "switches": int(np.count_nonzero(modes[1:] > modes[:-1])),
           "infer_s": t_inf}
    if record:
        rec["traj"] = {k: np.asarray(v, np.float32) for k, v in traj.items()}
        rec["traj"]["reach_goal"] = env.reach_goal.astype(np.float32)
        rec["traj"]["task_goal"] = env.slide_goal.astype(np.float32)
    if arbiter is not None:
        st = C.arb.hysteresis_state(rec["flags"], rec["mask"], 1, arbiter.hold)
        assert np.array_equal(st[:-1].astype(np.int8), modes[1:]), "arbiter mismatch"
    return rec


