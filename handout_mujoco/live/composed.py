"""Composed Reach->Push: one episode, two phases, one 26-dim observation.

The reach stage is a scripted P-SERVO, not a learned expert. ReachExpert (Gate 3) does
not exist. This is not a degraded substitute: Phase 1 measured the P-servo at 100%
success in a median of 4 steps, and `reach_handoff_sample.npy` -- the distribution
PushExpert was trained against -- was fitted to this exact servo. So the hand-off here is
MATCHED by construction, which is the one thing a learned reach expert would have put at
risk until the Phase 4 refit ran.

Label it a P-servo reach everywhere. It is not a learned reach expert.

Observation is the 26-dim flat vector from MUJOCO_TRAINING_SPEC section 3:
    core 18 (ee, ee_vel, cube, cube_rot, cube_lin_vel, cube_ang_vel)
    + reach_goal 3 + task_goal 3 + phase one-hot 2
Both single-phase envs are slices of it, which is why the cheap policy can run one
network across both phases.
"""

import numpy as np
from envs import (
    CUBE_FELL_Z,
    MOTION_FLOOR,
    NEUTRAL_ARM,
    NEUTRAL_FINGERS,
    PUSH_LIMIT,
    REACH_LIMIT,
    SUCCESS_RADIUS,
    quat_to_euler,
    sample_task,
    waypoint,
)
from mj_env import ACTION_SCALE, MJLive

import mujoco

REACH, PUSH = 0, 1
MAX_STEPS = REACH_LIMIT + PUSH_LIMIT


class ComposedEnv:
    """Live MuJoCo Reach->Push. Phase latches once and never returns to REACH."""

    def __init__(self, sim=None):
        self.sim = sim or MJLive()
        m = self.sim.m
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._cube_dof = m.jnt_dofadr[jid]
        self._jacp = np.zeros((3, m.nv))
        self.phase = REACH
        self.steps = self.phase_steps = 0

    # ------------------------------------------------------------------ observations
    def _core(self):
        s, m, d = self.sim, self.sim.m, self.sim.d
        mujoco.mj_jacSite(m, d, self._jacp, None, s.tcp)
        v = d.qvel[self._cube_dof : self._cube_dof + 6]
        w, x, y, z = d.xquat[s.cube_bid]
        return np.concatenate(
            [s.ee(), self._jacp @ d.qvel, np.array(d.xpos[s.cube_bid]),
             quat_to_euler(w, x, y, z), v[:3], v[3:6]]
        )

    def obs26(self):
        """What the CHEAP policy sees, in both phases."""
        ph = np.array([1.0, 0.0]) if self.phase == REACH else np.array([0.0, 1.0])
        return np.concatenate(
            [self._core(), self.sim.reach_goal, self.sim.task_goal, ph]
        ).astype(np.float32)

    def push_obs(self):
        """What the PUSH EXPERT sees. Must match PushOnly's Dict space exactly."""
        return {
            "observation": self._core().astype(np.float32),
            "achieved_goal": self.sim.cube().astype(np.float32),
            "desired_goal": self.sim.task_goal.astype(np.float32),
        }

    # ------------------------------------------------------------------------- reset
    def reset(self, seed):
        """Identical to ReachOnly's reset: arm at NEUTRAL, undisturbed cube."""
        rng = np.random.default_rng(seed)
        cube, task_goal = sample_task(rng)
        self.sim.reset(
            arm=NEUTRAL_ARM, fingers=NEUTRAL_FINGERS, cube=cube,
            cube_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            reach_goal=waypoint(cube, task_goal), task_goal=task_goal,
        )
        self.phase = REACH
        self.steps = self.phase_steps = 0
        self.cube0 = cube.copy()
        self.d0 = float(np.linalg.norm(cube - task_goal))  # regime feature, geometry only
        self.reached = False
        return self.obs26()

    # -------------------------------------------------------------------------- step
    def step(self, action, on_substep=None):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ee_before = self.sim.ee().copy()
        phase_at_action = self.phase
        self.sim.step(a, on_substep=on_substep)
        self.steps += 1
        self.phase_steps += 1

        ee, cube = self.sim.ee(), self.sim.cube()
        d_reach = float(np.linalg.norm(ee - self.sim.reach_goal))
        d_push = float(np.linalg.norm(cube - self.sim.task_goal))
        success = fell = False
        terminated = truncated = False

        if self.phase == REACH:
            if d_reach < SUCCESS_RADIUS:
                self.phase, self.phase_steps, self.reached = PUSH, 0, True
            elif self.phase_steps >= REACH_LIMIT:
                truncated = True  # reach ran out of budget; the episode is over
        else:
            if d_push < SUCCESS_RADIUS:
                terminated = success = True
            elif self.phase_steps >= PUSH_LIMIT:
                truncated = True

        if cube[2] < CUBE_FELL_Z:
            terminated, fell = True, True
        if self.steps >= MAX_STEPS and not terminated:
            truncated = True

        cmd = a * ACTION_SCALE
        cn = float(np.linalg.norm(cmd))
        realised = float(np.linalg.norm(ee - ee_before))
        scorable = cn >= MOTION_FLOOR
        score = float(np.linalg.norm(cmd - (ee - ee_before))) / cn if scorable else float("nan")

        info = {
            "phase": int(phase_at_action),           # the phase the ACTION was taken in
            "phase_after": int(self.phase),
            "phase_name": "reach" if phase_at_action == REACH else "push",
            "steps": int(self.steps), "phase_steps": int(self.phase_steps),
            "d_reach": d_reach, "d_push": d_push,
            "d_hand": float(np.linalg.norm(ee - cube)),
            "reached": bool(self.reached), "is_success": bool(success),
            "cube_fell": bool(fell), "score": score, "scorable": bool(scorable),
            # commanded vs realised EE displacement, metres. Shown as the TRACKING bars:
            # the policy asked for this much motion and the arm delivered this much.
            "cmd_norm": cn, "realised": realised,
            "ee": ee.copy(), "cube": cube.copy(),
            "reach_goal": self.sim.reach_goal.copy(),
            "task_goal": self.sim.task_goal.copy(), "d0": self.d0,
        }
        return self.obs26(), bool(terminated), bool(truncated), info


def pservo_action(env):
    """The scripted reach stage. 100% success, median 4 steps (Phase 1, N=1000)."""
    return np.clip((env.sim.reach_goal - env.sim.ee()) / ACTION_SCALE, -1.0, 1.0)


class ComposedExpert:
    """P-servo reach + TQC push expert. NOT a learned reach expert."""

    label = "pservo_reach + tqc_push"

    def __init__(self, push_model):
        self.push = push_model

    def act(self, env):
        if env.phase == REACH:
            return pservo_action(env), "pservo"
        a, _ = self.push.predict(env.push_obs(), deterministic=True)
        return np.asarray(a, dtype=np.float64).ravel(), "tqc_push"


def rollout(env, policy_fn, seed, max_steps=MAX_STEPS, record=None):
    """One episode. `policy_fn(env) -> (action, tag)`. Returns (info, n_steps)."""
    env.reset(seed)
    info = None
    for _ in range(max_steps):
        obs = env.obs26()
        a, tag = policy_fn(env)
        phase = env.phase
        _, term, trunc, info = env.step(a)
        if record is not None:
            record.append((obs, np.asarray(a, dtype=np.float32), phase, tag))
        if term or trunc:
            break
    return info, env.steps
