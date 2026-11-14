"""Gymnasium Reach->Push environment running the live MuJoCo physics.

Wraps `MJLive` rather than re-implementing a simulator, so training and the eventual
live demonstration execute the identical scene, controller and timestep. There is
exactly one simulator in this project.

Specification: live/MUJOCO_TRAINING_SPEC.md (approved 2026-09-14).
    observation  26-dim  Box(-inf, inf)
    action        3-dim  Box(-1, 1), EE displacement a * 0.05 m per 0.04 s step
    episode      REACH (50 steps) -> PUSH (50 steps), 100 max

Nothing from the PyBullet system is imported: no checkpoint, no threshold, no success
number. The reset distribution and success radii come from the task definition so the
two systems solve the same task; they are reported side by side, never pooled.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("gymnasium is required: uv pip install gymnasium") from e

import mujoco

from mj_env import ACTION_SCALE, MJLive

# --- task definition, measured from the recorded episodes (see the spec, section 7)
CUBE_XY = 0.15  # cube and task_goal sampled U(-0.15, 0.15)
REACH_XY = 0.15
REACH_Z = (0.0, 0.30)
CUBE_Z = 0.02  # cube resting on the table (edge 0.04)
NEUTRAL_ARM = np.array([0.0, 0.41, 0.0, -1.85, 0.0, 2.26, 0.79])
NEUTRAL_FINGERS = np.array([0.0, 0.0])

SUCCESS_RADIUS = 0.05
REACH_LIMIT = 50
PUSH_LIMIT = 50
MAX_STEPS = 100
CUBE_FELL_Z = -0.05

# conformal score: commands smaller than this are unscorable (the ratio's denominator
# vanishes). 10% of a full-scale command; revisited when calibrating in Phase 6.
MOTION_FLOOR = 0.1 * ACTION_SCALE

REACH, PUSH = 0, 1


def quat_to_euler(w, x, y, z):
    """(w,x,y,z) -> intrinsic xyz Euler angles."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


class MuJoCoReachPush(gym.Env):
    """Reach a point, then push a cube to a goal. Both phases in one episode."""

    metadata = {"render_modes": []}

    def __init__(self, seed_offset=0):
        super().__init__()
        self.sim = MJLive()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.seed_offset = seed_offset

        m = self.sim.m
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._cube_dof = m.jnt_dofadr[jid]
        self._jacp = np.zeros((3, m.nv))

        self.phase = REACH
        self.steps = 0
        self.phase_steps = 0

    # ------------------------------------------------------------------ observation
    def _obs(self):
        s, m, d = self.sim, self.sim.m, self.sim.d
        mujoco.mj_jacSite(m, d, self._jacp, None, s.tcp)
        ee_vel = self._jacp @ d.qvel
        cube_pos = np.array(d.xpos[s.cube_bid])
        w, x, y, z = d.xquat[s.cube_bid]
        cube_rot = quat_to_euler(w, x, y, z)
        v = d.qvel[self._cube_dof : self._cube_dof + 6]
        phase = np.array([1.0, 0.0]) if self.phase == REACH else np.array([0.0, 1.0])
        return np.concatenate(
            [
                s.ee(),
                ee_vel,
                cube_pos,
                cube_rot,
                v[:3],
                v[3:6],
                s.reach_goal,
                s.task_goal,
                phase,
            ]
        ).astype(np.float32)

    # ----------------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            seed = int(seed) + self.seed_offset
        super().reset(seed=seed)
        r = self.np_random
        cube = np.array(
            [r.uniform(-CUBE_XY, CUBE_XY), r.uniform(-CUBE_XY, CUBE_XY), CUBE_Z]
        )
        task_goal = np.array(
            [r.uniform(-CUBE_XY, CUBE_XY), r.uniform(-CUBE_XY, CUBE_XY), CUBE_Z]
        )
        reach_goal = np.array(
            [
                r.uniform(-REACH_XY, REACH_XY),
                r.uniform(-REACH_XY, REACH_XY),
                r.uniform(*REACH_Z),
            ]
        )
        self.sim.reset(
            arm=NEUTRAL_ARM,
            fingers=NEUTRAL_FINGERS,
            cube=cube,
            cube_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            reach_goal=reach_goal,
            task_goal=task_goal,
        )
        self.phase = REACH
        self.steps = 0
        self.phase_steps = 0
        return self._obs(), self._info(reward_terms={})

    # ------------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ee_before = self.sim.ee()
        self.sim.step(a)
        ee_after = self.sim.ee()

        self.steps += 1
        self.phase_steps += 1

        d_reach = float(np.linalg.norm(ee_after - self.sim.reach_goal))
        cube = self.sim.cube()
        d_push = float(np.linalg.norm(cube - self.sim.task_goal))
        d_hand = float(np.linalg.norm(ee_after - cube))

        terminated = truncated = False
        success = False
        bonus = 0.0

        if self.phase == REACH:
            reward = -d_reach
            if d_reach < SUCCESS_RADIUS:  # transition, not terminal
                self.phase = PUSH
                self.phase_steps = 0
                bonus = 10.0
            elif self.phase_steps >= REACH_LIMIT:
                terminated = True  # reach failed -> episode over
        else:
            reward = -d_push - 0.2 * d_hand
            if d_push < SUCCESS_RADIUS:
                terminated = True
                success = True
                bonus = 50.0
            elif self.phase_steps >= PUSH_LIMIT:
                truncated = True

        reward += bonus - 0.01 * float(np.dot(a, a))

        if cube[2] < CUBE_FELL_Z:
            reward -= 10.0
            terminated = True

        if self.steps >= MAX_STEPS and not terminated:
            truncated = True

        # conformal score, recorded every step so calibration needs no extra rollouts
        cmd = a * ACTION_SCALE
        cmd_norm = float(np.linalg.norm(cmd))
        scorable = cmd_norm >= MOTION_FLOOR
        score = (
            float(np.linalg.norm(cmd - (ee_after - ee_before))) / cmd_norm
            if scorable
            else float("nan")
        )

        info = self._info(
            reward_terms={
                "d_reach": d_reach,
                "d_push": d_push,
                "d_hand": d_hand,
                "bonus": bonus,
            },
            score=score,
            scorable=scorable,
            success=success,
        )
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------------ info
    def _info(self, reward_terms, score=float("nan"), scorable=False, success=False):
        return {
            "phase": int(self.phase),
            "phase_name": "reach" if self.phase == REACH else "push",
            "steps": int(self.steps),
            "phase_steps": int(self.phase_steps),
            "reach_done": bool(self.phase == PUSH),
            "is_success": bool(success),
            "score": float(score),
            "scorable": bool(scorable),
            "ee": self.sim.ee().copy(),
            "cube": self.sim.cube().copy(),
            "reach_goal": self.sim.reach_goal.copy(),
            "task_goal": self.sim.task_goal.copy(),
            **reward_terms,
        }

    def close(self):
        pass
