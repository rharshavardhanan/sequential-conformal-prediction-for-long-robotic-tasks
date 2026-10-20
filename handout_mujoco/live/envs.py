"""Two single-phase Gymnasium envs on the live MuJoCo physics: ReachOnly, PushOnly.

Both wrap the EXISTING MJLive, so training and the demo share one simulator, one tuned
controller and one 25 Hz control period. The composed MuJoCoReachPush stays on disk as
the record of the 4%-success single-agent attempt; only quat_to_euler is imported.

Design: docs/superpowers/specs/2026-09-14-mujoco-split-envs-design.md
"""

import pathlib

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("gymnasium is required: uv pip install gymnasium") from e

from mj_env import ACTION_SCALE, MJLive
from mujoco_reach_push_env import quat_to_euler

import mujoco

# --- task geometry -------------------------------------------------------------
CUBE_XY = 0.15
CUBE_Z = 0.02
SUCCESS_RADIUS = 0.05
# 8.0% of raw U(+-0.15) pairs land inside the success radius, i.e. the episode is won
# at t=0. Rejecting keeps every reported success rate free of that floor and keeps
# unit(cube - task_goal) well conditioned (raw min separation was 0.0087 m).
MIN_SEPARATION = 0.05
WAYPOINT_OFFSET = 0.08
NEUTRAL_ARM = np.array([0.0, 0.41, 0.0, -1.85, 0.0, 2.26, 0.79])
NEUTRAL_FINGERS = np.array([0.0, 0.0])
REACH_LIMIT = 50
PUSH_LIMIT = 50
CUBE_FELL_Z = -0.05

# --- push reset ----------------------------------------------------------------
# What the reach phase hands over is measured, not fitted: one row per reach episode,
# [d_along, d_lateral, d_z, o_along, o_lateral, o_z], in the approach frame.
#   d = the cube DISTURBANCE reach inflicted     o = the EE OFFSET at hand-off
# The pair is resampled TOGETHER. Measured corr(d_along, o_along) = -0.79: the gripper
# pushes the cube ahead of itself and closes the gap, so offset_along is +0.067 when the
# cube was untouched and +0.036 when it was shoved. Drawing the two independently would
# invent hand-off poses the reach phase never produces.
# Regenerate with: PYTHONPATH=live ../.venv-lerobot/bin/python analysis/phase1_measurements.py
HANDOFF_SAMPLE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "results" / "mujoco_train" / "reach_handoff_sample.npy"
)
_handoff = None
# Poses the P-servo reaches THROUGH PHYSICS carry cube<->gripper overlap up to 6 mm, so
# they are valid by construction. A kinematic placement can bury the closed gripper 15 mm
# into the cube, which ejects it on step 1. Resample rather than derive a clearance for a
# non-convex gripper.
MAX_RESET_PENETRATION = -0.002
# Resampling the (disturbance, offset) PAIR on a contact failure is not neutral: the
# penetrating draws are exactly the disturbed ones (a shoved cube sits closer to the
# gripper, offset_along +0.032 vs +0.068 undisturbed), so rejection preferentially throws
# disturbed cases away -- measured, it cut the disturbance rate from 31.7% to 13.0% and
# quietly undid the reason the disturbance is modelled at all. Retreat along the approach
# ray instead: it preserves the drawn disturbance exactly and perturbs only the offset,
# which is the quantity being reconstructed. The root cause is that only EE *position* is
# reconstructed, not gripper orientation, so a pose physics reached cleanly can penetrate
# when re-placed by IK from NEUTRAL.
# The residual failures are not the fingertips at all: they are geom70, the `hand` body
# (a 0.032 x 0.049 x 0.104 m mesh), clipping the cube from above while the tcp is still
# 8-11 cm clear. Moving the target cannot fix that. Re-seeding the IK can -- the arm is
# redundant, and 3 of 8 jittered start postures reach the IDENTICAL EE target with zero
# penetration. So the ladder tries postures first, which preserves the drawn disturbance
# AND offset exactly, and only falls back to retreating the target if none of them clear.
RESET_Q_JITTER = 0.35
RESET_POSTURE_TRIES = 8
RESET_RETREAT_STEP = 0.004
MAX_RESET_PLACEMENTS = 16

MOTION_FLOOR = 0.1 * ACTION_SCALE


def sample_task(rng):
    """cube and task_goal on the table, at least MIN_SEPARATION apart."""
    while True:
        cube = np.array(
            [rng.uniform(-CUBE_XY, CUBE_XY), rng.uniform(-CUBE_XY, CUBE_XY), CUBE_Z]
        )
        goal = np.array(
            [rng.uniform(-CUBE_XY, CUBE_XY), rng.uniform(-CUBE_XY, CUBE_XY), CUBE_Z]
        )
        if np.linalg.norm(cube - goal) >= MIN_SEPARATION:
            return cube, goal


def waypoint(cube, task_goal):
    """Option A: the pre-push waypoint, WAYPOINT_OFFSET behind the cube at cube height."""
    v = np.asarray(cube, float) - np.asarray(task_goal, float)
    u = v / np.linalg.norm(v)  # sample_task guarantees ||v|| >= MIN_SEPARATION
    wp = np.asarray(cube, float) + WAYPOINT_OFFSET * u
    wp[2] = CUBE_Z
    return wp


def place_ee(sim, target, q0=None, iters=100, lam=0.05):
    """Move the arm so the tcp sits at `target`. Kinematic only: qpos + mj_forward.

    Damped least squares on the tcp site, joint-limit clipped. No mj_step, so the cube
    is never disturbed by the placement. `q0` seeds the solve; the arm is redundant, so a
    different seed reaches the same target through a different posture. Returns the
    residual in metres.
    """
    sim.d.qpos[sim.arm_qadr] = NEUTRAL_ARM if q0 is None else q0
    mujoco.mj_forward(sim.m, sim.d)
    jacp = np.zeros((3, sim.m.nv))
    target = np.asarray(target, float)
    for _ in range(iters):
        err = target - sim.ee()
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(sim.m, sim.d, jacp, None, sim.tcp)
        J = jacp[:, sim.arm_dofadr]
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
        sim.d.qpos[sim.arm_qadr] = np.clip(
            sim.d.qpos[sim.arm_qadr] + dq,
            sim.m.jnt_range[:7, 0],
            sim.m.jnt_range[:7, 1],
        )
        mujoco.mj_forward(sim.m, sim.d)
    return float(np.linalg.norm(target - sim.ee()))


def cube_robot_penetration(sim):
    """Most-negative cube<->robot contact distance at the current state (0.0 if none).

    The cube<->table contact is excluded: a cube resting on a table is a contact, not a
    reset defect.
    """
    cube_gid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    worst = 0.0
    for i in range(sim.d.ncon):
        c = sim.d.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if cube_gid not in pair or pair == {cube_gid}:
            continue
        other = (pair - {cube_gid}).pop()
        if mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_GEOM, other) == "table":
            continue
        worst = min(worst, float(c.dist))
    return worst


def handoff_sample():
    """The measured (disturbance, offset) rows. Loaded once, cached."""
    global _handoff
    if _handoff is None:
        if not HANDOFF_SAMPLE.exists():
            raise FileNotFoundError(
                f"{HANDOFF_SAMPLE} is missing. Regenerate it with:\n"
                "  PYTHONPATH=live ../.venv-lerobot/bin/python "
                "analysis/phase1_measurements.py"
            )
        _handoff = np.load(HANDOFF_SAMPLE)
    return _handoff


def sample_handoff(rng, cube, task_goal):
    """Draw one (cube_disturbance, ee_offset) pair, rebuilt in this episode's frame.

    The rows where reach left the cube alone carry the zero-displacement mass, so no
    separate mixture is needed. The vertical component of the DISTURBANCE is dropped:
    a training reset must start from a settled cube, and the measured z excursion is
    transient bounce (sd ~3 mm). The offset keeps its z.
    """
    a = handoff_sample()
    row = a[rng.integers(len(a))]
    v = np.asarray(cube, float) - np.asarray(task_goal, float)
    u = v / np.linalg.norm(v)
    n = np.array([-u[1], u[0], 0.0])
    disturbance = row[0] * u + row[1] * n
    offset = row[3] * u + row[4] * n + np.array([0.0, 0.0, row[5]])
    return disturbance, offset

class _Base(gym.Env):
    """Shared plumbing: one MJLive, the 3-dim action space, the conformal score."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.sim = MJLive()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        m = self.sim.m
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._cube_dof = m.jnt_dofadr[jid]
        self._jacp = np.zeros((3, m.nv))
        self.steps = 0

    def _core(self):
        """The 18 dims both envs share: ee, ee_vel, cube, cube_rot, cube_vel."""
        s, m, d = self.sim, self.sim.m, self.sim.d
        mujoco.mj_jacSite(m, d, self._jacp, None, s.tcp)
        v = d.qvel[self._cube_dof : self._cube_dof + 6]
        w, x, y, z = d.xquat[s.cube_bid]
        return np.concatenate(
            [
                s.ee(),
                self._jacp @ d.qvel,
                np.array(d.xpos[s.cube_bid]),
                quat_to_euler(w, x, y, z),
                v[:3],
                v[3:6],
            ]
        )

    def _score(self, action, ee_before, ee_after):
        cmd = np.asarray(action, float) * ACTION_SCALE
        cmd_norm = float(np.linalg.norm(cmd))
        if cmd_norm < MOTION_FLOOR:
            return float("nan"), False
        return float(np.linalg.norm(cmd - (ee_after - ee_before))) / cmd_norm, True

    def close(self):
        pass


class ReachOnly(_Base):
    """Servo the end-effector to the pre-push waypoint. Free-space, no contact."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(24,), dtype=np.float32
        )

    def _obs(self):
        return np.concatenate(
            [self._core(), self.sim.reach_goal, self.sim.task_goal]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cube, task_goal = sample_task(self.np_random)
        self.sim.reset(
            arm=NEUTRAL_ARM,
            fingers=NEUTRAL_FINGERS,
            cube=cube,
            cube_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            reach_goal=waypoint(cube, task_goal),
            task_goal=task_goal,
        )
        self.steps = 0
        return self._obs(), self._info(
            float(np.linalg.norm(self.sim.ee() - self.sim.reach_goal)), False
        )

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ee_before = self.sim.ee().copy()
        self.sim.step(a)
        self.steps += 1

        d_reach = float(np.linalg.norm(self.sim.ee() - self.sim.reach_goal))
        reward = -d_reach - 0.01 * float(np.dot(a, a))
        success = bool(d_reach < SUCCESS_RADIUS)
        # Budget expiry is a TIME LIMIT, not an MDP-terminal state: with terminated=True
        # SB3 zeroes the bootstrap and the critic learns that running out of time has
        # value 0, which is not what the state means. Reach is easy enough to survive
        # the error; the same mistake in PushOnly would be a real bug.
        terminated = success
        truncated = (not success) and self.steps >= REACH_LIMIT
        score, scorable = self._score(a, ee_before, self.sim.ee())
        return (
            self._obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(d_reach, success, score, scorable),
        )

    def _info(self, d_reach, success, score=float("nan"), scorable=False):
        return {
            "phase": 0,
            "phase_name": "reach",
            "steps": int(self.steps),
            "d_reach": float(d_reach),
            "is_success": bool(success),
            "score": float(score),
            "scorable": bool(scorable),
            "ee": self.sim.ee().copy(),
            "cube": self.sim.cube().copy(),
            "reach_goal": self.sim.reach_goal.copy(),
            "task_goal": self.sim.task_goal.copy(),
        }


class PushOnly(_Base):
    """Push the cube to the goal, starting where the reach phase hands over.

    Sparse goal-conditioned reward so HerReplayBuffer can relabel: a dense reward makes
    relabelled transitions incomparable across goals.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf, np.inf, shape=(18,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
            }
        )
        self.reset_retreat = 0.0
        self.reset_placements = 0
        self.reset_fallback = False

    def _obs(self):
        return {
            "observation": self._core().astype(np.float32),
            "achieved_goal": self.sim.cube().astype(np.float32),
            "desired_goal": self.sim.task_goal.astype(np.float32),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Sparse: 0.0 inside the success radius, -1.0 outside. Vectorized for HER."""
        d = np.linalg.norm(
            np.asarray(achieved_goal, float) - np.asarray(desired_goal, float), axis=-1
        )
        return (d < SUCCESS_RADIUS).astype(np.float64) - 1.0

    def reset(self, *, seed=None, options=None):
        """Start where reach leaves it: cube nudged as reach nudges it, arm at the cube.

        `cube0` is where the cube was sampled; `cube` is where reach would have left it.
        The waypoint stays anchored to `cube0`, because that is the target reach was
        given before it disturbed anything.
        """
        super().reset(seed=seed)
        cube0, task_goal = sample_task(self.np_random)
        reach_goal = waypoint(cube0, task_goal)
        disturbance, offset = sample_handoff(self.np_random, cube0, task_goal)
        cube = cube0 + disturbance
        cube[2] = CUBE_Z
        self.reset_fallback = np.linalg.norm(cube - task_goal) < MIN_SEPARATION
        if self.reset_fallback:
            # the nudge alone solved the task; keep the undisturbed cube so the episode
            # is still a task instance
            cube = cube0.copy()

        self.sim.reset(
            arm=NEUTRAL_ARM,
            fingers=NEUTRAL_FINGERS,
            cube=cube,
            cube_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            reach_goal=reach_goal,
            task_goal=task_goal,
        )
        n_off = np.linalg.norm(offset)
        back = (
            offset / n_off
            if n_off > 1e-9
            else (cube0 - task_goal) / np.linalg.norm(cube0 - task_goal)
        )
        retreat = 0.0
        for k in range(MAX_RESET_PLACEMENTS):
            q0 = (
                NEUTRAL_ARM
                if k == 0
                else NEUTRAL_ARM + self.np_random.normal(0.0, RESET_Q_JITTER, 7)
            )
            retreat = RESET_RETREAT_STEP * max(0, k - RESET_POSTURE_TRIES)
            place_ee(self.sim, cube + offset + retreat * back, q0)
            self.sim.d.qvel[:] = 0.0
            self.sim.d.ctrl[:7] = self.sim.arm_q()
            mujoco.mj_forward(self.sim.m, self.sim.d)
            if cube_robot_penetration(self.sim) >= MAX_RESET_PENETRATION:
                break
        self.reset_placements = k + 1
        self.reset_retreat = retreat
        self.steps = 0
        return self._obs(), self._info(False)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ee_before = self.sim.ee().copy()
        self.sim.step(a)
        self.steps += 1

        cube = self.sim.cube()
        success = bool(np.linalg.norm(cube - self.sim.task_goal) < SUCCESS_RADIUS)
        reward = float(self.compute_reward(cube, self.sim.task_goal, {}))
        fell = bool(cube[2] < CUBE_FELL_Z)
        terminated = success or fell
        truncated = (not terminated) and self.steps >= PUSH_LIMIT
        score, scorable = self._score(a, ee_before, self.sim.ee())
        return (
            self._obs(),
            reward,
            terminated,
            truncated,
            self._info(success, score, scorable, fell),
        )

    def _info(self, success, score=float("nan"), scorable=False, fell=False):
        cube = self.sim.cube()
        return {
            "phase": 1,
            "phase_name": "push",
            "steps": int(self.steps),
            "is_success": bool(success),
            "cube_fell": bool(fell),
            "d_push": float(np.linalg.norm(cube - self.sim.task_goal)),
            "d_hand": float(np.linalg.norm(self.sim.ee() - cube)),
            "score": float(score),
            "scorable": bool(scorable),
            "reset_retreat": float(self.reset_retreat),
            "reset_placements": int(self.reset_placements),
            "reset_fallback": bool(self.reset_fallback),
            "ee": self.sim.ee().copy(),
            "cube": cube.copy(),
            "reach_goal": self.sim.reach_goal.copy(),
            "task_goal": self.sim.task_goal.copy(),
        }
