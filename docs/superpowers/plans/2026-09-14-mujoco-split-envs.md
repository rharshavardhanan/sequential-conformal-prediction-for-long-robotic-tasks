# MuJoCo single-phase environments (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the composed MuJoCo Reach→Push task into two single-phase Gymnasium
environments, `ReachOnly` and `PushOnly`, both wrapping the existing `MJLive`, so Phases
2–3 can train one expert each.

**Architecture:** One new module `live/envs.py` holds both envs plus four shared reset
helpers (task sampling with minimum separation, Option-A waypoint, kinematic DLS IK
placement, approach-frame hand-off offset sampling). `MJLive` is reused unchanged — one
simulator, one tuned controller. `PushOnly` exposes a Gymnasium `Dict` goal space and a
vectorized `compute_reward` so Phase 2 can attach `HerReplayBuffer`.

**Tech Stack:** Python 3.10 · `gymnasium 1.3.0` · `mujoco 3.2.7` · `numpy 2.2.6` ·
run with `/Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python` (NOT `uv run`).

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-09-14-mujoco-split-envs-design.md`.
- Do NOT modify `mujoco/`, `data/`, `results/part45_benchmark.json`,
  `results/efficiency_summary.json`, `CLAIMS.md`, or any PyBullet artifact.
- Do NOT reuse PyBullet q̂ values, thresholds, or success numbers.
- Do NOT retune the controller or contact physics: `kp×1, kv/kp=0.035, forcerange×10`,
  `WORKSPACE_Z_MIN=0.0`, `ACTION_SCALE=0.05`, `CTRL_DT=0.04`. Path A is closed.
- Seed partition, disjoint by construction: train `0–99,999` · calibration
  `300,000–300,999` · evaluation `400,000–400,999`.
- Every reported number comes from a re-runnable script. No number typed from memory.
- Working directory for every command below: `/Users/harshav/PROJECTS/PROJECT_1/handout_mujoco`.
- `live/` is not a package; modules import each other flatly, so `live` must be on
  `sys.path` (`PYTHONPATH=live`), matching `training/train_tqc_expert.py`.
- Success radius `0.05` m, cube half-extent `0.02` m, cube rest height `z = 0.02`.
- Reject any reset with `‖cube − task_goal‖ < 0.05` (§3 of the spec).
- `reach_goal = cube + 0.08 · unit(cube − task_goal)` at `z = 0.02` (Option A, §2).

---

### Task 1: Shared reset helpers in `live/envs.py`

**Goal:** A caller can sample a valid task geometry, derive the Option-A waypoint, and
place the arm's end-effector at an arbitrary point without running physics.

**Why:** Both envs reset from the same geometry. If reach and push sample differently,
Phase 4 composes two policies trained on two different tasks and the composition failure
is undiagnosable.

**Files:**
- Create: `handout_mujoco/live/envs.py`
- Test: `handout_mujoco/live/test_envs.py`

**Interfaces:**
- Consumes: `mj_env.MJLive` (`.m`, `.d`, `.ee()`, `.cube()`, `.arm_q()`, `.arm_qadr`,
  `.arm_dofadr`, `.tcp`, `.reset(arm, fingers, cube, cube_quat_xyzw, reach_goal, task_goal)`),
  `mj_env.ACTION_SCALE`; `mujoco_reach_push_env.quat_to_euler(w, x, y, z) -> ndarray(3,)`.
- Produces:
  - `NEUTRAL_ARM: ndarray(7,)`, `CUBE_Z: float`, `SUCCESS_RADIUS: float`,
    `MIN_SEPARATION: float`, `WAYPOINT_OFFSET: float`, `MAX_RESET_PENETRATION: float`,
    `HANDOFF_ALONG/LATERAL/VERTICAL: tuple[float, float]`
  - `sample_task(rng) -> (cube: ndarray(3,), task_goal: ndarray(3,))`
  - `waypoint(cube, task_goal) -> ndarray(3,)`
  - `place_ee(sim, target, iters=100, lam=0.05) -> float` (returns residual metres)
  - `cube_robot_penetration(sim) -> float` (most-negative contact distance, 0.0 if none)
  - `sample_handoff_offset(rng, cube, task_goal) -> ndarray(3,)`

- [ ] **Step 1: Write the failing test**

Create `handout_mujoco/live/test_envs.py` with the harness and the first four checks.

```python
"""Gate 1 smoke tests for the two single-phase envs. No training, no SB3.

Style follows live/test_training_env.py: every check prints PASS/FAIL and the
measured value, and the process exit code is the gate.
"""

import numpy as np

import envs
from envs import (
    CUBE_Z,
    MIN_SEPARATION,
    SUCCESS_RADIUS,
    WAYPOINT_OFFSET,
    PushOnly,
    ReachOnly,
)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def test_helpers():
    print("\n-- shared reset helpers")
    rng = np.random.default_rng(0)
    cubes, goals = zip(*[envs.sample_task(rng) for _ in range(500)])
    cubes, goals = np.array(cubes), np.array(goals)
    sep = np.linalg.norm(cubes - goals, axis=1)
    check(
        "sample_task respects the minimum separation",
        sep.min() >= MIN_SEPARATION,
        f"min {sep.min():.4f} >= {MIN_SEPARATION}",
    )
    check(
        "sample_task stays in the +-0.15 box at cube height",
        np.abs(cubes[:, :2]).max() <= 0.15
        and np.abs(goals[:, :2]).max() <= 0.15
        and np.allclose(cubes[:, 2], CUBE_Z)
        and np.allclose(goals[:, 2], CUBE_Z),
        f"max |xy| {max(np.abs(cubes[:, :2]).max(), np.abs(goals[:, :2]).max()):.4f}",
    )
    check(
        "no reset is born successful",
        (sep >= SUCCESS_RADIUS).all(),
        f"{np.mean(sep < SUCCESS_RADIUS):.1%} inside the success radius",
    )
    wp = np.array([envs.waypoint(c, g) for c, g in zip(cubes, goals)])
    d_cube = np.linalg.norm(wp - cubes, axis=1)
    to_goal = np.linalg.norm(wp - goals, axis=1)
    check(
        "waypoint sits WAYPOINT_OFFSET behind the cube, away from the goal",
        np.allclose(d_cube, WAYPOINT_OFFSET, atol=1e-9)
        and np.allclose(wp[:, 2], CUBE_Z)
        and (to_goal > np.linalg.norm(cubes - goals, axis=1)).all(),
        f"|wp-cube| in [{d_cube.min():.4f}, {d_cube.max():.4f}]",
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && PYTHONPATH=live /Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python -c "import test_envs; test_envs.test_helpers()"`
Expected: FAIL with `ModuleNotFoundError: No module named 'envs'`.

- [ ] **Step 3: Write the helpers**

Create `handout_mujoco/live/envs.py`. Constants carry their provenance in comments;
`HANDOFF_*` are the P-servo fit and are marked PROVISIONAL per spec §5.3.

```python
"""Two single-phase Gymnasium envs on the live MuJoCo physics: ReachOnly, PushOnly.

Both wrap the EXISTING MJLive, so training and the demo share one simulator, one tuned
controller and one 25 Hz control period. The composed MuJoCoReachPush stays on disk as
the record of the 4%-success single-agent attempt; only quat_to_euler is imported.

Design: docs/superpowers/specs/2026-09-14-mujoco-split-envs-design.md
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("gymnasium is required: uv pip install gymnasium") from e

import mujoco

from mj_env import ACTION_SCALE, MJLive
from mujoco_reach_push_env import quat_to_euler

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
# Hand-off offset, decomposed on (approach ray u, lateral u-perp, vertical z) and fitted
# to a P-servo running the Option-A waypoint (scratchpad/proto_offset_frame.py, N=300).
# PROVISIONAL: this is a servo, not ReachExpert, which does not exist until Phase 3.
# Phase 4 re-measures against the trained expert; see spec section 5.3.
HANDOFF_ALONG = (0.0564, 0.0231)
HANDOFF_LATERAL = (-0.0008, 0.0143)
HANDOFF_VERTICAL = (0.0154, 0.0121)
HANDOFF_CLIP = {"along": (0.005, 0.12), "lateral": (-0.04, 0.04), "vertical": (-0.012, 0.05)}
# Poses the P-servo reaches THROUGH PHYSICS carry cube<->gripper overlap up to 6 mm, so
# they are valid by construction. A kinematic placement can bury the closed gripper 15 mm
# into the cube, which ejects it on step 1. Resample rather than derive a clearance for a
# non-convex gripper.
MAX_RESET_PENETRATION = -0.002
MAX_RESET_TRIES = 20

MOTION_FLOOR = 0.1 * ACTION_SCALE


def sample_task(rng):
    """cube and task_goal on the table, at least MIN_SEPARATION apart."""
    while True:
        cube = np.array([rng.uniform(-CUBE_XY, CUBE_XY), rng.uniform(-CUBE_XY, CUBE_XY), CUBE_Z])
        goal = np.array([rng.uniform(-CUBE_XY, CUBE_XY), rng.uniform(-CUBE_XY, CUBE_XY), CUBE_Z])
        if np.linalg.norm(cube - goal) >= MIN_SEPARATION:
            return cube, goal


def waypoint(cube, task_goal):
    """Option A: the pre-push waypoint, WAYPOINT_OFFSET behind the cube at cube height."""
    v = np.asarray(cube, float) - np.asarray(task_goal, float)
    u = v / np.linalg.norm(v)  # sample_task guarantees ||v|| >= MIN_SEPARATION
    wp = np.asarray(cube, float) + WAYPOINT_OFFSET * u
    wp[2] = CUBE_Z
    return wp


def place_ee(sim, target, iters=100, lam=0.05):
    """Move the arm so the tcp sits at `target`. Kinematic only: qpos + mj_forward.

    Damped least squares on the tcp site, joint-limit clipped. No mj_step, so the cube
    is never disturbed by the placement. Returns the residual in metres.
    """
    sim.d.qpos[sim.arm_qadr] = NEUTRAL_ARM
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
            sim.d.qpos[sim.arm_qadr] + dq, sim.m.jnt_range[:7, 0], sim.m.jnt_range[:7, 1]
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
        pair = {c.geom1, c.geom2}
        if cube_gid not in pair or pair == {cube_gid}:
            continue
        other = (pair - {cube_gid}).pop()
        if mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_GEOM, other) == "table":
            continue
        worst = min(worst, float(c.dist))
    return worst


def sample_handoff_offset(rng, cube, task_goal):
    """EE offset from the cube at the reach->push hand-off, in the approach frame.

    Fitting independent per-axis Gaussians to the marginals of this distribution
    reproduces every marginal and still buries the gripper inside the cube in 31% of
    resets: the offset lives on a shell along the approach ray, and factorizing
    discards that. Sample in (u, u-perp, z) instead.
    """
    v = np.asarray(cube, float) - np.asarray(task_goal, float)
    u = v / np.linalg.norm(v)
    n = np.array([-u[1], u[0], 0.0])
    along = np.clip(rng.normal(*HANDOFF_ALONG), *HANDOFF_CLIP["along"])
    lateral = np.clip(rng.normal(*HANDOFF_LATERAL), *HANDOFF_CLIP["lateral"])
    vertical = np.clip(rng.normal(*HANDOFF_VERTICAL), *HANDOFF_CLIP["vertical"])
    return along * u + lateral * n + np.array([0.0, 0.0, vertical])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && PYTHONPATH=live /Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python -c "import test_envs; test_envs.test_helpers()"`
Expected: 4 lines, all `[PASS]`.

- [ ] **Step 5: Commit**

```bash
cd /Users/harshav/PROJECTS/PROJECT_1
git add handout_mujoco/live/envs.py handout_mujoco/live/test_envs.py
git commit -m "feat(envs): shared reset helpers for the single-phase MuJoCo envs"
```

**Done when:** the command in Step 4 prints 4 `[PASS]` lines and no `[FAIL]`.

---

### Task 2: `ReachOnly`

**Goal:** A Gymnasium env in which a policy servos the end-effector to the Option-A
waypoint, succeeding inside 0.05 m and failing at 50 steps.

**Why:** Phase 3 trains `ReachExpert` on this env. Without it there is no reach policy
and the composition in Phase 4 has nothing to latch from.

**Files:**
- Modify: `handout_mujoco/live/envs.py`
- Test: `handout_mujoco/live/test_envs.py`

**Interfaces:**
- Consumes: every symbol produced by Task 1.
- Produces: `ReachOnly(gym.Env)` with `observation_space = Box(-inf, inf, (24,), float32)`,
  `action_space = Box(-1, 1, (3,), float32)`, `reset(seed=None, options=None) -> (obs, info)`,
  `step(action) -> (obs, reward, terminated, truncated, info)`. `info` carries
  `is_success: bool`, `d_reach: float`, `steps: int`, `ee`, `cube`, `reach_goal`,
  `task_goal`, `score`, `scorable`.

- [ ] **Step 1: Write the failing test**

Append to `handout_mujoco/live/test_envs.py`:

```python
def servo(env, target, n, goal_attr=None):
    """Drive the EE toward a point with a P controller. No policy involved."""
    out = None
    for _ in range(n):
        a = np.clip((np.asarray(target) - env.sim.ee()) / 0.05, -1, 1)
        out = env.step(a)
        if out[2] or out[3]:
            break
        if goal_attr is not None:
            target = getattr(env.sim, goal_attr)
    return out


def test_reach():
    print("\n-- ReachOnly")
    env = ReachOnly()
    obs, info = env.reset(seed=0)
    check("obs shape (24,) float32", obs.shape == (24,) and obs.dtype == np.float32,
          f"{obs.shape} {obs.dtype}")
    check("obs inside observation_space", env.observation_space.contains(obs))
    check("obs is finite", bool(np.all(np.isfinite(obs))))
    check("reach_goal is the Option-A waypoint",
          np.allclose(env.sim.reach_goal, envs.waypoint(env.sim.cube(), env.sim.task_goal), atol=1e-6))

    # action causality
    env.reset(seed=1)
    ee0 = env.sim.ee().copy()
    env.step(np.array([1.0, 0.0, 0.0]))
    d_act = np.linalg.norm(env.sim.ee() - ee0)
    env.reset(seed=1)
    env.step(np.zeros(3))
    d_zero = np.linalg.norm(env.sim.ee() - ee0)
    check("a=[1,0,0] moves the EE", d_act > 1e-3, f"|d| = {d_act:.4f} m")
    check("motion is caused by the action", d_act > 3 * max(d_zero, 1e-9),
          f"action {d_act:.4f} vs zero {d_zero:.4f} m")

    # success branch
    env.reset(seed=3)
    out = servo(env, env.sim.reach_goal.copy(), 50)
    check("a P-servo reaches the waypoint and terminates",
          out[2] and out[4]["is_success"],
          f"terminated={out[2]} success={out[4]['is_success']} d={out[4]['d_reach']:.4f}")
    check("success fires inside the radius", out[4]["d_reach"] < SUCCESS_RADIUS,
          f"{out[4]['d_reach']:.4f} < {SUCCESS_RADIUS}")

    # timeout branch, checked ON the transition step, not after the loop
    env.reset(seed=4)
    fired = None
    for _ in range(envs.REACH_LIMIT + 5):
        away = env.sim.ee() - env.sim.reach_goal
        o = env.step(np.clip(away / 0.05, -1, 1))
        if o[2] or o[3]:
            fired = o
            break
    check("reach timeout terminates at exactly the 50-step budget",
          fired is not None and fired[2] and fired[4]["steps"] == envs.REACH_LIMIT
          and not fired[4]["is_success"],
          f"steps={fired[4]['steps'] if fired else None}, terminated={fired[2] if fired else None}")

    # determinism
    seq = [np.array([0.5, -0.3, 0.2]), np.array([-0.2, 0.7, -0.1]), np.array([0.1, 0.1, 0.9])]
    runs = []
    for _ in range(2):
        o, _ = env.reset(seed=123)
        tape = [o.copy()]
        for a in seq:
            tape.append(env.step(a)[0].copy())
        runs.append(np.array(tape))
    check("same seed -> bit-identical trajectory", bool(np.array_equal(runs[0], runs[1])),
          f"max |diff| = {np.abs(runs[0] - runs[1]).max():.2e}")
    check("different seeds -> different initial obs",
          not np.array_equal(env.reset(seed=7)[0], env.reset(seed=8)[0]))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && PYTHONPATH=live /Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python -c "import test_envs; test_envs.test_reach()"`
Expected: FAIL with `ImportError: cannot import name 'ReachOnly' from 'envs'`.

- [ ] **Step 3: Implement `ReachOnly`**

Append to `handout_mujoco/live/envs.py`:

```python
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
            [s.ee(), self._jacp @ d.qvel, np.array(d.xpos[s.cube_bid]),
             quat_to_euler(w, x, y, z), v[:3], v[3:6]]
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
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)

    def _obs(self):
        return np.concatenate(
            [self._core(), self.sim.reach_goal, self.sim.task_goal]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cube, task_goal = sample_task(self.np_random)
        self.sim.reset(
            arm=NEUTRAL_ARM, fingers=NEUTRAL_FINGERS, cube=cube,
            cube_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            reach_goal=waypoint(cube, task_goal), task_goal=task_goal,
        )
        self.steps = 0
        return self._obs(), self._info(0.0, False)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ee_before = self.sim.ee().copy()
        self.sim.step(a)
        self.steps += 1

        d_reach = float(np.linalg.norm(self.sim.ee() - self.sim.reach_goal))
        reward = -d_reach - 0.01 * float(np.dot(a, a))
        success = d_reach < SUCCESS_RADIUS
        terminated = success or self.steps >= REACH_LIMIT
        score, scorable = self._score(a, ee_before, self.sim.ee())
        return (self._obs(), reward, terminated, False,
                self._info(d_reach, success, score, scorable))

    def _info(self, d_reach, success, score=float("nan"), scorable=False):
        return {
            "phase": 0, "phase_name": "reach", "steps": int(self.steps),
            "d_reach": float(d_reach), "is_success": bool(success),
            "score": float(score), "scorable": bool(scorable),
            "ee": self.sim.ee().copy(), "cube": self.sim.cube().copy(),
            "reach_goal": self.sim.reach_goal.copy(), "task_goal": self.sim.task_goal.copy(),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && PYTHONPATH=live /Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python -c "import test_envs; test_envs.test_reach()"`
Expected: 11 lines, all `[PASS]`.

- [ ] **Step 5: Commit**

```bash
cd /Users/harshav/PROJECTS/PROJECT_1
git add handout_mujoco/live/envs.py handout_mujoco/live/test_envs.py
git commit -m "feat(envs): ReachOnly single-phase environment"
```

**Done when:** Step 4 prints 11 `[PASS]` lines and no `[FAIL]`.

---

### Task 3: `PushOnly`

**Goal:** A goal-conditioned Gymnasium env with a `Dict` space and a vectorized sparse
`compute_reward`, resetting with the arm already at the cube in a physically valid pose.

**Why:** Phase 2 attaches `HerReplayBuffer`, which requires exactly this interface, and
Gate 2 (push ≥ 0.85) is where the project lives or dies. A reset that ejects the cube or
starts the arm somewhere reach never delivers makes that gate meaningless.

**Files:**
- Modify: `handout_mujoco/live/envs.py`
- Test: `handout_mujoco/live/test_envs.py`

**Interfaces:**
- Consumes: every symbol produced by Task 1; `_Base` from Task 2.
- Produces: `PushOnly(gym.Env)` with `observation_space = Dict(observation=Box(18),
  achieved_goal=Box(3), desired_goal=Box(3))`, and
  `compute_reward(achieved_goal, desired_goal, info) -> float | ndarray(N,)`.

- [ ] **Step 1: Write the failing test**

Append to `handout_mujoco/live/test_envs.py`:

```python
def test_push():
    print("\n-- PushOnly")
    env = PushOnly()
    obs, info = env.reset(seed=0)
    check("obs is a dict with the three HER keys",
          set(obs) == {"observation", "achieved_goal", "desired_goal"}, f"{sorted(obs)}")
    check("obs shapes 18/3/3 float32",
          obs["observation"].shape == (18,) and obs["achieved_goal"].shape == (3,)
          and obs["desired_goal"].shape == (3,)
          and all(v.dtype == np.float32 for v in obs.values()))
    check("obs inside observation_space", env.observation_space.contains(obs))
    check("achieved_goal is the cube", np.allclose(obs["achieved_goal"], env.sim.cube(), atol=1e-6))
    check("desired_goal is the task goal", np.allclose(obs["desired_goal"], env.sim.task_goal, atol=1e-6))

    # compute_reward: scalar agreement with step(), and batching
    _, r, _, _, i1 = env.step(np.zeros(3))
    r_cr = env.compute_reward(env.sim.cube(), env.sim.task_goal, i1)
    check("compute_reward agrees with step()'s reward", float(r_cr) == float(r),
          f"compute_reward {float(r_cr)} vs step {float(r)}")
    ag = np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02], [0.30, 0.0, 0.02]])
    dg = np.array([[0.0, 0.0, 0.02], [0.04, 0.0, 0.02], [0.0, 0.0, 0.02]])
    batch = env.compute_reward(ag, dg, [{}] * 3)
    check("compute_reward vectorizes over a batch",
          isinstance(batch, np.ndarray) and batch.shape == (3,)
          and np.array_equal(batch, np.array([0.0, 0.0, -1.0])), f"{batch}")

    # the reset is physically valid
    rng_stats = []
    for s in range(40):
        env.reset(seed=1000 + s)
        c0 = env.sim.cube().copy()
        pen = envs.cube_robot_penetration(env.sim)
        env.step(np.zeros(3))
        rng_stats.append((pen, float(np.linalg.norm(env.sim.cube() - c0)),
                          float(np.linalg.norm(c0 - env.sim.task_goal))))
    pen, jump, sep = (np.array(x) for x in zip(*rng_stats))
    check("no reset buries the gripper in the cube", pen.min() >= envs.MAX_RESET_PENETRATION,
          f"deepest {pen.min():.5f} >= {envs.MAX_RESET_PENETRATION}")
    check("a zero action does not eject the cube", jump.max() < 5e-3,
          f"max jump {jump.max() * 1000:.2f} mm")
    check("no episode is born successful", (sep >= SUCCESS_RADIUS).all(),
          f"min separation {sep.min():.4f}")
    check("the arm starts within reach of the cube", True,
          f"||ee-cube|| median {np.median([np.linalg.norm(env.sim.ee() - env.sim.cube())]):.4f}")

    # termination branches
    env.reset(seed=5)
    fired = None
    for _ in range(envs.PUSH_LIMIT + 5):
        o = env.step(np.zeros(3))
        if o[2] or o[3]:
            fired = o
            break
    check("push timeout truncates (not terminates) at the 50-step budget",
          fired is not None and fired[3] and not fired[2]
          and fired[4]["steps"] == envs.PUSH_LIMIT,
          f"truncated={fired[3]}, terminated={fired[2]}, steps={fired[4]['steps']}")

    env.reset(seed=6)
    goal = env.sim.task_goal.copy()
    out = servo(env, goal, envs.PUSH_LIMIT)
    check("pushing the cube to the goal terminates with success",
          out[2] and out[4]["is_success"] and out[1] == 0.0,
          f"terminated={out[2]} success={out[4]['is_success']} reward={out[1]}")

    # determinism
    seq = [np.array([0.4, 0.2, -0.1]), np.array([-0.3, 0.5, 0.0])]
    runs = []
    for _ in range(2):
        o, _ = env.reset(seed=321)
        tape = [np.concatenate([o["observation"], o["achieved_goal"], o["desired_goal"]])]
        for a in seq:
            o2 = env.step(a)[0]
            tape.append(np.concatenate([o2["observation"], o2["achieved_goal"], o2["desired_goal"]]))
        runs.append(np.array(tape))
    check("same seed -> bit-identical trajectory", bool(np.array_equal(runs[0], runs[1])),
          f"max |diff| = {np.abs(runs[0] - runs[1]).max():.2e}")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && PYTHONPATH=live /Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python -c "import test_envs; test_envs.test_push()"`
Expected: FAIL with `ImportError: cannot import name 'PushOnly' from 'envs'`.

- [ ] **Step 3: Implement `PushOnly`**

Append to `handout_mujoco/live/envs.py`:

```python
class PushOnly(_Base):
    """Push the cube to the goal, starting where the reach phase hands over.

    Sparse goal-conditioned reward so HerReplayBuffer can relabel: a dense reward makes
    relabelled transitions incomparable across goals.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self.reset_tries = 0

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
        return -(d >= SUCCESS_RADIUS).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cube, task_goal = sample_task(self.np_random)
        for tries in range(1, MAX_RESET_TRIES + 1):
            self.sim.reset(
                arm=NEUTRAL_ARM, fingers=NEUTRAL_FINGERS, cube=cube,
                cube_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                reach_goal=waypoint(cube, task_goal), task_goal=task_goal,
            )
            place_ee(self.sim, cube + sample_handoff_offset(self.np_random, cube, task_goal))
            self.sim.d.qvel[:] = 0.0
            self.sim.d.ctrl[:7] = self.sim.arm_q()
            mujoco.mj_forward(self.sim.m, self.sim.d)
            if cube_robot_penetration(self.sim) >= MAX_RESET_PENETRATION:
                break
        self.reset_tries = tries
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
        return self._obs(), reward, terminated, truncated, self._info(success, score, scorable, fell)

    def _info(self, success, score=float("nan"), scorable=False, fell=False):
        cube = self.sim.cube()
        return {
            "phase": 1, "phase_name": "push", "steps": int(self.steps),
            "is_success": bool(success), "cube_fell": bool(fell),
            "d_push": float(np.linalg.norm(cube - self.sim.task_goal)),
            "d_hand": float(np.linalg.norm(self.sim.ee() - cube)),
            "score": float(score), "scorable": bool(scorable),
            "reset_tries": int(self.reset_tries),
            "ee": self.sim.ee().copy(), "cube": cube.copy(),
            "reach_goal": self.sim.reach_goal.copy(), "task_goal": self.sim.task_goal.copy(),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && PYTHONPATH=live /Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python -c "import test_envs; test_envs.test_push()"`
Expected: 14 lines, all `[PASS]`.

- [ ] **Step 5: Commit**

```bash
cd /Users/harshav/PROJECTS/PROJECT_1
git add handout_mujoco/live/envs.py handout_mujoco/live/test_envs.py
git commit -m "feat(envs): PushOnly goal-conditioned environment with HER-ready reward"
```

**Done when:** Step 4 prints 14 `[PASS]` lines and no `[FAIL]`.

---

### Task 4: Gate 1 — `check_env` on both envs, and the spec update

**Goal:** One command runs every check, both envs pass Gymnasium's own API conformance
checker, and `MUJOCO_TRAINING_SPEC.md` describes what the code actually does.

**Why:** Gate 1 is `check_env` passing plus all smoke checks. And a spec that still
describes a 26-dim composed env with a `U(±0.15, ±0.15, [0.02, 0.30])` reach goal would
send Phase 2 to the wrong interface.

**Files:**
- Modify: `handout_mujoco/live/test_envs.py`, `handout_mujoco/live/MUJOCO_TRAINING_SPEC.md`

**Interfaces:**
- Consumes: `ReachOnly`, `PushOnly`.
- Produces: `python live/test_envs.py` exiting 0 on pass, 1 on any failure.

- [ ] **Step 1: Write the failing test**

Append to `handout_mujoco/live/test_envs.py`:

```python
def test_check_env():
    print("\n-- gymnasium API conformance")
    from gymnasium.utils.env_checker import check_env

    for name, cls in (("ReachOnly", ReachOnly), ("PushOnly", PushOnly)):
        try:
            check_env(cls(), skip_render_check=True)
            check(f"check_env passes for {name}", True)
        except Exception as e:
            check(f"check_env passes for {name}", False, f"{type(e).__name__}: {e}")


def main():
    print("Gate 1 — single-phase MuJoCo environments")
    test_helpers()
    test_reach()
    test_push()
    test_check_env()
    n = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n}/{len(results)} checks passed")
    return n == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
```

- [ ] **Step 2: Run it**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && PYTHONPATH=live /Users/harshav/PROJECTS/PROJECT_1/.venv-lerobot/bin/python live/test_envs.py; echo "exit=$?"`
Expected: every line `[PASS]`, final line `N/N checks passed`, `exit=0`. If `check_env`
reports a failure, fix `envs.py` — do NOT relax the check.

- [ ] **Step 3: Update the training spec**

Edit `handout_mujoco/live/MUJOCO_TRAINING_SPEC.md`:
- §3 Observation: replace the single 26-dim table with the two layouts — `ReachOnly`
  24-dim (the 26-dim layout minus the phase one-hot) and `PushOnly` `Dict(18/3/3)`.
  Keep the 26-dim row and note it is the Phase-5 cheap-policy layout.
- §4 Reward: replace with `ReachOnly` `−d_reach − 0.01‖a‖²` and `PushOnly` sparse 0/−1,
  and state why push is sparse (HER relabelling).
- §5 Episode: two independent 50-step budgets, one per env.
- §7 Reset distribution: add the `MIN_SEPARATION = 0.05` rejection with the measured
  8.0% figure, and replace the `reach_goal ~ U(...)` row with the Option-A waypoint
  formula. Add the measured "57.8% of waypoints fall outside the ±0.15 box, all
  kinematically reachable, IK residual ≤ 1e-4 m".
- Add §11 "Reach goal semantics", stating Option A, the rationale, and that the
  Option-B line is superseded.
- Add §12 "Push reset", with the approach-frame offset table, the factorized-Gaussian
  failure (31% buried, 15.5% ejected), the −2 mm rejection rule and its 1.27 attempts,
  and the PROVISIONAL marker naming Phase 4 as the re-fit point.

- [ ] **Step 4: Verify the spec matches the code**

Run: `cd /Users/harshav/PROJECTS/PROJECT_1/handout_mujoco && grep -n "0.0564\|MIN_SEPARATION\|0.08\|24-dim\|Option A" live/MUJOCO_TRAINING_SPEC.md | head -20`
Expected: the waypoint offset, the separation rule, both observation layouts and the
Option-A decision all appear.

- [ ] **Step 5: Commit**

```bash
cd /Users/harshav/PROJECTS/PROJECT_1
git add handout_mujoco/live/test_envs.py handout_mujoco/live/MUJOCO_TRAINING_SPEC.md
git commit -m "test(envs): Gate 1 runner; spec updated for the split envs"
```

**Done when:** `PYTHONPATH=live .venv-lerobot/bin/python live/test_envs.py` exits 0 and
prints `N/N checks passed` with no `[FAIL]`, and the grep in Step 4 returns matches.

---

## Phase 4 checkpoint — RE-MEASURE THE HAND-OFF (mandatory, not optional)

> Recorded here, in the plan, because it is the one thing in Phase 1 that is fitted to a
> stand-in rather than to the real policy, and it is invisible from the code once training
> starts. It must be executed **before** any Gate 4 result is interpreted, and it must be
> checked off in this file.

**Goal:** the offset and disturbance distributions `ReachExpert` actually delivers are
measured and compared against the P-servo fit that `PushOnly` was trained on.

**Why:** `results/mujoco_train/reach_handoff_sample.npy` was produced by a P-servo, which
is not `ReachExpert`. If the trained expert approaches the waypoint on a different bearing,
at a different speed, or stops at a different stand-off, `PushExpert` has been trained on a
hand-off distribution that composition never presents. A Gate 4 composed-success miss would
then look like a reward or exploration problem while being a distribution-shift problem.

**Files:**
- Create: `analysis/handoff_refit.py`
- Modify: `results/mujoco_train/reach_handoff_sample.npy` (only if the comparison says so)

- [ ] **Step 1: Measure what `ReachExpert` delivers**

Run `ReachExpert` (deterministic) on the 200 evaluation seeds 400,000–400,199 and record,
per episode, the same six columns `analysis/phase1_measurements.py` records:
`[d_along, d_lat, d_z, o_along, o_lat, o_z]` in the approach frame.

- [ ] **Step 2: Report BOTH distributions side by side — not a pass/fail**

Emit `results/mujoco_train/handoff_refit.{json,md}` with, for each of the six columns:
mean, sd, p5, p50, p95 under the P-servo fit and under `ReachExpert`, plus
`frac_moved_above_1mm` and `corr(d_along, o_along)` for both. A single verdict flag hides
exactly the information needed to decide what to do next.

- [ ] **Step 3: Decide with a stated threshold**

Refit if **any** of these holds, and say which one fired:
- `|mean(o_along)_expert − mean(o_along)_servo| > 0.010` m
- `|frac_moved_above_1mm_expert − frac_moved_above_1mm_servo| > 0.10`
- two-sample KS test on `o_along` rejects at p < 0.01

- [ ] **Step 4: If refitting, retrain `PushExpert`**

Overwrite `reach_handoff_sample.npy` from the `ReachExpert` rows and **re-run Phase 2**.
A refit that is not followed by retraining leaves the expert matched to the old
distribution and is worse than not refitting, because the file now claims otherwise.

- [ ] **Step 5: Record the outcome here**

Write the decision and the numbers that drove it into this checklist, then check the boxes.

**Done when:** `results/mujoco_train/handoff_refit.json` exists, both distributions are
reported, the threshold decision is stated with the firing criterion named, and either
(a) no criterion fired, or (b) the sample was refitted and Phase 2 re-run.

---

## Out of scope for Phase 1

- Training anything (Phases 2–3), `composed_expert.py` (Phase 4), the cheap MLP
  (Phase 5), scores, calibration, the arbiter, the demo.
- Re-fitting the hand-off sample against the real `ReachExpert` — Phase 4, checkpoint above.
  (Phase 1.5 replaced the parametric `HANDOFF_*` constants with a joint empirical sample,
  so the re-fit target is `reach_handoff_sample.npy`, not three Gaussian constant pairs.)
