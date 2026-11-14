"""Live MuJoCo Reach->Push environment. Physics is simulated; nothing is replayed.

The only thing taken from the recorded episodes is the INITIAL condition of a seed
(cube pose, goals, starting joint angles) so a live run starts where the PyBullet
run started. Every state after t=0 comes from MuJoCo integrating the action.

Action convention, copied from data/meta.json and the panda-gym Push task:

    a in [-1, 1]^3  ->  desired end-effector displacement of a * 0.05 m
    one control step = 0.04 s (25 Hz) = 20 MuJoCo substeps at dt = 0.002

panda-gym applies that displacement through its own Cartesian controller. MuJoCo's
Panda exposes joint-position actuators, so the displacement is converted with a
damped-least-squares Jacobian step on the `tcp` site (hand + 0.105 m, the frame the
handout verified against the recorded end-effector at 1.3 mm median).

**The observation adapter is deliberately NOT implemented.** Building it requires
the cheap-MLP / TQC checkpoints to read the exact observation layout and
normalisation off, and those are not present in this repo. Guessing it would
produce a system that silently is not the benchmarked one. `observation()` raises
until a checkpoint is supplied; everything else here runs.
"""

import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SCENE = os.path.join(
    ROOT, "third_party", "mujoco_menagerie", "franka_emika_panda", "scene_live.xml"
)
NPZ = os.path.join(ROOT, "data", "replay_trajectories_a0.30.npz")

ARM = [f"joint{i}" for i in range(1, 8)]
FINGERS = ["finger_joint1", "finger_joint2"]
CTRL_DT = 0.04  # 25 Hz, meta.json
ACTION_SCALE = 0.05  # metres per unit action, meta.json
SUCCESS_RADIUS = 0.05  # meta.json: phase-B success when ||cube - goal|| < 0.05
REACH_RADIUS = 0.05
DLS_LAMBDA = 0.05

# Workspace floor for the commanded EE target. Without it the tuned controller's force
# headroom (x10) lets the arm press through the table and EXTRUDE the cube downward --
# measured: cube driven from z=+0.020 to z=-0.067 in 8 steps, in the middle of the
# table. In the reference system that never happens: of 1683 recorded frames with the
# cube below the table top, 100% are outside the table footprint (fell off an edge),
# 0% are penetration. Clamping the target, not the controller, keeps the tuned
# response intact. Table top is z = 0.
WORKSPACE_Z_MIN = 0.0

# Controller, selected by sweep against the recorded PyBullet response. panda.xml's
# default kv/kp = 0.10 OVER-DAMPS the servo: t90 = 109 substeps against a 20-substep
# control period, so only 0.30 of a commanded displacement lands. Damping is the fix;
# stiffness is not (kp x1 is best). Force headroom removes the saturation of
# actuators 2/4/6. Gives ratio 0.645 against PyBullet's own 0.655.
# See results/controller_tuning.md.
CTRL_KP_MULT, CTRL_KV_RATIO, CTRL_FORCE_MULT = 1.0, 0.035, 10.0


class MJLive:
    def __init__(self, scene=SCENE, seed_source=NPZ):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        self.z = np.load(seed_source) if os.path.exists(seed_source) else None
        self.substeps = max(1, int(round(CTRL_DT / self.m.opt.timestep)))

        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_qadr = np.array([self.m.jnt_qposadr[jid(n)] for n in ARM])
        self.arm_dofadr = np.array([self.m.jnt_dofadr[jid(n)] for n in ARM])
        self.fing_qadr = np.array([self.m.jnt_qposadr[jid(n)] for n in FINGERS])
        self.cube_qadr = self.m.jnt_qposadr[jid("cube_free")]
        self.tcp = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.cube_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        mocap = lambda n: self.m.body_mocapid[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        ]
        self.mc_push, self.mc_reach = mocap("push_goal"), mocap("reach_goal")
        self.mc_lamp = mocap("mode_lamp")
        self.task_goal = np.zeros(3)
        self.reach_goal = np.zeros(3)
        self.apply_controller()

    def apply_controller(self, kp_mult=CTRL_KP_MULT, kv_ratio=CTRL_KV_RATIO,
                         force_mult=CTRL_FORCE_MULT):
        """Retune the 7 arm position servos. Gripper actuator left alone."""
        if not hasattr(self, "_ctrl_base"):
            self._ctrl_base = (self.m.actuator_gainprm[:7, 0].copy(),
                               self.m.actuator_forcerange[:7].copy())
        kp0, fr0 = self._ctrl_base
        for i in range(7):
            kp = kp0[i] * kp_mult
            self.m.actuator_gainprm[i, 0] = kp
            self.m.actuator_biasprm[i, 1] = -kp
            self.m.actuator_biasprm[i, 2] = -kp * kv_ratio
            self.m.actuator_forcerange[i] = fr0[i] * force_mult

    # ---------------------------------------------------------------- state
    def ee(self):
        return np.array(self.d.site_xpos[self.tcp])

    def cube(self):
        return np.array(self.d.xpos[self.cube_bid])

    def arm_q(self):
        return np.array(self.d.qpos[self.arm_qadr])

    def reached(self):
        return bool(np.linalg.norm(self.ee() - self.reach_goal) < REACH_RADIUS)

    def success(self):
        return bool(np.linalg.norm(self.cube() - self.task_goal) < SUCCESS_RADIUS)

    # ---------------------------------------------------------------- reset
    def reset_from_seed(self, seed, condition="cheap"):
        """Start where the recorded PyBullet episode started. Initial condition only."""
        if self.z is None:
            raise FileNotFoundError(NPZ)
        g = lambda f: np.asarray(self.z[f"{condition}/seed{seed}/{f}"])
        qpos0, obj0 = g("qpos")[0], g("obj_pos")[0]
        quat0 = g("obj_quat")[0]  # PyBullet (x,y,z,w)
        return self.reset(
            arm=qpos0[:7],
            fingers=qpos0[7:9],
            cube=obj0,
            cube_quat_xyzw=quat0,
            reach_goal=g("reach_goal"),
            task_goal=g("task_goal"),
        )

    def reset(self, arm, fingers, cube, cube_quat_xyzw, reach_goal, task_goal):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_qadr] = arm
        self.d.qpos[self.fing_qadr] = fingers
        self.d.qpos[self.cube_qadr : self.cube_qadr + 3] = cube
        x, y, zc, w = cube_quat_xyzw  # MuJoCo free joints want (w,x,y,z)
        self.d.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = [w, x, y, zc]
        self.d.qvel[:] = 0.0
        self.task_goal = np.asarray(task_goal, float)
        self.reach_goal = np.asarray(reach_goal, float)
        self.d.mocap_pos[self.mc_push] = self.task_goal
        self.d.mocap_pos[self.mc_reach] = self.reach_goal
        self.d.ctrl[:7] = arm
        self.d.ctrl[7] = 0.0  # gripper blocked, as in panda-gym
        mujoco.mj_forward(self.m, self.d)
        return self.state()

    # ----------------------------------------------------------------- step
    def action_to_ctrl(self, action):
        """a in [-1,1]^3 -> joint-position targets, via damped least squares on tcp.

        This is the explicit policy_action -> mujoco_control_command adapter. It is
        the one place the Cartesian convention becomes joint commands; if the
        convention is ever wrong, it is wrong here and nowhere else.
        """
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        target = self.ee() + a * ACTION_SCALE
        target[2] = max(target[2], WORKSPACE_Z_MIN)
        dx = target - self.ee()
        jacp = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, None, self.tcp)
        J = jacp[:, self.arm_dofadr]  # 3 x 7
        JJt = J @ J.T + (DLS_LAMBDA**2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, dx)
        return self.arm_q() + dq

    def step(self, action, on_substep=None):
        """One 0.04 s control step = `substeps` physics substeps.

        `on_substep` is called after each substep. It exists so a viewer can render
        DURING the control step: rendering only between control steps teleports 0.04 s
        of motion at a time, which reads as a slideshow rather than a robot moving.
        """
        self.d.ctrl[:7] = self.action_to_ctrl(action)
        self.d.ctrl[7] = 0.0
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
            if on_substep is not None:
                on_substep()
        return self.state()

    def state(self):
        return {
            "ee": self.ee(),
            "cube": self.cube(),
            "arm_q": self.arm_q(),
            "reached": self.reached(),
            "success": self.success(),
            "t": float(self.d.time),
        }

    # ---------------------------------------------------------- observation
    def observation(self):
        raise NotImplementedError(
            "The observation adapter cannot be written without the policy checkpoints.\n"
            "The cheap MLP and TQC actors are not in this repo (the handout ships data +\n"
            "replay only; its requirements.txt lists no torch/SB3/panda-gym). Building\n"
            "the observation by guessing its layout, ordering or normalisation would\n"
            "produce a system that is silently not the benchmarked one.\n"
            "Supply the checkpoints, read the exact layout off them, and implement here."
        )


def load_meta():
    return json.load(open(os.path.join(ROOT, "data", "meta.json")))
