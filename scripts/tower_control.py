"""Layer 2: joint targets -> robosuite, and a scripted expert that works in that
same space so demonstrations and policies speak one language.

THE CONTROLLER FACT THAT BITES
------------------------------
robosuite's JointPositionController ALWAYS does
    goal_qpos = current_qpos + scale_action(action)
`control_delta` is never consulted (verified by reading `set_goal`). So an
absolute joint target cannot be handed to it directly -- it must be expressed as
a delta and rate-limited:

    action = clip((target - current) / RATE, -1, 1)

with the controller configured so +/-1 maps to +/-RATE radians per control step.
That is a real joint-speed limit, which is what a physical position controller
has anyway. RATE = 0.2 rad/step at 20 Hz = 4 rad/s.
"""

import numpy as np
from robosuite.controllers import load_controller_config

RATE = 0.2          # radians per control step that +/-1.0 maps to
LAMBDA = 0.08       # damped-least-squares regulariser


def controller_config(rate=RATE, kp=200):
    cfg = load_controller_config(default_controller="JOINT_POSITION")
    cfg.update(input_max=1, input_min=-1, output_max=rate, output_min=-rate, kp=kp)
    return cfg


def joint_action(env, target_q, grip, rate=RATE):
    """Absolute joint target (radians) -> the delta the controller consumes."""
    q = np.asarray(env.robots[0]._joint_positions)
    return np.concatenate([np.clip((np.asarray(target_q) - q) / rate, -1, 1),
                           [float(grip)]])


def eef_jacobian(env):
    """Position Jacobian of the end-effector w.r.t. the arm joints."""
    sim = env.sim
    site = env.robots[0].controller.eef_name
    jacp = sim.data.get_site_jacp(site).reshape(3, -1)
    return jacp[:, env.robots[0]._ref_joint_vel_indexes]


def ik_step(env, target_pos, max_step=0.35):
    """One damped-least-squares step: desired eef position -> joint target.

    Damped rather than a plain pseudo-inverse because the Panda passes through
    singular configurations while reaching across the table, where an undamped
    inverse produces enormous joint velocities.
    """
    q = np.asarray(env.robots[0]._joint_positions)
    cur = np.array(env.sim.data.site_xpos[
        env.sim.model.site_name2id(env.robots[0].controller.eef_name)])
    dx = np.asarray(target_pos) - cur
    n = np.linalg.norm(dx)
    if n > 0.05:                      # cap the per-step Cartesian request
        dx = dx / n * 0.05
    J = eef_jacobian(env)
    dq = J.T @ np.linalg.solve(J @ J.T + LAMBDA ** 2 * np.eye(3), dx)
    return q + np.clip(dq, -max_step, max_step)


def eef_pos(env):
    return np.array(env.sim.data.site_xpos[
        env.sim.model.site_name2id(env.robots[0].controller.eef_name)])


class TowerExpert:
    """Scripted stacker. Uses cube positions, which is exactly the information
    Route B gives the learned policies -- so what it demonstrates is learnable."""

    OPEN, CLOSED = -1.0, 1.0

    def __init__(self, env, clearance=0.10, place_gap=0.048):
        self.env, self.clearance, self.place_gap = env, clearance, place_gap
        self.reset()

    def reset(self):
        self.stage, self.idx, self.timer = "select", 0, 0
        self.base_xy = None
        self.order = None

    def _cubes(self):
        return [np.array(self.env.sim.data.body_xpos[b])
                for b in self.env.cube_body_ids]

    def act(self):
        """Return (joint_target, grip). Drives a pick-and-place state machine."""
        env, cubes = self.env, self._cubes()
        e = eef_pos(env)
        self.timer += 1

        if self.order is None:
            # stack onto the cube nearest the robot; carry the others to it
            order = sorted(range(len(cubes)), key=lambda i: np.linalg.norm(cubes[i][:2]))
            self.base_xy = cubes[order[0]][:2].copy()
            self.order = order[1:]
            self.idx = 0

        if self.idx >= len(self.order):                       # done
            return ik_step(env, e + np.array([0, 0, 0.02])), self.OPEN

        c = cubes[self.order[self.idx]]
        layer = self.idx + 1
        place = np.array([self.base_xy[0], self.base_xy[1],
                          env.table_offset[2] + 0.022 + layer * self.place_gap])

        if self.stage == "select":
            self.stage, self.timer = "above", 0
        if self.stage == "above":                             # get over the cube
            tgt = np.array([c[0], c[1], c[2] + self.clearance])
            if np.linalg.norm(e - tgt) < 0.015 or self.timer > 90:
                self.stage, self.timer = "descend", 0
            return ik_step(env, tgt), self.OPEN
        if self.stage == "descend":
            tgt = np.array([c[0], c[1], c[2] + 0.002])
            if np.linalg.norm(e - tgt) < 0.012 or self.timer > 70:
                self.stage, self.timer = "grasp", 0
            return ik_step(env, tgt), self.OPEN
        if self.stage == "grasp":
            if self.timer > 12:
                self.stage, self.timer = "lift", 0
            return ik_step(env, e), self.CLOSED
        if self.stage == "lift":
            tgt = np.array([c[0], c[1], env.table_offset[2] + 0.28])
            if e[2] > env.table_offset[2] + 0.25 or self.timer > 70:
                self.stage, self.timer = "carry", 0
            return ik_step(env, tgt), self.CLOSED
        if self.stage == "carry":
            tgt = np.array([place[0], place[1], env.table_offset[2] + 0.28])
            if np.linalg.norm(e[:2] - tgt[:2]) < 0.015 or self.timer > 110:
                self.stage, self.timer = "place", 0
            return ik_step(env, tgt), self.CLOSED
        if self.stage == "place":
            if np.linalg.norm(e - place) < 0.012 or self.timer > 80:
                self.stage, self.timer = "release", 0
            return ik_step(env, place), self.CLOSED
        if self.stage == "release":
            if self.timer > 10:
                self.stage, self.timer, self.idx = "retreat", 0, self.idx + 1
            return ik_step(env, e), self.OPEN
        if self.stage == "retreat":
            tgt = e + np.array([0, 0, 0.12])
            if self.timer > 25:
                self.stage, self.timer = "select", 0
            return ik_step(env, tgt), self.OPEN
        return ik_step(env, e), self.OPEN
