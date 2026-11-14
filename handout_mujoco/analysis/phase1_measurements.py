"""Reproduce every measured number quoted in MUJOCO_TRAINING_SPEC.md sections 7, 11, 12.

Phase 1 made three design decisions from measurement rather than argument:
  - reject resets with ||cube - task_goal|| < 0.05          (section 7)
  - reach_goal is the Option-A pre-push waypoint             (section 11)
  - the push hand-off offset is fitted in the APPROACH FRAME (section 12)

Each is defended by a number, and this is the script those numbers come from. It
trains nothing and writes no policy artifact.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python analysis/phase1_measurements.py
"""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))

import envs
from envs import (
    CUBE_XY,
    CUBE_Z,
    MIN_SEPARATION,
    SUCCESS_RADIUS,
    WAYPOINT_OFFSET,
)
from mj_env import MJLive

OUT = ROOT / "results" / "mujoco_train" / "phase1_measurements.json"
SAMPLE = ROOT / "results" / "mujoco_train" / "reach_handoff_sample.npy"


def raw_pairs(rng, n):
    """cube/task_goal WITHOUT the rejection rule — the distribution being judged."""
    cube = np.column_stack(
        [rng.uniform(-CUBE_XY, CUBE_XY, n), rng.uniform(-CUBE_XY, CUBE_XY, n), np.full(n, CUBE_Z)]
    )
    goal = np.column_stack(
        [rng.uniform(-CUBE_XY, CUBE_XY, n), rng.uniform(-CUBE_XY, CUBE_XY, n), np.full(n, CUBE_Z)]
    )
    return cube, goal


def m_separation(n=20000):
    """Section 7: how much of the raw uniform reset is already solved at t=0."""
    cube, goal = raw_pairs(np.random.default_rng(0), n)
    d = np.linalg.norm(cube - goal, axis=1)
    return {
        "n": n,
        "min": float(d.min()),
        "mean": float(d.mean()),
        "max": float(d.max()),
        "frac_born_successful": float(np.mean(d < SUCCESS_RADIUS)),
        "frac_below_0.01": float(np.mean(d < 0.01)),
    }


def m_waypoint(sim, n=400):
    """Sections 7 + 11: where Option-A waypoints land, and whether the arm can get there."""
    rng = np.random.default_rng(0)
    cube, goal = zip(*[envs.sample_task(rng) for _ in range(n)])
    cube, goal = np.array(cube), np.array(goal)
    wp = np.array([envs.waypoint(c, g) for c, g in zip(cube, goal)])
    res = np.array([envs.place_ee(sim, w) for w in wp])
    outside = (np.abs(wp[:, :2]) > CUBE_XY).any(axis=1)
    return {
        "n": n,
        "frac_outside_sample_box": float(outside.mean()),
        "max_abs_x": float(np.abs(wp[:, 0]).max()),
        "max_abs_y": float(np.abs(wp[:, 1]).max()),
        "ik_residual_median": float(np.median(res)),
        "ik_residual_max": float(res.max()),
        "frac_unreachable": float(np.mean(res > SUCCESS_RADIUS)),
    }


def m_reach_handoff(sim, n=1000):
    """Sections 11 + 12: what a P-servo to the waypoint achieves, and what it hands over.

    One pass produces three things: the reach-difficulty stats, the cube DISTURBANCE the
    reach phase inflicts, and the EE OFFSET at hand-off. The last two are saved as a
    joint (n, 6) sample rather than fitted, because they are strongly anti-correlated
    (the gripper pushes the cube ahead of itself, closing the gap): sampling them
    independently would invent hand-off poses the reach phase never produces.

    A P-servo is not ReachExpert. It bounds the difficulty of reach and gives the first
    empirical estimate of the hand-off. Phase 4 re-measures against the trained expert.
    """
    rng = np.random.default_rng(0)
    hit, steps, dz, frame = [], [], [], []
    for _ in range(n):
        cube0, goal = envs.sample_task(rng)
        wp = envs.waypoint(cube0, goal)
        sim.reset(arm=envs.NEUTRAL_ARM, fingers=envs.NEUTRAL_FINGERS, cube=cube0,
                  cube_quat_xyzw=[0, 0, 0, 1], reach_goal=wp, task_goal=goal)
        ok, t = False, envs.REACH_LIMIT
        for i in range(envs.REACH_LIMIT):
            sim.step(np.clip((wp - sim.ee()) / 0.05, -1, 1))
            if np.linalg.norm(sim.ee() - wp) < SUCCESS_RADIUS:
                ok, t = True, i + 1
                break
        hit.append(ok)
        steps.append(t)
        u = (cube0 - goal) / np.linalg.norm(cube0 - goal)
        nrm = np.array([-u[1], u[0], 0.0])
        d = sim.cube() - cube0            # disturbance the reach phase inflicted
        o = sim.ee() - sim.cube()         # hand-off offset, against the MOVED cube
        dz.append(float(o[2]))
        frame.append([d @ u, d @ nrm, d[2], o @ u, o @ nrm, o[2]])

    hit, steps, dz = map(np.array, (hit, steps, dz))
    a = np.array(frame)
    np.save(SAMPLE, a)

    mag = np.linalg.norm(a[:, :3], axis=1)
    moved = mag > 1e-3

    def stats(col):
        q = np.percentile(col, [5, 95])
        return {"mean": float(col.mean()), "sd": float(col.std()),
                "p5": float(q[0]), "p95": float(q[1])}

    names = ("along", "lateral", "vertical")
    return {
        "n": n,
        "reach_success": float(hit.mean()),
        "steps_median": float(np.median(steps[hit])),
        "steps_p95": float(np.percentile(steps[hit], 95)),
        "steps_max": int(steps[hit].max()),
        "arrival_dz_median": float(np.median(dz)),
        "arrival_dz_max": float(dz.max()),
        "frac_arrival_dz_above_0.04": float(np.mean(dz > 0.04)),
        "disturbance": {
            "frac_moved_above_1mm": float(moved.mean()),
            "frac_moved_above_1cm": float(np.mean(mag > 0.01)),
            "magnitude_max": float(mag.max()),
            "magnitude_median_when_moved": float(np.median(mag[moved])),
            **{nm: stats(a[:, i]) for i, nm in enumerate(names)},
            "moved_only": {nm: stats(a[moved, i]) for i, nm in enumerate(names)},
        },
        "handoff_offset_approach_frame": {nm: stats(a[:, 3 + i]) for i, nm in enumerate(names)},
        "correlation_disturbance_vs_offset": {
            "along": float(np.corrcoef(a[:, 0], a[:, 3])[0, 1]),
            "lateral": float(np.corrcoef(a[:, 1], a[:, 4])[0, 1]),
            "vertical": float(np.corrcoef(a[:, 2], a[:, 5])[0, 1]),
        },
        "offset_along_given_undisturbed": stats(a[~moved, 3]),
        "offset_along_given_disturbed": stats(a[moved, 3]),
        "sample_file": str(SAMPLE.relative_to(ROOT)),
    }


def m_naive_placement(sim, n=300):
    """The contrast that justifies the posture ladder: one IK solve from NEUTRAL, nothing else.

    Same cube, same disturbance, same offset as the shipped reset -- only the recovery
    ladder is removed. Whatever this row shows is what the ladder has to fix.
    """
    import mujoco

    rng = np.random.default_rng(1)
    pen, jump, hand = [], [], []
    for _ in range(n):
        cube0, goal = envs.sample_task(rng)
        d, off = envs.sample_handoff(rng, cube0, goal)
        cube = cube0 + d
        cube[2] = CUBE_Z
        if np.linalg.norm(cube - goal) < MIN_SEPARATION:
            cube = cube0.copy()
        sim.reset(arm=envs.NEUTRAL_ARM, fingers=envs.NEUTRAL_FINGERS, cube=cube,
                  cube_quat_xyzw=[0, 0, 0, 1], reach_goal=envs.waypoint(cube0, goal),
                  task_goal=goal)
        envs.place_ee(sim, cube + off)
        sim.d.qvel[:] = 0.0
        sim.d.ctrl[:7] = sim.arm_q()
        mujoco.mj_forward(sim.m, sim.d)
        pen.append(envs.cube_robot_penetration(sim))
        hand.append(float(np.linalg.norm(sim.ee() - sim.cube())))
        c0 = sim.cube().copy()
        sim.step(np.zeros(3))
        jump.append(float(np.linalg.norm(sim.cube() - c0)))
    pen, jump, hand = map(np.array, (pen, jump, hand))
    return {
        "n": n,
        "deepest_penetration": float(pen.min()),
        "frac_penetration_below_1mm": float(np.mean(pen < -1e-3)),
        "frac_penetration_below_tolerance": float(np.mean(pen < envs.MAX_RESET_PENETRATION)),
        "cube_jump_max": float(jump.max()),
        "frac_cube_jump_above_5mm": float(np.mean(jump > 5e-3)),
        "handoff_distance_median": float(np.median(hand)),
    }


def m_env_reset(n=500):
    """The live PushOnly.reset itself: does the shipped path reproduce the hand-off?

    Three things have to hold together, and the first two fight each other: the cube must
    be disturbed at the rate reach disturbs it, no start may be physically invalid, and
    the EE offset must still look like the measured hand-off.
    """
    env = envs.PushOnly()
    sep, hand, retreat, placements, fallbacks, disturbed = [], [], [], [], [], []
    for i in range(n):
        env.reset(seed=i)
        c = env.sim.cube().copy()
        sep.append(float(np.linalg.norm(c - env.sim.task_goal)))
        hand.append(float(np.linalg.norm(env.sim.ee() - c)))
        retreat.append(env.reset_retreat)
        placements.append(env.reset_placements)
        fallbacks.append(env.reset_fallback)
        # the waypoint stays anchored to the UNDISTURBED cube, so the departure of
        # ||cube - reach_goal|| from WAYPOINT_OFFSET is exactly the applied disturbance
        disturbed.append(
            abs(float(np.linalg.norm(c - env.sim.reach_goal)) - WAYPOINT_OFFSET)
        )
    sep, hand, retreat, placements, fallbacks, disturbed = (
        np.array(x) for x in (sep, hand, retreat, placements, fallbacks, disturbed)
    )
    r = retreat > 0
    return {
        "n": n,
        "min_cube_goal_separation": float(sep.min()),
        "frac_born_successful": float(np.mean(sep < SUCCESS_RADIUS)),
        "frac_cube_disturbed_above_1mm": float(np.mean(disturbed > 1e-3)),
        "cube_disturbance_max": float(disturbed.max()),
        "handoff_distance_median": float(np.median(hand)),
        "handoff_distance_max": float(hand.max()),
        "placements_mean": float(placements.mean()),
        "placements_max": int(placements.max()),
        "frac_needing_retreat": float(r.mean()),
        "retreat_mean_when_used": float(retreat[r].mean()) if r.any() else 0.0,
        "retreat_max": float(retreat.max()),
        "handoff_distance_median_no_retreat": float(np.median(hand[~r])),
        "handoff_distance_median_retreated": float(np.median(hand[r])) if r.any() else 0.0,
        "frac_separation_fallback": float(np.mean(fallbacks)),
    }


def main():
    sim = MJLive()
    out = {
        "separation_raw_uniform": m_separation(),
        "waypoint": m_waypoint(sim),
        "p_servo": m_reach_handoff(sim),
        "push_reset_naive_placement": m_naive_placement(sim),
        "live_env_reset": m_env_reset(),
        "constants": {
            "MIN_SEPARATION": MIN_SEPARATION, "SUCCESS_RADIUS": SUCCESS_RADIUS,
            "WAYPOINT_OFFSET": WAYPOINT_OFFSET, "CUBE_Z": CUBE_Z,
            "MAX_RESET_PENETRATION": envs.MAX_RESET_PENETRATION,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwritten: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
