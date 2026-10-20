"""Which cube the expert will pick up next, from environment geometry alone.

This is the supervision the Route-B observation was missing. `TowerExpert` knows
its own `idx`; the network never did, so two states that looked identical carried
opposite action labels -- measured at 44% target disagreement among nearest
neighbours, with 37% of the dataset's action variance surviving among states the
network cannot tell apart. A least-squares fit answers that with the mean of the
branches, which lands the gripper between the cubes: ~8 cm predicted, 12.5-24.7 cm
measured. See docs/closed-loop-diagnostic.md.

**This function must never read the expert's internal state.** It is a pure
function of cube positions, so the identical call works at collection time and at
runtime. Collection and closed-loop evaluation both reach it through
`scripts.tower_collect.observation`, so there is exactly one implementation.
"""

import numpy as np

STACKED = 0.045  # metres above the table that mean "already on the stack"


def target_cube(cubes, table_z):
    """(index of the cube to move next, index of the base cube).

    Mirrors TowerExpert's own rule: the cube nearest the robot base becomes the
    stack, the rest are carried to it in that order, and a cube already sitting
    above STACKED is done. Returns (None, base) once every cube is placed.
    """
    cubes = np.asarray(cubes)
    order = sorted(range(len(cubes)), key=lambda i: float(np.linalg.norm(cubes[i][:2])))
    base, rest = order[0], order[1:]
    for i in rest:
        if cubes[i][2] - table_z < STACKED:
            return i, base
    return None, base


def target_channels(cubes, table_z, eef, n_cubes=None):
    """The 9 channels appended to the observation: absolute (3), eef-relative (3),
    one-hot id (3).

    With no target left (every cube placed) the absolute slot holds the eef itself,
    so the relative vector is zero and nothing sits outside the workspace; the
    one-hot is all zeros, which is a distinct 'nothing to do' signal.
    """
    cubes = np.asarray(cubes)
    n = n_cubes or len(cubes)
    i, _ = target_cube(cubes, table_z)
    onehot = np.zeros(n, dtype=np.float64)
    if i is None:
        pos = np.asarray(eef, dtype=np.float64)
    else:
        pos = np.asarray(cubes[i], dtype=np.float64)
        onehot[i] = 1.0
    return np.concatenate([pos, pos - np.asarray(eef, dtype=np.float64), onehot])
