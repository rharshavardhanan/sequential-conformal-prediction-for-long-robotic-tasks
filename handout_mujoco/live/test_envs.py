"""Gate 1 smoke tests for the two single-phase envs. No training, no SB3.

Style follows live/test_training_env.py: every check prints PASS/FAIL and the
measured value, and the process exit code is the gate.
"""

import envs
import numpy as np
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


def servo(env, target, n):
    """Drive the EE toward a point with a P controller. No policy involved."""
    out = None
    for _ in range(n):
        a = np.clip((np.asarray(target) - env.sim.ee()) / 0.05, -1, 1)
        out = env.step(a)
        if out[2] or out[3]:
            break
    return out


def test_reach():
    print("\n-- ReachOnly")
    env = ReachOnly()
    obs, _ = env.reset(seed=0)
    check(
        "obs shape (24,) float32",
        obs.shape == (24,) and obs.dtype == np.float32,
        f"{obs.shape} {obs.dtype}",
    )
    check("obs inside observation_space", env.observation_space.contains(obs))
    check("obs is finite", bool(np.all(np.isfinite(obs))))
    check(
        "reach_goal is the Option-A waypoint",
        np.allclose(
            env.sim.reach_goal,
            envs.waypoint(env.sim.cube(), env.sim.task_goal),
            atol=1e-6,
        ),
    )

    env.reset(seed=1)
    ee0 = env.sim.ee().copy()
    env.step(np.array([1.0, 0.0, 0.0]))
    d_act = np.linalg.norm(env.sim.ee() - ee0)
    env.reset(seed=1)
    env.step(np.zeros(3))
    d_zero = np.linalg.norm(env.sim.ee() - ee0)
    check("a=[1,0,0] moves the EE", d_act > 1e-3, f"|d| = {d_act:.4f} m")
    check(
        "motion is caused by the action",
        d_act > 3 * max(d_zero, 1e-9),
        f"action {d_act:.4f} vs zero {d_zero:.4f} m",
    )

    env.reset(seed=3)
    out = servo(env, env.sim.reach_goal.copy(), 50)
    check(
        "reach success sets terminated, not truncated",
        out[2] and not out[3] and out[4]["is_success"],
        f"terminated={out[2]} truncated={out[3]} success={out[4]['is_success']} "
        f"d={out[4]['d_reach']:.4f}",
    )
    check(
        "success fires inside the radius",
        out[4]["d_reach"] < SUCCESS_RADIUS,
        f"{out[4]['d_reach']:.4f} < {SUCCESS_RADIUS}",
    )

    env.reset(seed=4)
    fired = None
    for _ in range(envs.REACH_LIMIT + 5):
        away = env.sim.ee() - env.sim.reach_goal
        o = env.step(np.clip(away / 0.05, -1, 1))
        if o[2] or o[3]:
            fired = o
            break
    check(
        "reach budget expiry TRUNCATES (a time limit is not MDP-terminal)",
        fired is not None
        and fired[3]
        and not fired[2]
        and fired[4]["steps"] == envs.REACH_LIMIT
        and not fired[4]["is_success"],
        f"steps={fired[4]['steps'] if fired else None}, "
        f"terminated={fired[2] if fired else None}, "
        f"truncated={fired[3] if fired else None}",
    )

    seq = [
        np.array([0.5, -0.3, 0.2]),
        np.array([-0.2, 0.7, -0.1]),
        np.array([0.1, 0.1, 0.9]),
    ]
    runs = []
    for _ in range(2):
        o, _ = env.reset(seed=123)
        tape = [o.copy()]
        for a in seq:
            tape.append(env.step(a)[0].copy())
        runs.append(np.array(tape))
    check(
        "same seed -> bit-identical trajectory",
        bool(np.array_equal(runs[0], runs[1])),
        f"max |diff| = {np.abs(runs[0] - runs[1]).max():.2e}",
    )
    check(
        "different seeds -> different initial obs",
        not np.array_equal(env.reset(seed=7)[0], env.reset(seed=8)[0]),
    )


def test_push():
    print("\n-- PushOnly")
    env = PushOnly()
    obs, _ = env.reset(seed=0)
    check(
        "obs is a dict with the three HER keys",
        set(obs) == {"observation", "achieved_goal", "desired_goal"},
        f"{sorted(obs)}",
    )
    check(
        "obs shapes 18/3/3 float32",
        obs["observation"].shape == (18,)
        and obs["achieved_goal"].shape == (3,)
        and obs["desired_goal"].shape == (3,)
        and all(v.dtype == np.float32 for v in obs.values()),
    )
    check("obs inside observation_space", env.observation_space.contains(obs))
    check(
        "achieved_goal is the cube",
        np.allclose(obs["achieved_goal"], env.sim.cube(), atol=1e-6),
    )
    check(
        "desired_goal is the task goal",
        np.allclose(obs["desired_goal"], env.sim.task_goal, atol=1e-6),
    )

    _, r, _, _, i1 = env.step(np.zeros(3))
    r_cr = env.compute_reward(env.sim.cube(), env.sim.task_goal, i1)
    check(
        "compute_reward agrees with step()'s reward",
        float(r_cr) == float(r),
        f"compute_reward {float(r_cr)} vs step {float(r)}",
    )
    ag = np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02], [0.30, 0.0, 0.02]])
    dg = np.array([[0.0, 0.0, 0.02], [0.04, 0.0, 0.02], [0.0, 0.0, 0.02]])
    batch = env.compute_reward(ag, dg, [{}] * 3)
    check(
        "compute_reward vectorizes over a batch",
        isinstance(batch, np.ndarray)
        and batch.shape == (3,)
        and np.array_equal(batch, np.array([0.0, 0.0, -1.0])),
        f"{batch}",
    )

    stats = []
    for s in range(40):
        env.reset(seed=1000 + s)
        c0 = env.sim.cube().copy()
        pen = envs.cube_robot_penetration(env.sim)
        d_hand = float(np.linalg.norm(env.sim.ee() - c0))
        retreat = env.reset_retreat
        env.step(np.zeros(3))
        stats.append(
            (
                pen,
                float(np.linalg.norm(env.sim.cube() - c0)),
                float(np.linalg.norm(c0 - env.sim.task_goal)),
                d_hand,
                retreat,
            )
        )
    pen, jump, sep, hand, retreat = (np.array(x) for x in zip(*stats))
    check(
        "no reset buries the gripper in the cube",
        pen.min() >= envs.MAX_RESET_PENETRATION,
        f"deepest {pen.min():.5f} >= {envs.MAX_RESET_PENETRATION} "
        f"({np.mean(retreat > 0):.0%} needed a retreat, max {1000 * retreat.max():.0f} mm)",
    )
    check(
        "a zero action does not eject the cube",
        jump.max() < 5e-3,
        f"max jump {jump.max() * 1000:.2f} mm",
    )
    check(
        "no episode is born successful",
        (sep >= SUCCESS_RADIUS).all(),
        f"min separation {sep.min():.4f}",
    )
    check(
        "the arm starts at the cube, as reach would leave it",
        hand.max() < 0.15,
        f"||ee-cube|| median {np.median(hand):.4f} max {hand.max():.4f}",
    )

    env.reset(seed=5)
    fired = None
    for _ in range(envs.PUSH_LIMIT + 5):
        o = env.step(np.zeros(3))
        if o[2] or o[3]:
            fired = o
            break
    check(
        "push timeout truncates (not terminates) at the 50-step budget",
        fired is not None
        and fired[3]
        and not fired[2]
        and fired[4]["steps"] == envs.PUSH_LIMIT,
        f"truncated={fired[3] if fired else None}, "
        f"terminated={fired[2] if fired else None}, "
        f"steps={fired[4]['steps'] if fired else None}",
    )

    env.reset(seed=6)
    out = servo(env, env.sim.task_goal.copy(), envs.PUSH_LIMIT)
    check(
        "push success sets terminated, not truncated",
        out[2] and not out[3] and out[4]["is_success"] and out[1] == 0.0,
        f"terminated={out[2]} success={out[4]['is_success']} reward={out[1]} "
        f"d_push={out[4]['d_push']:.4f}",
    )

    # action causality, the push env's own copy (it shares MJLive.step with reach, but
    # its reset is different, so the causal link is re-established here)
    env.reset(seed=11)
    ee0 = env.sim.ee().copy()
    env.step(np.array([1.0, 0.0, 0.0]))
    d_act = np.linalg.norm(env.sim.ee() - ee0)
    env.reset(seed=11)
    env.step(np.zeros(3))
    d_zero = np.linalg.norm(env.sim.ee() - ee0)
    check(
        "motion is caused by the action",
        d_act > 1e-3 and d_act > 3 * max(d_zero, 1e-9),
        f"action {d_act:.4f} vs zero {d_zero:.4f} m",
    )

    # cube-off-table branch. Driving the cube at the +x table edge does NOT reach it
    # (measured: the cube stalls around x = 0.12 against an edge at x = 0.25 over 12
    # seeds), so the branch is exercised by putting the cube below the threshold
    # directly. It is kept because Phase 4 doubles the step budget and a trained policy
    # can fling the cube in ways a P-servo does not.
    env.reset(seed=12)
    q = env.sim.cube_qadr
    env.sim.d.qpos[q + 2] = -0.20
    env.sim.d.qvel[:] = 0.0
    o = env.step(np.zeros(3))
    check(
        "cube below the table terminates with cube_fell",
        o[2] and not o[3] and o[4]["cube_fell"] and not o[4]["is_success"],
        f"terminated={o[2]} truncated={o[3]} cube_fell={o[4]['cube_fell']} "
        f"cube_z={o[4]['cube'][2]:.3f}",
    )

    # FIX 2: the reset must reproduce the cube disturbance reach inflicts. The waypoint
    # stays anchored to the UNDISTURBED cube, so ||cube - reach_goal|| departs from
    # WAYPOINT_OFFSET by exactly the disturbance the reset applied.
    a = envs.handoff_sample()
    check(
        "the measured hand-off sample loads, 6 columns",
        a.ndim == 2 and a.shape[1] == 6 and len(a) >= 200,
        f"shape {a.shape} from {envs.HANDOFF_SAMPLE.name}",
    )
    d, o = envs.sample_handoff(np.random.default_rng(0), np.array([0.1, 0.0, CUBE_Z]),
                               np.array([-0.1, 0.0, CUBE_Z]))
    check(
        "sample_handoff returns a (disturbance, offset) pair with disturbance z dropped",
        d.shape == (3,) and o.shape == (3,) and d[2] == 0.0,
        f"disturbance {np.round(d, 4)} offset {np.round(o, 4)}",
    )
    disp = []
    for s_ in range(300):
        env.reset(seed=5000 + s_)
        disp.append(abs(float(np.linalg.norm(env.sim.cube() - env.sim.reach_goal))
                        - WAYPOINT_OFFSET))
    disp = np.array(disp)
    frac = float(np.mean(disp > 1e-3))
    check(
        "the reset perturbs the cube at the measured rate, not always and not never",
        0.15 < frac < 0.50 and disp.max() < 0.10,
        f"{frac:.1%} of resets disturbed >1 mm (measured reach rate 31.7%), "
        f"max {disp.max():.4f} m",
    )
    check(
        "the undisturbed mass is preserved",
        float(np.mean(disp <= 1e-3)) > 0.50,
        f"{np.mean(disp <= 1e-3):.1%} of resets leave the cube where it was sampled",
    )

    seq = [np.array([0.4, 0.2, -0.1]), np.array([-0.3, 0.5, 0.0])]
    runs = []
    for _ in range(2):
        o, _ = env.reset(seed=321)
        tape = [
            np.concatenate([o["observation"], o["achieved_goal"], o["desired_goal"]])
        ]
        for a in seq:
            o2 = env.step(a)[0]
            tape.append(
                np.concatenate(
                    [o2["observation"], o2["achieved_goal"], o2["desired_goal"]]
                )
            )
        runs.append(np.array(tape))
    check(
        "same seed -> bit-identical trajectory",
        bool(np.array_equal(runs[0], runs[1])),
        f"max |diff| = {np.abs(runs[0] - runs[1]).max():.2e}",
    )


def test_check_env():
    print("\n-- gymnasium API conformance")
    from gymnasium.utils.env_checker import check_env

    for name, cls in (("ReachOnly", ReachOnly), ("PushOnly", PushOnly)):
        try:
            check_env(cls(), skip_render_check=True)
            check(f"check_env passes for {name}", True)
        except Exception as e:  # noqa: BLE001 - any failure is a gate failure
            check(f"check_env passes for {name}", False, f"{type(e).__name__}: {e}")


def main():
    print("Gate 1 - single-phase MuJoCo environments")
    test_helpers()
    test_reach()
    test_push()
    test_check_env()
    n = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n}/{len(results)} checks passed")
    return n == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
