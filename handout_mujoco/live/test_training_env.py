"""Smoke test for the Gymnasium Reach->Push environment. No training, no SB3.

Checks the eight properties the spec requires, plus Gymnasium's own API conformance
checker (which is what SB3 would rely on, so it validates compatibility without
installing SB3).
"""

import numpy as np

from mujoco_reach_push_env import (
    MAX_STEPS,
    MuJoCoReachPush,
    PUSH,
    PUSH_LIMIT,
    REACH,
    REACH_LIMIT,
    SUCCESS_RADIUS,
)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def servo_to(env, target, n=60, grip_cube=False):
    """Drive the EE toward a point with a simple P controller (no policy involved)."""
    last = None
    for _ in range(n):
        ee = env.sim.ee()
        v = np.asarray(target) - ee
        a = np.clip(v / 0.05, -1, 1)
        last = env.step(a)
        if last[2] or last[3]:
            break
    return last


def main():
    print("Gymnasium Reach->Push environment — smoke test\n")

    # ---------------------------------------------------------------- 1. reset works
    env = MuJoCoReachPush()
    obs, info = env.reset(seed=0)
    check("reset() returns (obs, info)", isinstance(info, dict) and obs is not None)
    check("observation shape is (26,)", obs.shape == (26,), f"got {obs.shape}")
    check("observation dtype float32", obs.dtype == np.float32, f"got {obs.dtype}")
    check("observation is finite", bool(np.all(np.isfinite(obs))))
    check("observation in observation_space", env.observation_space.contains(obs))
    check(
        "action_space is Box(-1,1,(3,))",
        env.action_space.shape == (3,)
        and float(env.action_space.low[0]) == -1.0
        and float(env.action_space.high[0]) == 1.0,
    )
    check("starts in REACH phase", info["phase"] == REACH)

    # --------------------------------------------------- 2. the action moves the robot
    obs0, _ = env.reset(seed=1)
    ee0 = env.sim.ee().copy()
    env.step(np.array([1.0, 0.0, 0.0]))
    d_act = np.linalg.norm(env.sim.ee() - ee0)
    env.reset(seed=1)
    env.step(np.zeros(3))
    d_zero = np.linalg.norm(env.sim.ee() - ee0)
    check("a=[1,0,0] moves the EE", d_act > 1e-3, f"|d| = {d_act:.4f} m")
    check(
        "motion is caused by the action, not drift",
        d_act > 3 * max(d_zero, 1e-9),
        f"action {d_act:.4f} vs zero-action {d_zero:.4f} m",
    )

    # ------------------------------------------------------- 3. the cube can be moved
    env.reset(seed=2)
    cube0 = env.sim.cube().copy()
    servo_to(env, cube0 + np.array([0, 0, 0.06]), n=40)  # above the cube
    servo_to(env, cube0 - np.array([0.05, 0, -0.0]), n=40)  # push through it
    servo_to(env, cube0 + np.array([0.12, 0, 0]), n=60)
    moved = float(np.linalg.norm(env.sim.cube() - cube0))
    check("the cube can be contacted and moved", moved > 0.01, f"moved {moved:.4f} m")

    # ------------------------------------------------- 4. REACH -> PUSH transition
    env.reset(seed=3)
    rg = env.sim.reach_goal.copy()
    bonuses, out = [], None
    for _ in range(REACH_LIMIT):
        v = rg - env.sim.ee()
        out = env.step(np.clip(v / 0.05, -1, 1))
        bonuses.append(out[4]["bonus"])
        if out[2] or out[3] or env.phase == PUSH:
            break
    reached = out is not None and out[4]["phase"] == PUSH
    check(
        "REACH -> PUSH transition occurs",
        reached,
        f"phase={out[4]['phase_name']}, d_reach={out[4]['d_reach']:.4f}"
        f" (radius {SUCCESS_RADIUS})",
    )
    check(
        "transition awards the +10 bonus",
        reached and out[4]["bonus"] == 10.0,
        f"bonus={out[4]['bonus']}",
    )
    check(
        "phase latches (does not return to REACH)",
        reached and env.step(np.zeros(3))[4]["phase"] == PUSH,
    )

    # ------------------------------------------------------ 5. reach timeout ends it
    env.reset(seed=4)
    term = trunc = False
    for i in range(REACH_LIMIT + 5):
        away = env.sim.ee() - env.sim.reach_goal          # move away, never into the table
        _, _, term, trunc, inf = env.step(np.clip(away / 0.05, -1, 1))
        if term or trunc:
            break
    check(
        "REACH timeout terminates the episode",
        term and inf["phase"] == REACH,
        f"after {inf['steps']} steps, terminated={term}",
    )
    check(
        "reach timeout fires at the 50-step budget",
        inf["steps"] == REACH_LIMIT,
        f"steps={inf['steps']}, limit={REACH_LIMIT}",
    )

    # ------------------------------------------------- 6. push timeout truncates
    env.reset(seed=5)
    servo_to(env, env.sim.reach_goal.copy(), n=REACH_LIMIT)
    if env.phase == PUSH:
        t2 = f2 = False
        for _ in range(PUSH_LIMIT + 5):
            _, _, t2, f2, inf2 = env.step(np.zeros(3))
            if t2 or f2:
                break
        check(
            "PUSH timeout truncates (not terminates)",
            f2 and not t2,
            f"truncated={f2}, terminated={t2}, phase_steps={inf2['phase_steps']}",
        )
    else:
        check("PUSH timeout truncates (not terminates)", False, "never reached PUSH")

    # ----------------------------------------------------------- 7. reproducibility
    a_seq = [
        np.array([0.5, -0.3, 0.2]),
        np.array([-0.2, 0.7, -0.1]),
        np.array([0.1, 0.1, 0.9]),
    ]
    runs = []
    for _ in range(2):
        o, _ = env.reset(seed=123)
        tape = [o.copy()]
        for a in a_seq:
            tape.append(env.step(a)[0].copy())
        runs.append(np.array(tape))
    check(
        "same seed -> identical trajectory",
        bool(np.array_equal(runs[0], runs[1])),
        f"max |diff| = {np.abs(runs[0] - runs[1]).max():.2e}",
    )
    o_a, _ = env.reset(seed=7)
    o_b, _ = env.reset(seed=8)
    check("different seeds -> different initial states", not np.array_equal(o_a, o_b))

    # ------------------------------------------------------------ 8. score plumbing
    env.reset(seed=9)
    _, _, _, _, i_big = env.step(np.array([1.0, 0.0, 0.0]))
    _, _, _, _, i_zero = env.step(np.zeros(3))
    check(
        "conformal score is finite for a large command",
        i_big["scorable"] and np.isfinite(i_big["score"]),
        f"s={i_big['score']:.4f}",
    )
    check("near-zero command is marked unscorable", not i_zero["scorable"])

    # ------------------------------------------------ 9. Gymnasium API conformance
    try:
        from gymnasium.utils.env_checker import check_env

        check_env(MuJoCoReachPush(), skip_render_check=True)
        check("gymnasium check_env passes (SB3-compatible API)", True)
    except Exception as e:
        check(
            "gymnasium check_env passes (SB3-compatible API)",
            False,
            f"{type(e).__name__}: {e}",
        )

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} checks passed")
    return n_pass == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
