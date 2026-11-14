"""Phase 1 — trustworthy closed-loop baseline for the scripted expert.

Establishes what the environment and controller can actually do before any
learned policy is judged against it, and verifies that replaying a recorded
expert trajectory really does drive the simulator to the same outcome (i.e.
the loop is causal, not a lookup).

    .venv-lerobot/bin/python -m scripts.tower_baseline --n 50

Every number here is measured in closed loop: the expert reads live sim state
each step and its action changes that state. Nothing is replayed from a file
except in the explicit replay check.
"""

import argparse, json, pathlib, time

import numpy as np

from scripts.tower_collect import make_env, observation  # noqa: F401
from scripts.tower_control import TowerExpert, joint_action, eef_pos

HERE = pathlib.Path(__file__).resolve().parent.parent
LIFTED = 0.06  # metres above table that counts as "off the table"
GRASP_OK = 0.03  # the gripper's usable closing radius


def cube_xyz(env):
    return [np.array(env.sim.data.body_xpos[b]) for b in env.cube_body_ids]


def run_episode(env, expert, max_steps, record=False):
    """One expert episode. Returns metrics and, optionally, the action tape."""
    env.reset()
    expert.reset()
    table_z = env.table_offset[2]

    tape = []
    grasps, places = [], []
    pending = None  # grasp attempt in flight
    prev_stage = None
    ctrl_fail = []

    for t in range(max_steps):
        q, g = expert.act()
        stage = expert.stage

        # a grasp attempt begins the step the gripper first closes
        if stage == "grasp" and prev_stage != "grasp":
            cubes = cube_xyz(env)
            tgt_i = expert.order[expert.idx] if expert.idx < len(expert.order) else None
            e = eef_pos(env)
            d_near = min(np.linalg.norm(c - e) for c in cubes)
            pending = {
                "t": t,
                "cube": tgt_i,
                "eef": e.copy(),
                "dist_target": float(np.linalg.norm(cubes[tgt_i] - e))
                if tgt_i is not None
                else float("nan"),
                "dist_nearest": float(d_near),
                "z0": float(cubes[tgt_i][2]) if tgt_i is not None else float("nan"),
            }

        # resolve it once the arm has finished lifting
        if pending is not None and stage == "carry" and prev_stage == "lift":
            z = cube_xyz(env)[pending["cube"]][2] if pending["cube"] is not None else -1
            pending["lifted"] = bool(z - table_z > LIFTED)
            pending["z1"] = float(z)
            grasps.append(pending)
            pending = None

        # a placement resolves when the expert releases and retreats
        if stage == "retreat" and prev_stage == "release":
            places.append({"t": t, "height": int(env.tower_height())})

        prev_stage = stage
        act = joint_action(env, q, g)
        if not np.all(np.isfinite(act)):
            ctrl_fail.append({"t": t, "why": "non-finite action"})
            break
        if record:
            tape.append(act.copy())
        env.step(act)

        # controller health: is the arm actually tracking the target it was given?
        err = float(np.max(np.abs(np.asarray(env.robots[0]._joint_positions) - q)))
        if err > 1.0:
            ctrl_fail.append({"t": t, "why": f"joint tracking error {err:.2f} rad"})

        if env._check_success() and expert.idx >= len(expert.order or []):
            break

    if pending is not None:
        pending["lifted"] = False
        pending["z1"] = float("nan")
        grasps.append(pending)

    return {
        "success": bool(env._check_success()),
        "height": int(env.tower_height()),
        "steps": t + 1,
        "grasps": grasps,
        "places": places,
        "ctrl_fail": ctrl_fail,
    }, tape


def replay_check(seed, max_steps):
    """Does replaying the recorded actions reach the same state? If the loop were
    not causal -- if outcomes came from anywhere but the simulator -- this fails."""
    env = make_env(seed=seed)
    rec, tape = run_episode(env, TowerExpert(env), max_steps, record=True)
    env.close()

    env = make_env(seed=seed)  # same seed -> same placement draw
    env.reset()
    for a in tape:
        env.step(a)
    out = {
        "live_height": rec["height"],
        "live_success": rec["success"],
        "replay_height": int(env.tower_height()),
        "replay_success": bool(env._check_success()),
        "n_actions": len(tape),
    }
    env.close()
    out["match"] = (
        out["live_height"] == out["replay_height"]
        and out["live_success"] == out["replay_success"]
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/tower/baseline_expert.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    env = make_env(seed=a.seed)
    expert = TowerExpert(env)
    eps = []
    for i in range(a.n):
        r, _ = run_episode(env, expert, a.max_steps)
        eps.append(r)
        print(
            f"  ep{i:3d}  height {r['height']}  steps {r['steps']:4d}  "
            f"grasps {sum(g['lifted'] for g in r['grasps'])}/{len(r['grasps'])}  "
            f"{'SUCCESS' if r['success'] else 'fail'}",
            flush=True,
        )
    env.close()
    wall = time.perf_counter() - t0

    succ = [e["success"] for e in eps]
    heights = [e["height"] for e in eps]
    steps = [e["steps"] for e in eps]
    allg = [g for e in eps for g in e["grasps"]]
    lifted = [g for g in allg if g["lifted"]]
    dists = np.array([g["dist_target"] for g in allg if np.isfinite(g["dist_target"])])
    nfail = sum(len(e["ctrl_fail"]) for e in eps)

    # per-cube-index grasp rate: is the 2nd/3rd pick harder than the 1st?
    by_order = {}
    for e in eps:
        for k, g in enumerate(e["grasps"]):
            by_order.setdefault(k, []).append(g["lifted"])

    p = float(np.mean(succ))
    n = len(succ)
    half = 1.96 * (p * (1 - p) / n) ** 0.5 if n else 0.0
    out = {
        "n_episodes": n,
        "seed": a.seed,
        "max_steps": a.max_steps,
        "wall_seconds": wall,
        "seconds_per_episode": wall / max(1, n),
        "success_rate": p,
        "success_ci95": [max(0, p - half), min(1, p + half)],
        "height_mean": float(np.mean(heights)),
        "height_median": float(np.median(heights)),
        "height_hist": {str(h): heights.count(h) for h in sorted(set(heights))},
        "steps_mean": float(np.mean(steps)),
        "steps_median": float(np.median(steps)),
        "grasp_attempts": len(allg),
        "grasp_success": len(lifted),
        "grasp_rate": len(lifted) / max(1, len(allg)),
        "grasp_dist_median": float(np.median(dists)) if dists.size else None,
        "grasp_dist_within_1cm": float(np.mean(dists < 0.01)) if dists.size else None,
        "grasp_dist_within_3cm": float(np.mean(dists < 0.03)) if dists.size else None,
        "grasp_dist_within_5cm": float(np.mean(dists < 0.05)) if dists.size else None,
        "grasp_rate_by_pick": {
            str(k): float(np.mean(v)) for k, v in sorted(by_order.items())
        },
        "placements": sum(len(e["places"]) for e in eps),
        "controller_failures": nfail,
    }

    print(
        f"\n{'=' * 66}\nPHASE 1 — expert closed-loop baseline ({n} episodes)\n{'=' * 66}"
    )
    print(
        f"  task success      {p:.1%}   95% CI [{out['success_ci95'][0]:.1%}, "
        f"{out['success_ci95'][1]:.1%}]"
    )
    print(
        f"  tower height      mean {out['height_mean']:.2f}  median "
        f"{out['height_median']:.1f}   {out['height_hist']}"
    )
    print(
        f"  grasp rate        {out['grasp_rate']:.1%}  ({len(lifted)}/{len(allg)} attempts)"
    )
    print(f"  by pick order     {out['grasp_rate_by_pick']}")
    print(
        f"  grasp distance    median {out['grasp_dist_median']:.4f} m   "
        f"<1cm {out['grasp_dist_within_1cm']:.0%}  <3cm {out['grasp_dist_within_3cm']:.0%}"
    )
    print(f"  placements        {out['placements']}")
    print(
        f"  episode length    mean {out['steps_mean']:.0f}  median {out['steps_median']:.0f}"
    )
    print(f"  controller faults {nfail}")
    print(f"  throughput        {out['seconds_per_episode']:.1f} s/episode")

    print(f"\nreplay check (is the loop causal?)")
    rc = replay_check(a.seed + 999, a.max_steps)
    out["replay_check"] = rc
    print(f"  live   height {rc['live_height']} success {rc['live_success']}")
    print(
        f"  replay height {rc['replay_height']} success {rc['replay_success']}  "
        f"({rc['n_actions']} actions)"
    )
    print(f"  -> {'MATCH' if rc['match'] else 'MISMATCH'}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
