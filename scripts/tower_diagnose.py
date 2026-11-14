"""Phase 5 — find the FIRST closed-loop failure of the DART policies.

No retraining. This runs the existing checkpoints in the existing closed loop
(scripts/tower_policies.evaluate uses live sim state; nothing is replayed) and
instruments every timestep so the earliest thing that goes wrong is identified
by measurement rather than assumption.

    .venv-lerobot/bin/python -m scripts.tower_diagnose --n 20

Writes data/tower/closed_loop_diagnostic.json (summaries + every grasp event +
per-episode first failure) and data/tower/closed_loop_traces.npz (the full
per-timestep tape, which is too large for JSON).
"""

import argparse
import json
import pathlib
import time

import numpy as np

from scripts.tower_collect import make_env, observation
from scripts.tower_control import joint_action, eef_pos
from scripts.tower_policies import ACTPolicy, BCRNNPolicy
from scripts.tower_target import STACKED as _STACKED, target_cube

HERE = pathlib.Path(__file__).resolve().parent.parent

CUBE_HALF = 0.022  # cube half-extent, metres
from scripts.tower_target import STACKED  # noqa: F401  (one definition)
LIFTED = 0.06  # z above table that counts as picked up
NEAR = 0.06  # eef-to-cube distance that counts as "held"
CLOSE_OK = 0.03  # the gripper's usable closing radius
RESOLVE = 60  # steps allowed for a grasp attempt to prove itself

FAILURES = {
    "A": "wrong cube selection",
    "B": "wrong spatial targeting",
    "C": "bad approach trajectory (never attempted a grasp)",
    "D": "premature/late gripper closure",
    "E": "failed grasp despite good alignment",
    "F": "failed lift",
    "G": "failed carry",
    "H": "failed placement/release",
    "-": "no failure (task succeeded)",
}


def cubes_of(env):
    return np.array([env.sim.data.body_xpos[b] for b in env.cube_body_ids])


def intended_cube(cubes, table_z):
    """Delegates to the one shared implementation (scripts/tower_target.py).

    Collection and evaluation must agree on this rule; two copies drift silently.
    """
    return target_cube(cubes, table_z)


def stage_of(grip_closed, dz, holding):
    """Infer the task stage for a policy that has no state machine."""
    if holding:
        if dz > 0.004:
            return "lift"
        if dz < -0.004:
            return "place"
        return "carry"
    if grip_closed:
        return "grasp"
    return "retreat" if dz > 0.004 else "reach"


def run_episode(env, policy, max_steps, ep, pname):
    env.reset()
    policy.reset()
    table_z = env.table_offset[2]

    tape = []
    grasps = []
    pending = None
    prev_grip = False
    prev_z = None
    min_dist_seen = {}  # cube id -> closest the eef ever got, before closure
    lifted_any = False
    carried_any = False

    for t in range(max_steps):
        obs = observation(env)[: policy.cfg["sdim"]]
        q, g = policy.act(obs)
        grip_closed = g >= 0.5

        cubes = cubes_of(env)
        e = eef_pos(env)
        tgt_i, base_i = intended_cube(cubes, table_z)
        d_int = float(np.linalg.norm(cubes[tgt_i] - e)) if tgt_i is not None else np.nan
        dists = np.linalg.norm(cubes - e, axis=1)
        near_i = int(np.argmin(dists))

        for i, d in enumerate(dists):
            min_dist_seen[i] = min(min_dist_seen.get(i, 1e9), float(d))

        holding = bool(
            dists.min() < NEAR and cubes[int(np.argmin(dists))][2] - table_z > LIFTED
        )
        dz = 0.0 if prev_z is None else float(e[2] - prev_z)
        stage = stage_of(grip_closed, dz, holding)
        if holding:
            lifted_any = True
            if stage == "carry":
                carried_any = True

        tape.append(
            np.concatenate(
                [
                    [
                        ep,
                        t,
                        {
                            "reach": 0,
                            "grasp": 1,
                            "lift": 2,
                            "carry": 3,
                            "place": 4,
                            "retreat": 5,
                        }.get(stage, 0),
                    ],
                    e,
                    cubes[tgt_i] if tgt_i is not None else np.full(3, np.nan),
                    [d_int],
                    np.asarray(env.robots[0]._joint_positions),
                    [float(grip_closed)],
                    np.asarray(q),
                    [float(env.tower_height())],
                ]
            ).astype(np.float32)
        )

        # ---- grasp attempt opens on the first closed step ----
        if grip_closed and not prev_grip:
            pending = {
                "t": t,
                "stage_before": stage,
                "intended": None if tgt_i is None else int(tgt_i),
                "nearest": near_i,
                "dist_intended": d_int,
                "dist_nearest": float(dists[near_i]),
                "min_dist_intended_before": (
                    float(min_dist_seen.get(tgt_i, np.nan))
                    if tgt_i is not None
                    else np.nan
                ),
                "eef": e.tolist(),
                "cube": cubes[tgt_i].tolist() if tgt_i is not None else None,
                "height_above_cube": (
                    float(e[2] - cubes[tgt_i][2]) if tgt_i is not None else np.nan
                ),
                "horiz_dist": (
                    float(np.linalg.norm((cubes[tgt_i] - e)[:2]))
                    if tgt_i is not None
                    else np.nan
                ),
                "resolved": False,
                "grasped": False,
            }
        # ---- resolve it: did a cube actually come up with the gripper? ----
        if pending is not None and not pending["resolved"]:
            if t - pending["t"] >= RESOLVE or (
                t - pending["t"] > 3 and not grip_closed
            ):
                pending["grasped"] = bool(holding)
                pending["resolved"] = True
                grasps.append(pending)
                pending = None

        prev_grip, prev_z = grip_closed, float(e[2])
        env.step(joint_action(env, q, 1.0 if grip_closed else -1.0))
        if env._check_success():
            break

    if pending is not None:
        pending["grasped"] = False
        pending["resolved"] = True
        grasps.append(pending)

    success = bool(env._check_success())
    height = int(env.tower_height())
    fail = classify(success, grasps, lifted_any, carried_any, height, min_dist_seen)
    return {
        "policy": pname,
        "episode": ep,
        "success": success,
        "height": height,
        "steps": t + 1,
        "grasps": grasps,
        "failure": fail,
        "n_grasp_attempts": len(grasps),
    }, np.stack(tape)


def classify(success, grasps, lifted, carried, height, min_seen):
    """Earliest failure event, as one of A-H."""
    if success:
        return {"code": "-", "why": FAILURES["-"], "t": None}
    if not grasps:
        return {
            "code": "C",
            "why": FAILURES["C"] + " — gripper never closed",
            "t": None,
        }

    g = grasps[0]
    d, dmin = g["dist_intended"], g["min_dist_intended_before"]
    dn, near, intd = g["dist_nearest"], g["nearest"], g["intended"]

    if np.isfinite(dmin) and dmin < CLOSE_OK and d > CLOSE_OK:
        return {
            "code": "D",
            "why": "closed too LATE — had been within "
            f"{dmin * 100:.1f} cm but closed at {d * 100:.1f} cm",
            "t": g["t"],
        }
    if np.isfinite(d) and d > 0.05 and (not np.isfinite(dmin) or dmin > 0.05):
        return {
            "code": "B",
            "why": "closed "
            f"{d * 100:.1f} cm from the intended cube, never got closer than "
            f"{dmin * 100:.1f} cm",
            "t": g["t"],
        }
    if intd is not None and near != intd and dn < CLOSE_OK:
        return {
            "code": "A",
            "why": f"closed on cube {near}, intended {intd}",
            "t": g["t"],
        }
    if np.isfinite(d) and CLOSE_OK < d <= 0.05:
        return {
            "code": "D",
            "why": "closed EARLY — "
            f"{d * 100:.1f} cm out, {g['height_above_cube'] * 100:.1f} cm above the cube",
            "t": g["t"],
        }
    if np.isfinite(d) and d <= CLOSE_OK and not g["grasped"]:
        return {
            "code": "E",
            "why": f"aligned to {d * 100:.1f} cm but the cube did not come up",
            "t": g["t"],
        }
    if g["grasped"] and not lifted:
        return {"code": "F", "why": FAILURES["F"], "t": g["t"]}
    if lifted and not carried:
        return {"code": "G", "why": FAILURES["G"], "t": g["t"]}
    return {
        "code": "H",
        "why": FAILURES["H"] + f" — ended at height {height}",
        "t": g["t"],
    }


def summarise(eps, pname):
    allg = [g for e in eps for g in e["grasps"]]
    d = np.array([g["dist_intended"] for g in allg if np.isfinite(g["dist_intended"])])
    firsts = [e["grasps"][0] for e in eps if e["grasps"]]
    df = np.array(
        [g["dist_intended"] for g in firsts if np.isfinite(g["dist_intended"])]
    )
    codes = {}
    for e in eps:
        codes[e["failure"]["code"]] = codes.get(e["failure"]["code"], 0) + 1
    late = sum(
        1 for e in eps if e["failure"]["code"] == "D" and "LATE" in e["failure"]["why"]
    )
    early = sum(
        1 for e in eps if e["failure"]["code"] == "D" and "EARLY" in e["failure"]["why"]
    )
    wrongloc = sum(
        1
        for g in allg
        if np.isfinite(g["min_dist_intended_before"])
        and g["min_dist_intended_before"] > 0.05
    )
    return {
        "policy": pname,
        "n_episodes": len(eps),
        "success_rate": float(np.mean([e["success"] for e in eps])),
        "height_mean": float(np.mean([e["height"] for e in eps])),
        "height_hist": {
            str(h): sum(1 for e in eps if e["height"] == h)
            for h in sorted({e["height"] for e in eps})
        },
        "steps_mean": float(np.mean([e["steps"] for e in eps])),
        "grasp_attempts_total": len(allg),
        "grasp_attempts_per_episode": len(allg) / max(1, len(eps)),
        "grasp_success_rate": float(np.mean([g["grasped"] for g in allg]))
        if allg
        else 0.0,
        "close_dist_median": float(np.median(d)) if d.size else None,
        "close_dist_mean": float(d.mean()) if d.size else None,
        "within_1cm": float(np.mean(d < 0.01)) if d.size else None,
        "within_3cm": float(np.mean(d < 0.03)) if d.size else None,
        "within_5cm": float(np.mean(d < 0.05)) if d.size else None,
        "first_close_dist_median": float(np.median(df)) if df.size else None,
        "closed_too_early": early,
        "closed_too_late": late,
        "attempts_never_within_5cm": wrongloc,
        "failure_codes": codes,
        "episodes_with_no_grasp": sum(1 for e in eps if not e["grasps"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--variants", nargs="+", default=["dart"],
                    help="checkpoint dirs under data/tower (dart = 32-dim, rel = 41-dim)")
    ap.add_argument("--out", default="data/tower/closed_loop_diagnostic.json")
    ap.add_argument("--traces", default="data/tower/closed_loop_traces.npz")
    a = ap.parse_args()

    specs = [(f"BC-RNN/{v}", BCRNNPolicy, f"data/tower/{v}/tower_bcrnn.pt")
             for v in a.variants]
    specs += [(f"ACT/{v}", ACTPolicy, f"data/tower/{v}/tower_act.pt")
              for v in a.variants]
    out = {
        "n_episodes": a.n,
        "max_steps": a.max_steps,
        "seed": a.seed,
        "failure_legend": FAILURES,
        "policies": {},
    }
    traces = {}

    for pname, cls, ckpt in specs:
        print(f"\n{'=' * 68}\n{pname}  ({ckpt})\n{'=' * 68}", flush=True)
        env = make_env(seed=a.seed)
        pol = cls(ckpt, device=a.device)
        t0 = time.perf_counter()
        eps, tp = [], []
        for i in range(a.n):
            r, tape = run_episode(env, pol, a.max_steps, i, pname)
            eps.append(r)
            tp.append(tape)
            f = r["failure"]
            gd = (
                f"{r['grasps'][0]['dist_intended'] * 100:.1f}cm"
                if r["grasps"] and np.isfinite(r["grasps"][0]["dist_intended"])
                else "  —  "
            )
            print(
                f"  ep{i:3d}  h{r['height']}  {r['steps']:4d} steps  "
                f"{r['n_grasp_attempts']:2d} closes  first@{gd}  "
                f"[{f['code']}] {f['why'][:58]}",
                flush=True,
            )
        env.close()
        s = summarise(eps, pname)
        s["checkpoint"] = ckpt
        s["fingerprint"] = pol.fingerprint()
        s["seconds_per_episode"] = (time.perf_counter() - t0) / a.n
        out["policies"][pname] = {"summary": s, "episodes": eps}
        traces[pname] = np.concatenate(tp)

        print(
            f"\n  success {s['success_rate']:.0%}   mean height {s['height_mean']:.2f}"
            f"   {s['grasp_attempts_per_episode']:.1f} closes/ep"
            f"   grasp success {s['grasp_success_rate']:.0%}"
        )
        print(
            f"  close distance: median "
            f"{(s['close_dist_median'] or float('nan')) * 100:.1f} cm   "
            f"<1cm {s['within_1cm'] or 0:.0%}  <3cm {s['within_3cm'] or 0:.0%}  "
            f"<5cm {s['within_5cm'] or 0:.0%}"
        )
        print(f"  first failures: {s['failure_codes']}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    np.savez_compressed(a.traces, **traces)
    cols = (
        "episode t stage eef_x eef_y eef_z cube_x cube_y cube_z dist_to_cube "
        "q0..q6 grip pred_q0..pred_q6 tower_height"
    )
    print(f"\nwrote {a.out}")
    print(f"wrote {a.traces}  (columns: {cols})")


if __name__ == "__main__":
    main()
