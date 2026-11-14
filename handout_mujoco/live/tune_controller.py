"""Systematic controller sweep: make MuJoCo's Panda track like panda-gym's.

The target is NOT a ratio of 1.0. Measured on the recorded episodes, PyBullet itself
delivers a median achieved/commanded end-effector displacement of 0.655 (expert) /
0.597 (cheap) -- its controller does not complete the full 0.05 m either. Tuning
MuJoCo to 1.0 would make it track BETTER than the system the policies were trained
in, which is a different mismatch, not a fix. So the acceptance band is PyBullet's
own ratio.

Baseline problem, measured: ratio 0.21 at 20 substeps, with actuators 2, 4 and 6
pinned at their force limits on the first substep. Scaling the action does nothing
because the servo is already against its torque ceiling.

Swept, on MuJoCo's `general` affine-bias position servo:
    kp          gainprm[0] and -biasprm[1]
    kv          -biasprm[2]      (damping)
    forcerange  actuator_forcerange

    python live/tune_controller.py
"""

import itertools
import json
import os

import mujoco
import numpy as np

from mj_env import MJLive, ACTION_SCALE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

PYBULLET_RATIO = 0.655  # measured, expert condition, recorded episodes
PYBULLET_RATIO_CHEAP = 0.597
BAND = (0.55, 0.80)  # acceptable: at least as responsive as PyBullet, not wildly more

SEEDS = [230044, 230011, 230120]
DIRS = [
    np.array(v, float)
    for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])
]


def set_ctrl_params(env, kp_mult, kv_ratio, f_mult, base):
    """Apply a candidate to the 7 arm actuators. Gripper (actuator 8) untouched."""
    kp0, fr0 = base["kp"], base["forcerange"]
    for i in range(7):
        kp = kp0[i] * kp_mult
        env.m.actuator_gainprm[i, 0] = kp
        env.m.actuator_biasprm[i, 1] = -kp
        env.m.actuator_biasprm[i, 2] = -kp * kv_ratio
        env.m.actuator_forcerange[i, 0] = fr0[i, 0] * f_mult
        env.m.actuator_forcerange[i, 1] = fr0[i, 1] * f_mult


def probe(env, seed, direction, settle=200):
    """One commanded step. Returns response metrics."""
    env.reset_from_seed(seed)
    before = env.ee()
    target = before + direction * ACTION_SCALE
    env.d.ctrl[:7] = env.action_to_ctrl(direction)
    env.d.ctrl[7] = 0.0
    disp, peak_f, sat = [], 0.0, 0
    for k in range(settle):
        mujoco.mj_step(env.m, env.d)
        disp.append(np.linalg.norm(env.ee() - before))
        f = np.abs(env.d.actuator_force[:7])
        lim = np.abs(env.m.actuator_forcerange[:7, 1])
        peak_f = max(peak_f, float((f / np.maximum(lim, 1e-9)).max()))
        if k < 20:
            sat = max(sat, int((f >= lim * 0.98).sum()))
    disp = np.array(disp)
    cmd = ACTION_SCALE
    at20 = float(disp[19])
    final = float(disp[-1])
    # time to 90% of what it eventually reaches
    thresh = 0.9 * final
    t90 = int(np.argmax(disp >= thresh)) if (disp >= thresh).any() else settle
    overshoot = float(disp.max() / max(cmd, 1e-9) - 1.0)
    tail = disp[-40:]
    osc = float(tail.std() / max(final, 1e-9))
    err = float(np.linalg.norm(env.ee() - target))
    return {
        "ratio20": at20 / cmd,
        "ratio_final": final / cmd,
        "t90_substeps": t90,
        "peak_force_frac": peak_f,
        "n_saturated": sat,
        "overshoot": overshoot,
        "oscillation": osc,
        "tracking_err_m": err,
    }


def score(agg):
    """Closeness to PyBullet's response, penalised for saturation and instability."""
    r = agg["ratio20"]
    d = abs(r - PYBULLET_RATIO)
    pen = 0.0
    if agg["n_saturated"] >= 3:
        pen += 0.25
    elif agg["n_saturated"] >= 1:
        pen += 0.08
    if agg["oscillation"] > 0.02:
        pen += 0.25
    if agg["overshoot"] > 0.5:
        pen += 0.25
    return d + pen


def main():
    env = MJLive()
    base = {
        "kp": env.m.actuator_gainprm[:7, 0].copy(),
        "forcerange": env.m.actuator_forcerange[:7].copy(),
    }
    print(f"baseline kp {base['kp']}  forcerange hi {base['forcerange'][:, 1]}")
    print(
        f"target ratio at 20 substeps: PyBullet expert {PYBULLET_RATIO:.3f} "
        f"(cheap {PYBULLET_RATIO_CHEAP:.3f}), accept {BAND[0]:.2f}-{BAND[1]:.2f}\n"
    )

    grid = list(
        itertools.product((1, 2, 5, 10, 20), (0.10, 0.05, 0.02), (1, 3, 10, 30))
    )
    rows = []
    for kp_m, kv_r, f_m in grid:
        set_ctrl_params(env, kp_m, kv_r, f_m, base)
        ms = [probe(env, s, d) for s in SEEDS for d in DIRS]
        agg = {k: float(np.median([m[k] for m in ms])) for k in ms[0]}
        agg["n_saturated"] = int(np.max([m["n_saturated"] for m in ms]))
        agg.update(kp_mult=kp_m, kv_ratio=kv_r, force_mult=f_m)
        agg["score"] = score(agg)
        rows.append(agg)

    rows.sort(key=lambda r: r["score"])
    print(
        f"{'kp':>4}{'kv/kp':>7}{'force':>7}{'ratio@20':>10}{'ratio_end':>11}"
        f"{'t90':>6}{'sat':>5}{'over':>8}{'osc':>8}{'score':>8}"
    )
    for r in rows[:12]:
        print(
            f"{r['kp_mult']:>4}{r['kv_ratio']:>7.2f}{r['force_mult']:>7}"
            f"{r['ratio20']:>10.3f}{r['ratio_final']:>11.3f}{r['t90_substeps']:>6}"
            f"{r['n_saturated']:>5}{r['overshoot']:>8.2f}{r['oscillation']:>8.4f}"
            f"{r['score']:>8.3f}"
        )
    print("\n  baseline for reference:")
    set_ctrl_params(env, 1, 0.10, 1, base)
    ms = [probe(env, s, d) for s in SEEDS for d in DIRS]
    b = {k: float(np.median([m[k] for m in ms])) for k in ms[0]}
    b["n_saturated"] = int(np.max([m["n_saturated"] for m in ms]))
    print(
        f"{1:>4}{0.10:>7.2f}{1:>7}{b['ratio20']:>10.3f}{b['ratio_final']:>11.3f}"
        f"{b['t90_substeps']:>6}{b['n_saturated']:>5}{b['overshoot']:>8.2f}"
        f"{b['oscillation']:>8.4f}"
    )

    best3 = rows[:3]
    json.dump(
        {
            "pybullet_ratio_expert": PYBULLET_RATIO,
            "pybullet_ratio_cheap": PYBULLET_RATIO_CHEAP,
            "accept_band": BAND,
            "baseline": b,
            "grid_size": len(grid),
            "all": rows,
            "best3": best3,
        },
        open(os.path.join(RESULTS, "controller_tuning.json"), "w"),
        indent=2,
        default=float,
    )
    print(f"\nwrote results/controller_tuning.json")
    plot(rows, b)
    return best3


def plot(rows, base):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, key, lab in (
        (ax[0], "force_mult", "force-limit multiplier"),
        (ax[1], "kp_mult", "stiffness multiplier kp"),
    ):
        xs = sorted({r[key] for r in rows})
        for kv in sorted({r["kv_ratio"] for r in rows}):
            ys = [
                np.median(
                    [r["ratio20"] for r in rows if r[key] == x and r["kv_ratio"] == kv]
                )
                for x in xs
            ]
            a.plot(xs, ys, "o-", label=f"kv/kp = {kv:.2f}")
        a.axhspan(BAND[0], BAND[1], color="#2ecc71", alpha=0.15)
        a.axhline(
            PYBULLET_RATIO, color="#27ae60", ls="--", lw=1.5, label="PyBullet (0.655)"
        )
        a.axhline(
            base["ratio20"], color="#c0392b", ls=":", lw=1.5, label="baseline (0.21)"
        )
        a.set_xscale("log")
        a.set_xlabel(lab)
        a.set_ylabel("achieved / commanded at 0.04 s")
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8)
    fig.suptitle(
        "MuJoCo Panda controller sweep — tracking a 0.05 m EE command in one control period"
    )
    fig.tight_layout()
    p = os.path.join(RESULTS, "controller_tuning.png")
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
