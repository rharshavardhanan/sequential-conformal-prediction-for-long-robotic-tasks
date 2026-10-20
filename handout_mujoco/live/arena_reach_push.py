"""Sequential CP safety monitor — MuJoCo Reach->Push policy arbitration.

Same panel grammar as scripts/demo_monitor.py (the LIBERO monitor), pointed at the
MuJoCo system: offscreen render on the left, decision panel on the right, live signal
below, event log at the bottom.

Why offscreen + matplotlib rather than drawing in the MuJoCo scene: mjvScene primitives
cannot render text, axes or numbers, so no labelled threshold plot is achievable there.

Plain python (no mjpython): nothing here opens a GLFW window.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python live/arena_reach_push.py --save arena.mp4
    PYTHONPATH=live ../.venv-lerobot/bin/python live/arena_reach_push.py
"""

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cheap_policy import load as load_cheap
from composed import PUSH, REACH, ComposedEnv, ComposedExpert
from envs import PUSH_LIMIT, PushOnly
from mujoco_arbiter import EXPERT, HANDBACK_STEPS, Arbiter, phase_q
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

import mujoco

TRAIN = ROOT / "results" / "mujoco_train"
WINDOW = 40
HOLD_FRAMES = 40
SUCCESS_RADIUS = 0.05
# a in [-1,1]^3 scaled by 0.05 m, so the largest commandable displacement is
# 0.05*sqrt(3) = 86.6 mm, not 50 mm. A 50 mm scale saturated the bar on any
# diagonal command and read as "maxed out" when it was not.
BAR_FULL_SCALE = 0.05 * (3 ** 0.5)
PHASES = ("reach", "push")

# Camera chosen by rendering candidates and looking at them. It TRACKS the midpoint of
# cube and goal rather than sitting still: a fixed wide shot framed the whole arm but left
# the 4 cm cube about 20 px across, which is the thing the run is actually about.
# Three angles, all TRACKING the midpoint of cube and goal. A single fixed wide shot left
# the 4 cm cube about 20 px across; a single tracking shot fixes the size but the arm
# occludes the cube from whichever side it happens to be on. Two opposed 3/4 views plus a
# plan view mean the cube is legible in at least one panel at every step.
CAMS = (("TRACKING", dict(azimuth=140, elevation=-45, distance=0.72)),
        ("OPPOSITE", dict(azimuth=320, elevation=-40, distance=0.72)),
        # CHASE looks along the push axis, from behind the cube toward the goal, so the
        # two are always lined up in frame. A fixed plan view was tried and discarded:
        # straight down, the forearm covers the cube for most of the push.
        ("CHASE (along push axis)", dict(azimuth=None, elevation=-22, distance=0.55)))

# Goal geoms are contype=0/conaffinity=0, so restyling them is render-only -- verified by
# re-running all three seeds and getting identical trajectories.
G_PUSH_RED, G_PUSH_GREEN = (0.85, 0.15, 0.15, 0.32), (0.15, 0.80, 0.25, 0.45)
G_REACH_BLUE, G_HIDDEN = (0.20, 0.40, 1.00, 0.25), (0.0, 0.0, 0.0, 0.0)

C_EXPERT, C_SAFE, C_BAD = "#ef6c00", "#2e7d32", "#c62828"
BG_CHEAP, BG_EXPERT = "#f0f1f3", "#ffe9d1"


def style_goals(m):
    """Make the push goal the success radius itself, not an arbitrary marker."""
    def by_body(name):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return next(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)
    gid_push, gid_reach, gid_lamp = by_body("push_goal"), by_body("reach_goal"), by_body("mode_lamp")
    m.geom_size[gid_push] = [SUCCESS_RADIUS] * 3
    m.geom_size[gid_reach] = [SUCCESS_RADIUS, 0.0, 0.0]
    m.geom_rgba[gid_lamp] = G_HIDDEN
    return gid_push, gid_reach


def episode_frames(env, arb, renderer, cams, seed, q_hat, gids):
    env.reset(seed)
    arb.reset_stats()
    gid_push, gid_reach = gids
    hist, events = [], []
    info, done, latched = None, None, {}
    for _ in range(100):
        prev = arb.mode
        a, _tag = arb.act(env)
        tr = arb.trace[-1]
        _, term, trunc, info = env.step(a)
        if arb.mode != prev:
            kind = "esc" if arb.mode == EXPERT else "back"
            events.append((tr["step"], kind))
            latched.setdefault("escalate" if kind == "esc" else "hand back", tr["step"])
        if tr["score"] > q_hat[tr["phase"]]:
            latched.setdefault("above q_safe", tr["step"])
        hist.append({"step": tr["step"], "score": tr["score"], "q_esc": tr["q_esc"],
                     "q_hat": q_hat[tr["phase"]], "phase": tr["phase"],
                     "mode": tr["mode"], "d_push": info["d_push"],
                     "cube": info["cube"].copy(), "ee": info["ee"].copy()})
        env.sim.m.geom_rgba[gid_push] = (G_PUSH_GREEN if info["d_push"] < SUCCESS_RADIUS
                                         else G_PUSH_RED)
        env.sim.m.geom_rgba[gid_reach] = G_REACH_BLUE if env.phase == REACH else G_HIDDEN
        cube_xy, goal_xy = env.sim.cube(), env.sim.task_goal
        mid = (cube_xy + goal_xy) / 2 + [0, 0, 0.02]
        v = cube_xy[:2] - goal_xy[:2]
        chase_az = float(np.degrees(np.arctan2(v[1], v[0]))) if np.linalg.norm(v) > 1e-6 else 140.0
        imgs = []
        for (nm, spec), c in zip(CAMS, cams):
            if spec["azimuth"] is None:
                c.azimuth = chase_az
            c.lookat[:] = mid
            renderer.update_scene(env.sim.d, camera=c)
            imgs.append(renderer.render().copy())
        if term or trunc:
            done = bool(info["is_success"])
        pkt = {"imgs": imgs, "hist": list(hist), "events": list(events), "seed": seed,
               "step": tr["step"], "phase": tr["phase"], "mode": tr["mode"],
               "score": tr["score"], "q_esc": tr["q_esc"], "q_hat": q_hat[tr["phase"]],
               "scorable": info["scorable"], "cmd_norm": info["cmd_norm"],
               "realised": info["realised"], "d_push": info["d_push"],
               "phase_steps": info["phase_steps"], "in_band": arb.in_band,
               "task_goal": info["task_goal"].copy(), "reach_goal": info["reach_goal"].copy(),
               "expert_pct": 100 * arb.expert_steps / max(arb.n_steps, 1),
               "calls": arb.expert_calls, "switches": arb.switches,
               "latched": dict(latched), "done": done}
        yield pkt
        if done is not None:
            break
    for _ in range(HOLD_FRAMES):
        yield dict(pkt)


def stream(seeds, env, arb, renderer, cams, q_hat, gids):
    for sd in seeds:
        yield from episode_frames(env, arb, renderer, cams, sd, q_hat, gids)


def _build_figure(title):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # Views span the FULL width. Squeezing them into one column left each render about
    # 180 px, which is the same legibility problem the zoom was meant to fix.
    fig = plt.figure(figsize=(17.0, 12.2))
    try:
        fig.canvas.manager.set_window_title("Sequential CP safety monitor")
    except Exception:  # noqa: BLE001 - Agg backend has no window manager
        pass
    gs = GridSpec(6, 1, figure=fig, height_ratios=[3.3, 0.30, 2.1, 1.7, 1.2, 0.28],
                  hspace=0.40, left=0.045, right=0.985, top=0.935, bottom=0.045)
    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.05)
    ax_views = [fig.add_subplot(top[i]) for i in range(3)]
    for a_ in ax_views:
        a_.axis("off")
    ax_map = fig.add_subplot(top[3])
    ax_banner = fig.add_subplot(gs[1]); ax_banner.axis("off")
    mid_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.05,
                                     width_ratios=[1.0, 1.15])
    ax_pl = fig.add_subplot(mid_gs[0]); ax_pl.axis("off")
    ax_pr = fig.add_subplot(mid_gs[1]); ax_pr.axis("off")
    ax_sig = fig.add_subplot(gs[3])
    ax_d = fig.add_subplot(gs[4], sharex=ax_sig)
    ax_log = fig.add_subplot(gs[5]); ax_log.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", ha="left", x=0.02)
    ax_sig.set_ylabel("s(t)  ensemble disagreement")
    ax_d.set_xlabel("timestep"); ax_d.set_ylabel("‖cube − goal‖ (m)")
    for a in (ax_sig, ax_d):
        a.grid(alpha=0.25, linewidth=0.5)
    return fig, ax_views, ax_map, ax_banner, ax_pl, ax_pr, ax_sig, ax_d, ax_log


def _draw_panel(axes, p, q_hat, stack):
    """Banner, then two columns. The panel is a short wide strip here, not a tall narrow
    one, so the rows are split across axes instead of stacked into a collision."""
    ax_b, ax_l, ax_r = axes
    escalated = p["mode"] == EXPERT or "escalate" in p["latched"]
    above = "above q_safe" in p["latched"]
    verdict = "ABOVE q_safe" if above else ("ESCALATED" if escalated else "SAFE")
    col = C_BAD if above else (C_EXPERT if escalated else C_SAFE)

    ax_b.clear(); ax_b.axis("off")
    ax_b.add_patch(matplotlib.patches.Rectangle(
        (0, 0.30), 1.0, 0.70, transform=ax_b.transAxes, facecolor=col, zorder=2))
    ax_b.text(0.5, 0.65, verdict, transform=ax_b.transAxes, ha="center", va="center",
              fontsize=17, fontweight="bold", color="white", zorder=3)
    sub = ("no threshold crossed so far this episode" if not p["latched"]
           else "LATCHED  " + ",  ".join(f"{k} @ step {v}" for k, v in p["latched"].items()))
    ax_b.text(0.005, 0.02, sub, transform=ax_b.transAxes, fontsize=9,
              color="#555", style="italic", va="bottom")
    live = ("not scored (below motion floor)" if not p["scorable"]
            else "IN BAND" if p["score"] <= p["q_esc"]
            else "ABOVE q_esc" if p["score"] <= p["q_hat"] else "ABOVE q_safe")
    ax_b.text(0.995, 0.02, f"RIGHT NOW: {live}", transform=ax_b.transAxes, fontsize=10,
              fontweight="bold", family="monospace", ha="right", va="bottom",
              color=C_SAFE if live == "IN BAND" else C_BAD)

    ax_l.clear(); ax_l.axis("off")
    ph = "reach" if p["phase"] == REACH else "push"
    rows = [
        ("phase", ph),
        ("policy", "CHEAP  MLP 755p" if p["mode"] != EXPERT else "EXPERT P-servo+TQC"),
        ("step", f"{p['step']} / 100"),
        ("s(t)", f"{p['score']:.3f}" + ("" if p["scorable"] else "  (below floor)")),
        ("q_esc", f"{p['q_esc']:.3f}"),
        ("q_safe", f"{p['q_hat']:.3f}"),
        ("", f"{p['score']:.3f} {'<' if p['score'] < p['q_esc'] else '>'} "
             f"{p['q_esc']:.3f} {'<' if p['score'] < p['q_hat'] else '>'} {p['q_hat']:.3f}"),
        ("dwell", f"{p['phase_steps']} / {PUSH_LIMIT} steps"),
        ("cube to goal", f"{p['d_push']:.3f} m   target < {SUCCESS_RADIUS:.3f}"),
    ]
    y = 0.96
    for k, v in rows:
        good = k == "cube to goal" and p["d_push"] < SUCCESS_RADIUS
        ax_l.text(0.01, y, k, transform=ax_l.transAxes, fontsize=10.5, color="#555",
                  va="top")
        ax_l.text(0.30, y, str(v)[:34], transform=ax_l.transAxes, fontsize=10,
                  family="monospace", va="top", color=C_SAFE if good else "#000",
                  fontweight="bold" if k in ("phase", "s(t)") or good else "normal")
        y -= 0.093
    ax_l.text(0.66, 0.96, "ALL THRESHOLDS", transform=ax_l.transAxes, fontsize=9,
              color="#555", fontweight="bold", va="top")
    yy = 0.86
    for name, key in (("reach", REACH), ("push", PUSH)):
        cur = key == p["phase"]
        ax_l.text(0.66, yy, f"{'> ' if cur else '  '}{name:9s} {q_hat[key]:.3f}",
                  transform=ax_l.transAxes, fontsize=9, family="monospace", va="top",
                  fontweight="bold" if cur else "normal", color="#000" if cur else "#777")
        yy -= 0.085

    ax_r.clear(); ax_r.axis("off")
    ax_r.text(0.01, 0.96, "TRACKING", transform=ax_r.transAxes, fontsize=10,
              color="#555", fontweight="bold", va="top")
    ax_r.text(0.20, 0.96, "bars 0-87mm", transform=ax_r.transAxes, fontsize=8,
              color="#999", va="top")
    for j, (lab, val, c) in enumerate((("commanded", p["cmd_norm"], "#1a237e"),
                                       ("actual", p["realised"], "#00897b"))):
        yy = 0.86 - 0.085 * j
        ax_r.text(0.01, yy, lab, transform=ax_r.transAxes, fontsize=9, color="#555",
                  va="top")
        ax_r.barh(yy - 0.03, 0.22 * min(val / BAR_FULL_SCALE, 1.0), height=0.045,
                  left=0.16, transform=ax_r.transAxes, color=c, zorder=3)
        ax_r.text(0.40, yy, f"{val * 1000:5.1f}mm", transform=ax_r.transAxes,
                  fontsize=9, family="monospace", va="top")
    frac = min(p["cmd_norm"] / BAR_FULL_SCALE, 1.0)
    filled = int(round(frac * 14))
    ax_r.text(0.01, 0.63, "MOTION", transform=ax_r.transAxes, fontsize=10,
              color="#555", fontweight="bold", va="top")
    ax_r.text(0.01, 0.53, "|" + "#" * filled + "." * (14 - filled) + "|",
              transform=ax_r.transAxes, fontsize=10, family="monospace", va="top")
    ax_r.text(0.30, 0.53, "MOVING" if p["scorable"] else "BELOW FLOOR",
              transform=ax_r.transAxes, fontsize=10, family="monospace",
              fontweight="bold", va="top",
              color=C_SAFE if p["scorable"] else "#9e9e9e")

    ax_r.text(0.52, 0.96, "ARBITRATION", transform=ax_r.transAxes, fontsize=10,
              color="#555", fontweight="bold", va="top")
    for j, (k, v) in enumerate((("policy", "EXPERT" if p["mode"] == EXPERT else "CHEAP"),
                                ("compute", "EXPENSIVE" if p["mode"] == EXPERT else "CHEAP"),
                                ("hand-back", f"{p['in_band']} / {HANDBACK_STEPS} in band"
                                              if p["mode"] == EXPERT else "n/a"),
                                ("switches", f"{p['switches']}  ({p['calls']} call"
                                             f"{'s' if p['calls'] != 1 else ''})"),
                                ("expert use", f"{p['expert_pct']:.1f}% of steps"))):
        yy = 0.86 - 0.085 * j
        ax_r.text(0.52, yy, k, transform=ax_r.transAxes, fontsize=9, color="#555",
                  va="top")
        ax_r.text(0.70, yy, str(v)[:24], transform=ax_r.transAxes, fontsize=9,
                  family="monospace", va="top",
                  color=C_EXPERT if (k == "compute" and p["mode"] == EXPERT) else "#000")

    ax_r.text(0.01, 0.30, "POLICY STACK", transform=ax_r.transAxes, fontsize=10,
              color="#555", fontweight="bold", va="top")
    yy = 0.19
    for e in stack:
        active = (e["name"] == "expert") == (p["mode"] == EXPERT)
        dot, colour, state = ("*", C_SAFE, "ACTIVE") if active else ("o", "#777", "STANDBY")
        ax_r.text(0.01, yy, f"{dot} {e['display']:17s} {e['klass']:9s} "
                            f"{e['success']:11s} {state}",
                  transform=ax_r.transAxes, fontsize=8.5, family="monospace", va="top",
                  color=colour, fontweight="bold" if active else "normal")
        yy -= 0.085


def _draw_map(ax, p):
    """Top-down world x-y. The 3-D view shows what the arm is doing; this shows what the
    CUBE is doing, which is the quantity the task is scored on."""
    ax.clear()
    h = p["hist"]
    cube = np.array([r["cube"] for r in h])
    ee = np.array([r["ee"] for r in h])
    g, rg = p["task_goal"], p["reach_goal"]
    inside = p["d_push"] < SUCCESS_RADIUS
    ax.add_patch(matplotlib.patches.Circle(
        (g[0], g[1]), SUCCESS_RADIUS, facecolor=(0.15, 0.8, 0.25, 0.25) if inside
        else (0.85, 0.15, 0.15, 0.18), edgecolor=C_SAFE if inside else C_BAD,
        lw=1.8, zorder=2))
    ax.plot(g[0], g[1], "+", color=C_SAFE if inside else C_BAD, ms=11, mew=2, zorder=3)
    if p["phase"] == REACH:
        ax.add_patch(matplotlib.patches.Circle((rg[0], rg[1]), SUCCESS_RADIUS,
                     facecolor=(0.2, 0.4, 1.0, 0.13), edgecolor="#3355cc",
                     lw=1.2, ls="--", zorder=2))
    ax.plot(ee[:, 0], ee[:, 1], "-", color="#b0b4ba", lw=1.2, zorder=3, label="gripper")
    # colour the cube trail by which policy was driving at each step
    for i in range(1, len(cube)):
        ax.plot(cube[i - 1:i + 1, 0], cube[i - 1:i + 1, 1], "-",
                color=C_EXPERT if h[i]["mode"] == EXPERT else "#555",
                lw=2.4, zorder=4, solid_capstyle="round")
    ax.plot(cube[0, 0], cube[0, 1], "o", color="#888", ms=6, zorder=5)
    ax.plot(cube[-1, 0], cube[-1, 1], "s", color="#0b8a2a", ms=11, zorder=6,
            markeredgecolor="white", markeredgewidth=1.4)
    moved = float(np.linalg.norm(cube[-1] - cube[0]))
    ax.set_title(f"CUBE PATH  grey=cheap orange=expert   moved {moved * 100:.1f} cm",
                 fontsize=8.5, loc="left", color="#888", pad=2)
    # zoom to what is actually happening, with room for the goal disc
    pts = np.vstack([cube[:, :2], ee[:, :2], g[None, :2]])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    half = max(float(np.abs(pts[:, 0] - cx).max()), float(np.abs(pts[:, 1] - cy).max()),
               SUCCESS_RADIUS) * 1.35 + 0.04
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x (m)", fontsize=9); ax.set_ylabel("y (m)", fontsize=9)


def _redraw(ax_sig, ax_d, p):
    h = p["hist"][-WINDOW:]
    xs = [r["step"] for r in h]
    x0, x1 = (xs[0] - 0.5, max(xs[-1], xs[0] + 8) + 0.5)

    for ax in (ax_sig, ax_d):
        ax.clear()
        run_start, run_mode = None, None
        for r in h + [None]:
            m = r["mode"] if r else None
            if run_mode is None:
                run_start, run_mode = (r["step"] if r else None), m
            elif m != run_mode:
                ax.axvspan(run_start - 0.5, (r["step"] if r else xs[-1] + 1) - 0.5,
                           color=BG_EXPERT if run_mode == EXPERT else BG_CHEAP,
                           zorder=0, lw=0)
                run_start, run_mode = (r["step"] if r else None), m
        for st, kind in p["events"]:
            if x0 <= st <= x1:
                ax.axvline(st, color=C_EXPERT if kind == "esc" else "#555",
                           lw=1.6, ls=":", zorder=5)
        ax.set_xlim(x0, x1); ax.grid(alpha=0.25, linewidth=0.5, zorder=1)
        # Plain tick labels. With the default formatter matplotlib factors out 1e-1 and
        # prints "0,1,2,3" with a small multiplier in the corner -- which reads as s(t)=3
        # against a q_safe of 0.197 if the corner is cropped or missed.
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    ys = [r["score"] for r in h]
    ax_sig.plot(xs, [r["q_hat"] for r in h], color=C_BAD, lw=1.8, ls="--",
                zorder=2, label=r"$\hat{q}_{safe}$ (per phase)")
    ax_sig.plot(xs, [r["q_esc"] for r in h], color=C_EXPERT, lw=1.8, ls=":",
                zorder=2, label=r"$\hat{q}_{esc}$")
    ax_sig.plot(xs, ys, color="#1a237e", lw=2.0, zorder=3, label="s(t)")
    ax_sig.scatter(xs, ys, s=14, color="#1a237e", zorder=4)
    for st, kind in p["events"]:
        if x0 <= st <= x1:
            ax_sig.annotate("^ ESCALATE" if kind == "esc" else "v CHEAP",
                            xy=(st, max(ys + [h[-1]["q_hat"]]) * 1.02), fontsize=9,
                            rotation=90, ha="right", va="top", zorder=6,
                            color=C_EXPERT if kind == "esc" else "#444", fontweight="bold")
    ax_sig.set_ylim(0, max(ys + [h[-1]["q_hat"]]) * 1.30 + 1e-3)
    ax_sig.set_ylabel("s(t)  ensemble disagreement")
    ax_sig.tick_params(labelbottom=False)
    ax_sig.legend(loc="upper left", ncol=3, fontsize=9, framealpha=0.9)

    dh = [r["d_push"] for r in h]
    ax_d.plot(xs, dh, color="#7a4fbf", lw=2.0, zorder=3)
    ax_d.scatter(xs, dh, s=14, color="#7a4fbf", zorder=4)
    ax_d.axhline(SUCCESS_RADIUS, color=C_BAD, lw=1.8, ls="--", zorder=2)
    ax_d.text(x1, SUCCESS_RADIUS, " success radius 0.050 m ", color=C_BAD, fontsize=9,
              va="bottom", ha="right", fontweight="bold", zorder=5)
    ax_d.set_ylim(0, max(max(dh), SUCCESS_RADIUS) * 1.20 + 1e-3)
    ax_d.set_xlabel("timestep"); ax_d.set_ylabel("‖cube − goal‖ (m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alpha-esc", type=float, default=0.30)
    ap.add_argument("--save", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--res", type=int, default=520)
    ap.add_argument("--seconds-per-step", type=float, default=1.0,
                    help="wall-clock seconds per control step when interactive")
    ap.add_argument("--hold", type=float, default=4.0,
                    help="seconds to hold the final frame of each episode")
    a = ap.parse_args()
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = json.loads((TRAIN / "calibration.json").read_text())
    ev = json.loads((TRAIN / "arbitration_eval.json").read_text())["arms"]
    q_esc = phase_q(cal, a.alpha_esc)
    t = cal["thresholds"]["b_ensemble_disagreement"]["K2"]
    q_hat = {REACH: t["reach"]["q_hat"], PUSH: t["push"]["q_hat"]}
    stack = [
        {"name": "expert", "display": "P-servo + TQC", "klass": "expensive",
         "success": f"{ev['expert_only']['composed_success']:.0%} n={ev['expert_only']['n']}"},
        {"name": "cheap", "display": "MLP 755 params", "klass": "cheap",
         "success": f"{ev['cheap_only']['composed_success']:.0%} n={ev['cheap_only']['n']}"},
    ]

    push_model = TQC.load(TRAIN / "push_expert" / "seed0" / "best_model.zip",
                          env=DummyVecEnv([PushOnly]), device="cpu")
    cheap, _ = load_cheap(TRAIN / "cheap_policy_16x16.pt")
    ens = [cheap] + [load_cheap(TRAIN / f"cheap_policy_ens{i}.pt")[0] for i in (1, 2)]
    env = ComposedEnv()
    arb = Arbiter(cheap, ens, ComposedExpert(push_model), q_esc)
    renderer = mujoco.Renderer(env.sim.m, a.res, a.res)
    cams = []
    for _nm, c in CAMS:
        cm = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.sim.m, cm)
        cm.azimuth = c["azimuth"] if c["azimuth"] is not None else 140.0
        cm.elevation, cm.distance = c["elevation"], c["distance"]
        cams.append(cm)
    gids = style_goals(env.sim.m)
    seeds = [a.seed] if a.seed else json.loads(
        (TRAIN / "demo_seeds.json").read_text())["demo_seeds"]

    (fig, ax_views, ax_map, ax_banner, ax_pl, ax_pr,
     ax_sig, ax_d, ax_log) = _build_figure("")
    state = {"ims": None}

    def draw(p):
        ph = "REACH" if p["phase"] == REACH else "PUSH"
        fig.suptitle(f"[LIVE] seed {p['seed']} — MuJoCo Reach→Push — "
                     f"{'EXPERT' if p['mode'] == EXPERT else 'CHEAP'}\n{ph}",
                     fontsize=14, fontweight="bold", ha="left", x=0.02)
        if state["ims"] is None:
            state["ims"] = [ax.imshow(im) for ax, im in zip(ax_views, p["imgs"])]
            for ax, (nm, _c) in zip(ax_views, CAMS):
                ax.axis("off")
                ax.set_title(nm, fontsize=8.5, color="#888", loc="left", pad=2)
        else:
            for im_artist, im in zip(state["ims"], p["imgs"]):
                im_artist.set_data(im)
        for tx in list(ax_views[0].texts):
            tx.remove()
        for t_ in list(fig.texts):
            if getattr(t_, "_verdict", False):
                t_.remove()
        if p["done"] is not None:
            tv = fig.text(0.985, 0.962, " SUCCESS " if p["done"] else " FAILURE ",
                          fontsize=17, fontweight="bold", color="white",
                          ha="right", va="center", zorder=12,
                          bbox=dict(boxstyle="round,pad=0.35",
                                    fc=C_SAFE if p["done"] else C_BAD, ec="none"))
            tv._verdict = True
        _draw_map(ax_map, p)
        _draw_panel((ax_banner, ax_pl, ax_pr), p, q_hat, stack)
        _redraw(ax_sig, ax_d, p)
        ax_log.clear(); ax_log.axis("off")
        ax_log.text(0.5, 0.5, "SPACE pause/resume     -> step once     "
                              "R reset     Q quit",
                    transform=ax_log.transAxes, ha="center", va="center",
                    fontsize=10, family="monospace", color="#777")
        return ()

    def make_stream(sds):
        return stream(sds, env, arb, renderer, cams, q_hat, gids)

    if a.save:
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, draw, frames=make_stream(seeds), repeat=False,
                             cache_frame_data=False, save_count=10**5)
        anim.save(a.save, writer="ffmpeg", fps=a.fps, dpi=100)
        print(f"wrote {a.save}")
        return

    # ---------------- interactive
    ctl = {"paused": False, "step": False, "quit": False, "reset": False, "next": False}

    def on_key(e):
        k = (e.key or "").lower()
        if k == " ":
            ctl["paused"] = not ctl["paused"]
        elif k == "right":
            ctl["step"] = True
        elif k == "q":
            ctl["quit"] = True
        elif k == "r":
            ctl["reset"] = True
        elif k == "n":
            ctl["next"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.ion()
    fig.show()
    hold_pause = max(a.hold / HOLD_FRAMES, 0.01)
    print("Sequential CP safety monitor -- closed loop, no replay")
    print(f"  seeds {seeds}   alpha_esc {a.alpha_esc}  K=2")
    print(f"  pace {a.seconds_per_step:.2f} s per control step, {a.hold:.0f} s hold\n"
          "  keys: SPACE pause   RIGHT step   R reset   N next   Q quit\n", flush=True)

    i = 0
    while i < len(seeds) and not ctl["quit"]:
        restart = False
        print(f"--- seed {seeds[i]}", flush=True)
        for pkt in make_stream([seeds[i]]):
            while ctl["paused"] and not any(ctl[k] for k in ("step", "quit", "reset", "next")):
                plt.pause(0.05)
            if ctl["quit"]:
                break
            if ctl["reset"]:
                ctl["reset"] = False
                restart = True
                break
            if ctl["next"]:
                ctl["next"] = False
                break
            ctl["step"] = False
            draw(pkt)
            fig.canvas.draw_idle()
            plt.pause(hold_pause if pkt["done"] is not None else a.seconds_per_step)
        if not restart:
            i += 1
    plt.ioff()
    print("done.")


if __name__ == "__main__":
    main()
