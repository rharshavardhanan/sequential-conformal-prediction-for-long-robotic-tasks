"""Kinematic replay of a recorded panda-gym Reach->Push episode in MuJoCo.

This is an ANIMATION of what happened in the PyBullet closed-loop run: joint angles and cube
pose are written frame by frame and MuJoCo only computes kinematics + renders.  No MuJoCo
physics, no policy is executed here.

Examples
  python replay_episode.py --condition arbitrated_a0.30 --seed 230007 --out arb_230007.mp4
  python replay_episode.py --condition arbitrated_a0.30 --seed 230007 --viewer      # interactive
  python replay_episode.py --list-showcase

Conditions in the data file: cheap, expert, arbitrated_a0.30, random_a0.30 (500 seeds each: 230000..230499).
"""
import argparse
import csv
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SCENE = os.path.join(ROOT, "third_party", "mujoco_menagerie", "franka_emika_panda", "scene_pushdemo.xml")
DATA = os.path.join(ROOT, "data", "replay_trajectories_a0.30.npz")
DT = 0.04                                   # panda-gym control step: 20 substeps x 1/500 s
ARM = ["joint%d" % i for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
LAMP = {0: (0.6, 0.6, 0.6, 1.0), 1: (1.0, 0.55, 0.0, 1.0)}       # cheap grey / expert orange


def load_episode(z, condition, seed):
    p = "%s/seed%d/" % (condition, seed)
    if p + "qpos" not in z.files:
        raise KeyError("no episode %s" % p)
    ep = {k: z[p + k] for k in ("qpos", "obj_pos", "obj_quat", "ee", "reach_goal", "task_goal",
                                "modes", "labels", "score", "flags", "success")}
    return ep


def pose_frame(m, d, ep, t):
    adr = [m.joint(n).qposadr[0] for n in ARM]
    q = ep["qpos"][t].astype(float).copy()
    q[7:] = np.clip(q[7:], 0.0, 0.04)                       # Menagerie finger range
    d.qpos[adr] = q
    ca = m.joint("cube_free").qposadr[0]
    x, y, zq, w = ep["obj_quat"][t]                          # PyBullet quaternion is (x, y, z, w)
    d.qpos[ca:ca + 7] = np.r_[ep["obj_pos"][t], w, x, y, zq]  # MuJoCo free joint wants (pos, w, x, y, z)
    d.mocap_pos[m.body("push_goal").mocapid[0]] = ep["task_goal"]
    d.mocap_pos[m.body("reach_goal").mocapid[0]] = ep["reach_goal"]
    # frame t is the state BEFORE action t; the policy that produced the motion into frame t is modes[t-1]
    mode = int(ep["modes"][min(max(t - 1, 0), len(ep["modes"]) - 1)])
    m.geom("mode_lamp").rgba[:] = LAMP[mode]
    phase = int(ep["labels"][min(max(t - 1, 0), len(ep["labels"]) - 1)])
    m.geom_rgba[m.body("reach_goal").geomadr[0]][3] = 0.25 if phase == 0 else 0.05
    mujoco.mj_forward(m, d)
    return mode, phase


def overlay(img, lines, color_line=None):
    """Burn a text banner into the frame (Pillow)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    dr.rectangle([0, 0, im.width, 14 + 34 * len(lines)], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        dr.text((18, 8 + 34 * i), ln, fill=(color_line if (color_line and i == 1) else (235, 235, 235)), font=font)
    return np.asarray(im)


def render(condition, seed, out, width=1280, height=720, hold_end=1.0, text=True):
    import imageio.v2 as imageio
    import json
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    ep = load_episode(np.load(DATA), condition, seed)
    meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
    q_esc = meta["arbiter"]["q_esc_alpha0.30"]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (-0.25, 0.0, 0.15)
    cam.distance, cam.azimuth, cam.elevation = 1.7, 135, -25
    frames = []
    T = len(ep["qpos"])
    with mujoco.Renderer(m, height, width) as r:
        for t in range(T):
            mode, phase = pose_frame(m, d, ep, t)
            r.update_scene(d, camera=cam)
            img = r.render()
            if text:
                a = max(t - 1, 0)
                s = float(ep["score"][a]) if t > 0 else float("nan")
                ph = ("reach", "push")[phase]
                qq = q_esc[ph]
                flag = " FLAG" if t > 0 and bool(ep["flags"][a]) else ""
                done = ("   SUCCESS" if bool(ep["success"]) else "   FAILED") if t == T - 1 else ""
                lines = ["%s   seed %d   t = %2d / %d   (%.2f s)%s" % (condition, seed, t, T - 1, t * DT, done),
                         "policy: %s" % ("EXPERT (escalated)" if mode else "cheap"),
                         "phase: %s    s(t) = %s   q_esc[%s] = %.2f%s" % (
                             ph, "  -  " if not np.isfinite(s) else "%.2f" % s, ph, qq, flag)]
                img = overlay(img, lines, color_line=(255, 150, 0) if mode else (190, 190, 190))
            frames.append(img)
    frames += [frames[-1]] * int(hold_end / DT)
    imageio.mimsave(out, frames, fps=round(1 / DT), macro_block_size=1)
    return ep


def viewer(condition, seed):
    import time
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    ep = load_episode(np.load(DATA), condition, seed)
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            for t in range(len(ep["qpos"])):
                pose_frame(m, d, ep, t)
                v.sync()
                time.sleep(DT)
            time.sleep(1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="arbitrated_a0.30")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--list-showcase", action="store_true")
    a = ap.parse_args()
    if a.list_showcase:
        for row in csv.DictReader(open(os.path.join(ROOT, "data", "showcase_episodes.csv"))):
            print(row)
    elif a.viewer:
        viewer(a.condition, a.seed)
    else:
        ep = render(a.condition, a.seed, a.out or "%s_%d.mp4" % (a.condition, a.seed))
        print("rendered", a.out, "success" if bool(ep["success"]) else "failure", "expert steps", int(ep["modes"].sum()))
