"""Build the handout's data/ folder: meta.json, episode_index.csv, showcase_episodes.csv, results copies.

Run from the project venv (needs the composed/results_push artefacts).
"""
import csv
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.join(HERE, "..", "composed", "results_push")
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

B = json.load(open(os.path.join(RP, "part45_benchmark.json")))
EFF = json.load(open(os.path.join(RP, "efficiency_summary.json")))
z = np.load(os.path.join(RP, "replay_trajectories_a0.30.npz"))
CONDS = ("cheap", "expert", "arbitrated_a0.30", "random_a0.30")
seeds = sorted({int(k.split("/")[1][4:]) for k in z.files if k.startswith("cheap/")})

meta = {
    "what": "Recorded closed-loop episodes of the composed Reach->Push task (panda-gym 3.0.7 / PyBullet 3.2.7), "
            "4 conditions x 500 paired seeds, alpha_esc = 0.30.  For KINEMATIC replay only.",
    "control_dt_s": 0.04, "control_hz": 25,
    "frames": "qpos/obj_pos/obj_quat/ee have T+1 rows: row 0 = state after reset, row t = state after action t-1. "
              "modes/labels/score/flags/actions have T rows: entry t describes action t (taken from row t to row t+1).",
    "world_frame": "panda-gym world: z up, table top at z = 0, robot base at (-0.6, 0, 0), units metres",
    "robot": {"model": "Franka Emika Panda (panda-gym uses pybullet_data franka_panda/panda.urdf)",
              "qpos_order": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
                             "finger_joint1", "finger_joint2"],
              "qpos_units": "rad for joints 1-7, metres for finger slides (gripper is blocked, ~0-0.01 m)",
              "neutral_qpos": [0.0, 0.41, 0.0, -1.85, 0.0, 2.26, 0.79, 0.0, 0.0],
              "ee": "panda_grasptarget (URDF link 11) = hand + 0.105 m along hand z",
              "mujoco_menagerie_fk_check": None},
    "table": {"size_xyz_m": [1.1, 0.7, 0.4], "centre": [-0.3, 0.0, -0.2], "top_z": 0.0},
    "floor_z": -0.4,
    "cube": {"edge_m": 0.04, "rest_centre_z": 0.02,
             "obj_quat_convention": "PyBullet (x, y, z, w) -- MuJoCo free joints use (w, x, y, z)"},
    "goals": {"reach_goal": "3-D point; phase-A success when ||ee - reach_goal|| < 0.05",
              "task_goal": "Push goal (cube centre); phase-B success when ||cube - task_goal|| < 0.05"},
    "arrays": {"modes": "0 = cheap policy acted, 1 = expert acted", "labels": "0 = reach phase, 1 = push phase",
               "score": "compliance residual s(t) (NaN/inf possible when commanded motion ~0)",
               "flags": "arbiter flag s(t) > q_esc[phase] (arbitrated condition only; random condition = random coin)",
               "actions": "end-effector displacement command in [-1, 1]^3 (x 0.05 m per step)",
               "success": "episode success (both phases)"},
    "arbiter": {"alpha": 0.10, "alpha_esc": 0.30, "hold_steps": 8,
                "q_hat_alpha0.10": {p: v["value"] for p, v in B["part4"]["q_hat"].items()},
                "q_esc_alpha0.30": B["part4"]["escalation"]["0.3"]["q_esc"]},
    "gate_override_disclosure": B["gate_override"],
}

# Menagerie FK check result is filled in by mujoco/verify_fk.py
fk = os.path.join(DATA, "fk_check.json")
if os.path.exists(fk):
    meta["robot"]["mujoco_menagerie_fk_check"] = json.load(open(fk))

rows = []
for s in seeds:
    for c in CONDS:
        p = "%s/seed%d/" % (c, s)
        modes = z[p + "modes"]
        labels = z[p + "labels"]
        sw = np.flatnonzero(np.diff(labels) != 0)
        rows.append({"seed": s, "condition": c, "success": int(bool(z[p + "success"])), "steps": len(modes),
                     "expert_steps": int(modes.sum()), "frac_expert": round(float(modes.mean()), 4),
                     "escalations": int(np.count_nonzero(np.diff(np.r_[0, modes]) > 0)),
                     "first_escalation_step": int(np.flatnonzero(modes)[0]) if modes.any() else -1,
                     "reach_to_push_step": int(sw[0] + 1) if sw.size else -1})
with open(os.path.join(DATA, "episode_index.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# showcase: same seed where cheap fails, expert succeeds, arbitrated escalates and succeeds (switching visible)
by = {(r["seed"], r["condition"]): r for r in rows}
cand = [s for s in seeds if not by[(s, "cheap")]["success"] and by[(s, "expert")]["success"]
        and by[(s, "arbitrated_a0.30")]["success"] and 1 <= by[(s, "arbitrated_a0.30")]["escalations"]
        and 0.1 <= by[(s, "arbitrated_a0.30")]["frac_expert"] <= 0.6]
cand.sort(key=lambda s: -by[(s, "arbitrated_a0.30")]["steps"])       # longer episodes show more
show = [{"seed": s, "story": "cheap fails / arbitrated escalates and succeeds / expert succeeds",
         **{"arb_" + k: by[(s, "arbitrated_a0.30")][k] for k in ("steps", "expert_steps", "escalations", "first_escalation_step")}}
        for s in cand[:3]]
fail = [s for s in seeds if not by[(s, "arbitrated_a0.30")]["success"] and by[(s, "arbitrated_a0.30")]["escalations"] >= 1]
if fail:
    s = fail[0]
    show.append({"seed": s, "story": "honest failure case: arbitrated escalates but still fails",
                 **{"arb_" + k: by[(s, "arbitrated_a0.30")][k] for k in ("steps", "expert_steps", "escalations", "first_escalation_step")}})
with open(os.path.join(DATA, "showcase_episodes.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(show[0]))
    w.writeheader()
    w.writerows(show)

json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=2)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
for f in ("part45_benchmark.json", "efficiency_summary.json", "part3_horizon_rescue.json", "REPORT_reach_push.md"):
    shutil.copy(os.path.join(RP, f), os.path.join(HERE, "results", f))
shutil.copy(os.path.join(RP, "figures", "rescue_curve.png"), os.path.join(HERE, "results", "rescue_curve.png"))
print("episodes indexed:", len(rows), "showcase:", [r["seed"] for r in show])
