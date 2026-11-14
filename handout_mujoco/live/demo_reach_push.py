"""Live MuJoCo policy arbitration. Closed loop, real models, nothing replayed.

Every observation comes from MuJoCo physics at the current step; every action comes from
a network or the P-servo; the arbiter decides in real time. There is no recorded
trajectory anywhere in this file.

WHAT TO WATCH: the score bar rises, crosses the ORANGE line (q_esc) and the lamp turns
orange as control hands to the expert -- and the bar does not reach the RED line (q_hat).
That gap is the leading-indicator property: q_esc < q_hat by construction, so escalation
fires strictly before the safety threshold would be crossed.

The bar, both threshold lines and the policy lamp are drawn IN THE SCENE, because the
passive viewer has no text overlay. Step-by-step state is printed to the terminal and
logged to JSONL.

Keys:  SPACE pause/resume   RIGHT single-step   R reset   N next seed   Q quit

macOS: the passive viewer allows exactly ONE launch per process, so the viewer is opened
once and every seed is run inside it. Relaunching per seed raises from
`_MJPYTHON.launch_on_ui_thread`.

    cd handout_mujoco
    ./run_policy.sh                 # all three rescue seeds
    ./run_policy.sh 400169          # one seed
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import mujoco.viewer
from cheap_policy import load as load_cheap
from composed import PUSH, REACH, ComposedEnv, ComposedExpert
from envs import PushOnly
from mujoco_arbiter import EXPERT, Arbiter, phase_q
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

import mujoco

TRAIN = ROOT / "results" / "mujoco_train"
LOG = TRAIN / "demo_log.jsonl"
# The in-scene score gauge was REMOVED. mjvScene primitives cannot render text, axes or
# numbers, so no labelled threshold plot is achievable there, and the decorative geoms
# that were tried rendered as large planes cutting through the arm. The readable visual
# lives in live/arena_reach_push.py (offscreen render + matplotlib composite). This file
# is the headless / logging path and the plain viewer.
LAMP_PARK = np.array([0.0, 0.0, -5.0])   # non-collidable (contype=0); verified inert


def slow_step(env, a, v, per_step, alive):
    """Advance one control step, rendering BETWEEN physics substeps.

    Syncing only once per control step teleports 0.04 s of motion at a time, which reads
    as a slideshow. Rendering inside the substep loop is what makes it look like a robot.
    """
    if v is None:
        return env.step(a)
    delay = max(per_step, 0.0) / max(env.sim.substeps, 1)

    def on_substep():
        if alive[0]:
            v.sync()
            if delay > 0:
                time.sleep(delay)

    return env.step(a, on_substep=on_substep)


def run_seed(seed, env, arb, v, fh, per_step, hold, keys):
    """One episode inside an ALREADY-OPEN viewer."""
    env.reset(seed)
    arb.reset_stats()
    q_hat = arb.q_hat
    done, info = None, None

    while (v is None or v.is_running()) and not keys["quit"]:
        if keys["reset"]:
            env.reset(seed); arb.reset_stats(); done = None; keys["reset"] = False
        if keys["next"]:
            keys["next"] = False
            return done, arb.stats(), info
        if done is None and (not keys["paused"] or keys["step"]):
            keys["step"] = False
            prev_mode = arb.mode
            a, tag = arb.act(env)
            tr = arb.trace[-1]
            _, term, trunc, info = env.step(a, on_substep=(
                (lambda: (v.sync(), time.sleep(max(per_step, 0.0) / env.sim.substeps)))
                if v is not None else None))
            if arb.mode != prev_mode:
                ev = "ESCALATE -> EXPERT" if arb.mode == EXPERT else "hand back -> CHEAP"
                print(f"    ** {ev} at step {tr['step']} "
                      f"(s={tr['score']:.4f} vs q_esc={tr['q_esc']:.4f})")
            pct = 100 * arb.expert_steps / max(arb.n_steps, 1)
            rec = {"seed": seed, "step": tr["step"], "phase": tr["phase"],
                   "phase_name": "reach" if tr["phase"] == REACH else "push",
                   "score": tr["score"], "q_esc": tr["q_esc"],
                   "q_hat": q_hat[tr["phase"]],
                   "policy": "EXPERT" if tr["mode"] == EXPERT else "CHEAP",
                   "tag": tag, "expert_pct": pct, "d_push": info["d_push"],
                   "success": info["is_success"]}
            if fh:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"  [{tr['step']:>3}] {rec['phase_name']:<5} {rec['policy']:<6} "
                  f"s={tr['score']:.4f} q_esc={tr['q_esc']:.4f} "
                  f"q_hat={q_hat[tr['phase']]:.4f} exp={pct:4.0f}%  "
                  f"d_push={info['d_push']:.3f}")
            if term or trunc:
                done = bool(info["is_success"])
                print(f"    == {'SUCCESS' if done else 'FAILURE'} in {tr['step']} steps, "
                      f"expert on {pct:.0f}% of steps, {arb.expert_calls} call(s)")
        if v is not None:
            env.sim.d.mocap_pos[env.sim.mc_lamp] = LAMP_PARK
            v.sync()
            time.sleep(0.02)
            if done is not None:
                t_end = time.perf_counter() + hold
                while (time.perf_counter() < t_end and v.is_running()
                       and not keys["quit"] and not keys["reset"] and not keys["next"]):
                    v.sync(); time.sleep(0.03)
                if not keys["reset"]:
                    break
        elif done is not None:
            break
    return done, arb.stats(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alpha-esc", type=float, default=0.30)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--seconds-per-step", type=float, default=1.5,
                    help="wall-clock seconds per 0.04 s control step (slow motion factor)")
    ap.add_argument("--hold", type=float, default=5.0,
                    help="seconds to hold the final frame before the next seed")
    a = ap.parse_args()

    cal = json.loads((TRAIN / "calibration.json").read_text())
    seeds = [a.seed] if a.seed else json.loads(
        (TRAIN / "demo_seeds.json").read_text())["demo_seeds"]
    q_esc = phase_q(cal, a.alpha_esc)
    t = cal["thresholds"]["b_ensemble_disagreement"]["K2"]
    q_hat = {REACH: t["reach"]["q_hat"], PUSH: t["push"]["q_hat"]}

    push_model = TQC.load(TRAIN / "push_expert" / "seed0" / "best_model.zip",
                          env=DummyVecEnv([PushOnly]), device="cpu")
    expert = ComposedExpert(push_model)
    cheap, _ = load_cheap(TRAIN / "cheap_policy_16x16.pt")
    ens = [cheap] + [load_cheap(TRAIN / f"cheap_policy_ens{i}.pt")[0] for i in (1, 2)]
    env = ComposedEnv()
    arb = Arbiter(cheap, ens, expert, q_esc)
    arb.q_hat = q_hat

    print("LIVE MuJoCo POLICY ARBITRATION -- closed loop, no replay")
    print("  reach stage : scripted P-servo (NOT a learned reach expert)")
    print("  push expert : TQC 300k, 0.930 on 200 held-out seeds (single seed)")
    print(f"  cheap arm   : {cheap.n_params():,}-parameter MLP")
    print(f"  alpha_esc {a.alpha_esc}  K=2   q_esc reach {q_esc[REACH]:.4f} "
          f"push {q_esc[PUSH]:.4f}")
    print(f"  q_hat       reach {q_hat[REACH]:.4f} push {q_hat[PUSH]:.4f}   (q_esc < q_hat)")
    print(f"  pace        {a.seconds_per_step:.2f} s per control step, "
          f"{a.hold:.0f} s hold at the end of each episode")
    print("  keys: SPACE pause  RIGHT step  R reset  N next  Q quit\n")

    keys = {"paused": False, "step": False, "quit": False, "reset": False, "next": False}

    def on_key(k):
        if k == 32: keys["paused"] = not keys["paused"]
        elif k == 262: keys["step"] = True
        elif k in (81, 113): keys["quit"] = True
        elif k in (82, 114): keys["reset"] = True
        elif k in (78, 110): keys["next"] = True

    # ONE viewer for every seed: mjpython permits a single launch per process.
    v = None if a.no_viewer else mujoco.viewer.launch_passive(
        env.sim.m, env.sim.d, key_callback=on_key,
        show_left_ui=False, show_right_ui=False)
    try:
        with open(LOG, "w") as fh:
            for sd in seeds:
                if keys["quit"]:
                    break
                if v is not None and not v.is_running():
                    # the window was closed: stop, rather than silently running
                    # 0-step episodes for every remaining seed
                    print("viewer closed -- stopping.")
                    break
                print(f"--- seed {sd}")
                _, st, _ = run_seed(sd, env, arb, v, fh, a.seconds_per_step,
                                    a.hold, keys)
                if st["steps"] == 0 and v is not None and not v.is_running():
                    print("viewer closed -- stopping.")
                    break
                print(f"    stats: {st}\n")
    finally:
        if v is not None:
            v.close()
    print(f"logged to {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
