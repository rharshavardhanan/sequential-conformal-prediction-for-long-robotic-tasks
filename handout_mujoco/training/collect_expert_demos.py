"""Collect (obs_26, action_3, phase) from the composed expert, SUCCESSFUL episodes only.

Successful only, because the cheap policy is a distillation target: cloning the expert's
failures teaches the cheap arm to reproduce them, and the whole point of the arbitration
experiment is that the cheap arm fails in ways the expert does not.

Train seeds 0-99,999 only. Calibration (300,000+) and evaluation (400,000+) are never
touched here.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python training/collect_expert_demos.py
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))

from composed import PUSH, REACH, ComposedEnv, ComposedExpert, rollout
from envs import PushOnly
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

CKPT = ROOT / "results" / "mujoco_train" / "push_expert" / "seed0" / "best_model.zip"
OUT = ROOT / "results" / "mujoco_train" / "expert_demos.npz"
TRAIN_LO, TRAIN_HI = 0, 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50_000, help="transitions to collect")
    ap.add_argument("--max-episodes", type=int, default=20_000)
    a = ap.parse_args()

    model = TQC.load(CKPT, env=DummyVecEnv([PushOnly]), device="cpu")
    env = ComposedEnv()
    expert = ComposedExpert(model)

    obs, act, phase, seeds = [], [], [], []
    n_ep = n_ok = 0
    t0 = time.perf_counter()
    for i in range(a.max_episodes):
        if len(obs) >= a.target:
            break
        seed = TRAIN_LO + (i % (TRAIN_HI - TRAIN_LO))
        rec = []
        info, _ = rollout(env, expert.act, seed, record=rec)
        n_ep += 1
        if not info["is_success"]:
            continue  # distil successes only
        n_ok += 1
        seeds.append(seed)
        for o, ac, ph, _tag in rec:
            obs.append(o)
            act.append(ac)
            phase.append(ph)

    obs = np.asarray(obs, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.int8)
    dt = time.perf_counter() - t0

    n_reach = int((phase == REACH).sum())
    n_push = int((phase == PUSH).sum())
    np.savez_compressed(OUT, obs=obs, act=act, phase=phase, seeds=np.asarray(seeds))
    meta = {
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "expert": ComposedExpert.label,
        "reach_stage": "scripted P-servo (NOT a learned reach expert)",
        "episodes_run": n_ep,
        "episodes_kept_successful": n_ok,
        "episode_success_rate_on_train_seeds": n_ok / max(n_ep, 1),
        "transitions": len(obs),
        "transitions_reach": n_reach,
        "transitions_push": n_push,
        "reach_fraction": n_reach / max(len(obs), 1),
        "train_seed_range": [TRAIN_LO, TRAIN_HI],
        "seconds": dt,
        "obs_dim": int(obs.shape[1]),
        "act_dim": int(act.shape[1]),
    }
    (OUT.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nwritten: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
