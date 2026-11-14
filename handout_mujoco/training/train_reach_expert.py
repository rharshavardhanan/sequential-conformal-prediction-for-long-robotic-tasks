"""Train the ReachOnly expert: TQC, no HER, on the live MuJoCo physics.

No HER here, deliberately: ReachOnly's reward is dense (-||ee - reach_goal||), so there
is no sparse-reward credit problem for hindsight relabeling to solve, and a goal-
conditioned Dict space would buy nothing.

Phase 1 measured that a plain P-servo solves this task 100% of the time in a median of 4
steps against the 50-step budget. If this lands below Gate 3's 0.95, the problem is the
training pipeline, not the task.

Shares SeededEval / summarise / param accounting with train_push_expert so the two
experts are evaluated by identical code on identical seeds.

    python training/train_reach_expert.py --steps 300000 --seed 0
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
sys.path.insert(0, str(HERE))

import torch
from envs import REACH_LIMIT, ReachOnly
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from train_push_expert import (
    EVAL_SEED_BASE,
    TRAIN_SEED_HI,
    TRAIN_SEED_LO,
    SeededEval,
    TrainSeeds,
    actor_params,
    summarise,
    total_params,
)

OUT_ROOT = ROOT / "results" / "mujoco_train" / "reach_expert"

# Identical to the push expert except net_arch (the task is far simpler) and the absence
# of HER. Keeping lr/gamma/tau/buffer/learning_starts aligned means a difference between
# the two experts is a difference in the task, not in the optimiser.
HYPERPARAMS = dict(
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=2048,
    gamma=0.95,
    tau=0.05,
    learning_starts=1_000,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def make_env(seed):
    return DummyVecEnv([lambda: Monitor(TrainSeeds(ReachOnly(), seed))])


def evaluate(model, n_episodes, seed_base=EVAL_SEED_BASE, deterministic=True):
    """Deterministic rollout on explicit held-out seeds. Same seeds as the push expert."""
    env = ReachOnly()
    rows, t_inf = [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        term = trunc = False
        steps = 0
        while not (term or trunc):
            t0 = time.perf_counter()
            a, _ = model.predict(obs, deterministic=deterministic)
            t_inf.append(time.perf_counter() - t0)
            obs, _, term, trunc, info = env.step(a)
            steps += 1
        rows.append(
            {
                "seed": seed_base + i,
                "success": bool(info["is_success"]),
                "steps": steps,
                "d_reach": float(info["d_reach"]),
                "cube_fell": False,
            }
        )
    env.close()
    return rows, (float(np.median(t_inf)) if t_inf else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    out_dir = OUT_ROOT / f"seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ReachOnly expert -- TQC, no HER  |  seed{a.seed}  |  {a.steps:,} steps  "
          f"|  device {a.device}")
    env = make_env(a.seed)
    model = TQC("MlpPolicy", env, device=a.device, seed=a.seed, verbose=0, **HYPERPARAMS)
    print(f"  actor parameters: {actor_params(model):,}  "
          f"(total saved policy incl. critics: {total_params(model):,})")
    print(f"  train_freq {model.train_freq}  gradient_steps {model.gradient_steps}  "
          f"learning_starts {HYPERPARAMS['learning_starts']:,}", flush=True)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "algo": "TQC",
                "replay_buffer": "ReplayBuffer (no HER -- dense reward)",
                "hyperparams": {k: (dict(v) if isinstance(v, dict) else v)
                                for k, v in HYPERPARAMS.items()},
                "train_freq": str(model.train_freq),
                "gradient_steps": model.gradient_steps,
                "top_quantiles_to_drop_per_net": model.top_quantiles_to_drop_per_net,
                "device": str(model.device),
                "seed": a.seed,
                "use_vecnormalize": False,
                "env": "live/envs.py:ReachOnly",
                "reach_limit": REACH_LIMIT,
                "train_seed_range": [TRAIN_SEED_LO, TRAIN_SEED_HI],
                "eval_seed_base": EVAL_SEED_BASE,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )

    cb = SeededEval(out_dir, a.eval_freq, a.n_eval, HYPERPARAMS["learning_starts"],
                    eval_fn=evaluate)
    t0 = time.perf_counter()
    model.learn(total_timesteps=a.steps, callback=cb, log_interval=4, progress_bar=False)
    wall = time.perf_counter() - t0
    model.save(out_dir / "final_model.zip")

    print(f"\n  wall-clock {wall / 60:.1f} min  |  overall {a.steps / wall:.1f} steps/s")
    post = [h for h in cb.history if h["after_learning_starts"]]
    if post:
        f = np.array([h["fps_interval"] for h in post])
        print(f"  steady-state fps AFTER the buffer fills: median {np.median(f):.1f} "
              f"(min {f.min():.1f}, max {f.max():.1f}) over {len(f)} intervals "
              f"-- MEASURED, not extrapolated")

    rows, inf_s = evaluate(model, 200)
    final = summarise(rows, inf_s, actor_params(model), f"seed{a.seed}_final",
                      total_params(model))
    final.update(total_timesteps=a.steps, wall_clock_s=wall, overall_fps=a.steps / wall,
                 steady_fps_median=(float(np.median([h["fps_interval"] for h in post]))
                                    if post else None))
    (out_dir / "reach_eval.json").write_text(json.dumps(final, indent=2))
    print(f"\n  FINAL eval (200 held-out seeds): success {final['success']:.3f} "
          f"CI95 [{final['success_ci95'][0]:.3f}, {final['success_ci95'][1]:.3f}]  "
          f"ep_len {final['mean_episode_len']:.1f}")
    print(f"  artifacts -> {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
