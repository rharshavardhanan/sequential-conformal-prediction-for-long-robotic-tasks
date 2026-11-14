"""Train a TQC expert on the MuJoCo Reach->Push task.

MuJoCo-native: no PyBullet checkpoint, threshold or success number is involved. The
environment is `live/mujoco_reach_push_env.MuJoCoReachPush` exactly as smoke-tested
(21/21) -- same scene, same tuned controller, same physics the live demo will run.

Seed discipline (live/MUJOCO_TRAINING_SPEC.md section 7), disjoint by construction:
    train        0 -  99,999
    calibration  300,000 - 300,999      (Phase 6, untouched here)
    evaluation   400,000 - 400,999

Evaluation is deterministic and uses *explicit* seeds, so the expert, the cheap policy
and the arbiter are later measured on the identical episodes and can be compared paired.

    python training/train_tqc_expert.py --sanity            # 5k steps, verifies plumbing
    python training/train_tqc_expert.py --steps 400000 --seed 0
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))

from mujoco_reach_push_env import (
    MuJoCoReachPush,
    PUSH,
    REACH,  # noqa: E402
    REACH_LIMIT,
    PUSH_LIMIT,
)

TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 400_000
OUT = ROOT / "results" / "mujoco_train" / "expert"


def make_env(seed_offset=0, seed=None):
    env = MuJoCoReachPush(seed_offset=seed_offset)
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate(
    model,
    n_episodes=200,
    seed_base=EVAL_SEED_BASE,
    vecnorm=None,
    deterministic=True,
    verbose=False,
):
    """Deterministic rollout on explicit held-out seeds. Returns per-episode records."""
    env = MuJoCoReachPush()
    rows = []
    t_inf = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_base + i)
        reach_done = False
        steps = 0
        term = trunc = False
        while not (term or trunc):
            o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            t0 = time.perf_counter()
            a, _ = model.predict(o, deterministic=deterministic)
            t_inf.append(time.perf_counter() - t0)
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            reach_done = reach_done or info["reach_done"]
        rows.append(
            {
                "seed": seed_base + i,
                "reach_success": bool(reach_done),
                "push_success": bool(info["is_success"]),
                "success": bool(info["is_success"]),
                "steps": steps,
                "final_phase": info["phase_name"],
                "d_push": float(info.get("d_push", np.nan)),
                "failure_phase": (
                    None
                    if info["is_success"]
                    else ("reach" if not reach_done else "push")
                ),
            }
        )
        if verbose and (i + 1) % 50 == 0:
            print(
                f"    {i + 1}/{n_episodes}  success so far "
                f"{np.mean([r['success'] for r in rows]):.3f}",
                flush=True,
            )
    env.close()
    return rows, float(np.median(t_inf)) if t_inf else float("nan")


def summarise(rows, inf_ms, n_params, label):
    n = len(rows)
    reach = float(np.mean([r["reach_success"] for r in rows]))
    push = float(np.mean([r["push_success"] for r in rows]))
    both = float(np.mean([r["success"] for r in rows]))
    # push success CONDITIONAL on having reached, which is the honest phase-B number
    reached = [r for r in rows if r["reach_success"]]
    push_given_reach = (
        float(np.mean([r["push_success"] for r in reached])) if reached else 0.0
    )
    fails = [r["failure_phase"] for r in rows if r["failure_phase"]]
    z = 1.96
    ci = lambda p: (
        max(0.0, p - z * (p * (1 - p) / n) ** 0.5),
        min(1.0, p + z * (p * (1 - p) / n) ** 0.5),
    )
    return {
        "label": label,
        "n_episodes": n,
        "reach_success": reach,
        "reach_ci95": ci(reach),
        "push_success_unconditional": push,
        "push_success_given_reach": push_given_reach,
        "complete_success": both,
        "complete_ci95": ci(both),
        "mean_episode_len": float(np.mean([r["steps"] for r in rows])),
        "median_episode_len": float(np.median([r["steps"] for r in rows])),
        "failure_phase_counts": {p: fails.count(p) for p in set(fails)},
        "inference_ms_median": inf_ms * 1000.0,
        "n_params": int(n_params),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--device", default="cpu")  # SB3: CPU beats MPS for small MlpPolicy
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    from sb3_contrib import TQC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if a.sanity:
        a.steps, a.eval_episodes = 5_000, 10
    tag = a.tag or ("sanity" if a.sanity else f"seed{a.seed}")
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([lambda: Monitor(make_env(seed_offset=TRAIN_SEED_BASE))])
    train_env.seed(a.seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: Monitor(make_env(seed_offset=EVAL_SEED_BASE))])
    eval_env.seed(EVAL_SEED_BASE + a.seed)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )
    eval_env.obs_rms = train_env.obs_rms

    hp = dict(
        learning_rate=7e-4,
        buffer_size=300_000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        learning_starts=min(5_000, a.steps // 5),
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256], n_critics=2),
    )
    model = TQC("MlpPolicy", train_env, seed=a.seed, device=a.device, verbose=1, **hp)
    n_params = sum(p.numel() for p in model.policy.actor.parameters())
    print(f"TQC actor parameters: {n_params:,}  device={a.device}")

    cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out),
        log_path=str(out),
        eval_freq=max(2_000, a.steps // 20),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=a.steps, callback=cb, progress_bar=False)
    train_secs = time.perf_counter() - t0
    model.save(str(out / "final_model"))
    train_env.save(str(out / "vecnormalize.pkl"))
    print(f"\ntrained {a.steps:,} steps in {train_secs / 60:.1f} min")

    # reload the best checkpoint the way evaluation will load it
    best = out / "best_model.zip"
    load_path = best if best.exists() else out / "final_model.zip"
    model = TQC.load(str(load_path), device=a.device)
    vn = VecNormalize.load(
        str(out / "vecnormalize.pkl"), DummyVecEnv([lambda: make_env()])
    )
    vn.training = False

    print(
        f"\nevaluating {load_path.name} on {a.eval_episodes} held-out seeds "
        f"({EVAL_SEED_BASE}..{EVAL_SEED_BASE + a.eval_episodes - 1})"
    )
    rows, inf = evaluate(
        model, a.eval_episodes, EVAL_SEED_BASE, vecnorm=vn, verbose=True
    )
    s = summarise(rows, inf, n_params, f"TQC expert ({tag})")
    s.update(
        train_steps=a.steps,
        train_seconds=train_secs,
        train_seed=a.seed,
        checkpoint=str(load_path.name),
        hyperparameters={
            k: (v if not isinstance(v, dict) else {kk: str(vv) for kk, vv in v.items()})
            for k, v in hp.items()
        },
        eval_seed_base=EVAL_SEED_BASE,
        train_seed_base=TRAIN_SEED_BASE,
        device=a.device,
    )
    json.dump(
        {"summary": s, "episodes": rows},
        open(out / "expert_eval.json", "w"),
        indent=2,
        default=float,
    )

    print(
        f"\n{'=' * 64}\nMuJoCo live closed-loop evaluation — TQC expert ({tag})\n{'=' * 64}"
    )
    print(
        f"  reach success            {s['reach_success']:.3f}  "
        f"[{s['reach_ci95'][0]:.3f}, {s['reach_ci95'][1]:.3f}]"
    )
    print(f"  push success | reached   {s['push_success_given_reach']:.3f}")
    print(
        f"  COMPLETE Reach->Push     {s['complete_success']:.3f}  "
        f"[{s['complete_ci95'][0]:.3f}, {s['complete_ci95'][1]:.3f}]"
    )
    print(
        f"  mean / median ep length  {s['mean_episode_len']:.1f} / {s['median_episode_len']:.0f}"
    )
    print(f"  failure phase            {s['failure_phase_counts']}")
    print(f"  inference (median)       {s['inference_ms_median']:.3f} ms")
    print(f"  actor parameters         {s['n_params']:,}")
    print(f"\nwrote {out}/expert_eval.json")
    return s


if __name__ == "__main__":
    main()
