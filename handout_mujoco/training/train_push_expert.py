"""Train the PushOnly expert: TQC + HER on the live MuJoCo physics.

MuJoCo-native. No PyBullet checkpoint, threshold or success number is involved. The
environment is `live/envs.PushOnly` exactly as gated in Phase 1/1.5 -- same scene, same
tuned controller, same hand-off distribution the composed system will present.

Seed discipline (live/MUJOCO_TRAINING_SPEC.md section 7), disjoint BY CONSTRUCTION:
    train        0 -  99,999      drawn explicitly, never implicitly
    calibration  300,000 - 300,999   (Phase 7, untouched here)
    evaluation   400,000 - 400,999

Training episodes draw their seed from the train range through `TrainSeeds`, so the
partition is a property of the code rather than a claim in a document.

    python training/train_push_expert.py --sanity               # 2a: 5k, blocking checks
    python training/train_push_expert.py --steps 300000 --seed 0  # 2b: shape check
    python training/train_push_expert.py --steps 1000000 --seed 0 # 2c
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

import gymnasium as gym
import torch
from envs import PUSH_LIMIT, PushOnly
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

TRAIN_SEED_LO, TRAIN_SEED_HI = 0, 100_000
EVAL_SEED_BASE = 400_000
OUT_ROOT = ROOT / "results" / "mujoco_train" / "push_expert"

HYPERPARAMS = dict(
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=2048,
    gamma=0.95,
    tau=0.05,
    learning_starts=1_000,
    policy_kwargs=dict(net_arch=[512, 512, 512]),
)
HER_KWARGS = dict(n_sampled_goal=4, goal_selection_strategy="future")
# No VecNormalize. The reward is sparse 0/-1 and the observation is goal-conditioned;
# normalising observations would have to be replayed identically inside HER's reward
# recomputation and inside the Phase-5 distillation. Recorded as a decision, not an
# oversight -- revisit only if Gate 2 fails and the diagnostic points at scaling.
USE_VECNORMALIZE = False


class TrainSeeds(gym.Wrapper):
    """Reset from an explicit seed drawn from the TRAIN partition, never implicitly.

    Without this the env would reset off its own PRNG stream and the seed partition would
    be an assertion in a document rather than a property of the run.
    """

    def __init__(self, env, seed):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.seeds_used = []

    def reset(self, **kwargs):
        s = int(self.rng.integers(TRAIN_SEED_LO, TRAIN_SEED_HI))
        self.seeds_used.append(s)
        return self.env.reset(seed=s)


def make_env(seed):
    return DummyVecEnv([lambda: Monitor(TrainSeeds(PushOnly(), seed))])


def evaluate(model, n_episodes, seed_base=EVAL_SEED_BASE, deterministic=True):
    """Deterministic rollout on explicit held-out seeds. One env, no wrappers."""
    env = PushOnly()
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
                "d_push": float(info["d_push"]),
                "cube_fell": bool(info["cube_fell"]),
            }
        )
    env.close()
    return rows, (float(np.median(t_inf)) if t_inf else float("nan"))


def _dist_key(rows):
    return "d_push" if rows and "d_push" in rows[0] else "d_reach"


def actor_params(model):
    """Only the ACTOR runs at inference. The saved policy also carries 2 critics and 2
    target critics, so `sum(policy.parameters())` overstates deployed cost ~4x. Phase 9's
    compute ratio must use this number, not the total."""
    return int(sum(p.numel() for p in model.actor.parameters()))


def total_params(model):
    return int(sum(p.numel() for p in model.policy.parameters()))


def summarise(rows, inf_s, n_params, label, n_total=None):
    n = len(rows)
    p = float(np.mean([r["success"] for r in rows]))
    z = 1.96
    half = z * (p * (1 - p) / n) ** 0.5
    steps = np.array([r["steps"] for r in rows])
    return {
        "label": label,
        "n_episodes": n,
        "success": p,
        "success_ci95": [max(0.0, p - half), min(1.0, p + half)],
        "mean_episode_len": float(steps.mean()),
        "median_episode_len": float(np.median(steps)),
        "frac_cube_fell": float(np.mean([r["cube_fell"] for r in rows])),
        # push rows carry d_push, reach rows carry d_reach. Keyed off the rows rather
        # than hard-coded, so the push schema is unchanged and reach gets its own field.
        f"mean_final_{_dist_key(rows)}": float(
            np.mean([r[_dist_key(rows)] for r in rows])
        ),
        "inference_ms_median": inf_s * 1000.0,
        "n_params_actor": int(n_params),
        "total_params": int(n_total if n_total is not None else n_params),
    }


class SeededEval(BaseCallback):
    """Deterministic eval on EXPLICIT held-out seeds. SB3's EvalCallback cannot do this.

    Also records wall-clock and steps-per-second measured only over the interval BETWEEN
    evaluations, so the reported fps is the steady-state rate after the buffer has filled
    and gradients are running -- not the free-running rate of the first minute.
    """

    def __init__(self, out_dir, eval_freq, n_eval, learning_starts, eval_fn=None, verbose=1):
        super().__init__(verbose)
        self.eval_fn = eval_fn or evaluate
        self.out_dir = pathlib.Path(out_dir)
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.learning_starts = learning_starts
        self.history = []
        self.best = -1.0
        self._last_t = None
        self._last_n = 0

    def _on_training_start(self):
        self._last_t, self._last_n = time.perf_counter(), self.num_timesteps

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
        now = time.perf_counter()
        d_t, d_n = now - self._last_t, self.num_timesteps - self._last_n
        fps = d_n / d_t if d_t > 0 else float("nan")
        rows, inf_s = self.eval_fn(self.model, self.n_eval)
        s = summarise(rows, inf_s, actor_params(self.model),
                      f"step{self.num_timesteps}", total_params(self.model))
        log = self.model.logger.name_to_value
        s.update(
            timesteps=int(self.num_timesteps),
            fps_interval=float(fps),
            after_learning_starts=bool(self._last_n >= self.learning_starts),
            actor_loss=float(log.get("train/actor_loss", float("nan"))),
            critic_loss=float(log.get("train/critic_loss", float("nan"))),
            ent_coef=float(log.get("train/ent_coef", float("nan"))),
        )
        self.history.append(s)
        if s["success"] > self.best:
            self.best = s["success"]
            self.model.save(self.out_dir / "best_model.zip")
        np.savez(
            self.out_dir / "evaluations.npz",
            timesteps=np.array([h["timesteps"] for h in self.history]),
            success=np.array([h["success"] for h in self.history]),
            ep_len=np.array([h["mean_episode_len"] for h in self.history]),
            fps=np.array([h["fps_interval"] for h in self.history]),
        )
        (self.out_dir / "eval_history.json").write_text(json.dumps(self.history, indent=2))
        if self.verbose:
            print(
                f"  [{self.num_timesteps:>7,}] success {s['success']:.3f}  "
                f"ep_len {s['mean_episode_len']:5.1f}  fps {fps:6.1f}  "
                f"actor {s['actor_loss']:+.3f}  critic {s['critic_loss']:+.3f}  "
                f"ent {s['ent_coef']:.4f}",
                flush=True,
            )
        self._last_t, self._last_n = time.perf_counter(), self.num_timesteps
        return True


def build(seed, device, out_dir):
    env = make_env(seed)
    model = TQC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(HER_KWARGS),
        device=device,
        seed=seed,
        verbose=0,
        **HYPERPARAMS,
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "algo": "TQC",
                "replay_buffer": "HerReplayBuffer",
                "her": HER_KWARGS,
                "hyperparams": {
                    k: (v if not isinstance(v, dict) else {kk: list(vv) if isinstance(vv, list) else vv
                                                           for kk, vv in v.items()})
                    for k, v in HYPERPARAMS.items()
                },
                "train_freq": str(model.train_freq),
                "gradient_steps": model.gradient_steps,
                "top_quantiles_to_drop_per_net": model.top_quantiles_to_drop_per_net,
                "device": str(model.device),
                "seed": seed,
                "use_vecnormalize": USE_VECNORMALIZE,
                "env": "live/envs.py:PushOnly",
                "train_seed_range": [TRAIN_SEED_LO, TRAIN_SEED_HI],
                "eval_seed_base": EVAL_SEED_BASE,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )
    return env, model


# ----------------------------------------------------------------- 2a sanity checks
def sanity(model, env, out_dir, steps):
    """Step 2a. Every check prints PASS/FAIL; the relabeling check is BLOCKING.

    The relabeling check exists because a mis-wired goal_selection_strategy is silent:
    the buffer fills, losses are finite, training runs, and you get 1M steps of vanilla
    TQC on a sparse reward that almost never fires. It is checked against the exact code
    path SB3 uses (`_get_virtual_samples`), not a proxy.
    """
    res = []

    def check(name, ok, detail="", blocking=False):
        res.append((name, bool(ok), detail, blocking))
        tag = "PASS" if ok else ("FAIL-BLOCKING" if blocking else "FAIL")
        print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""), flush=True)
        return bool(ok)

    rb = model.replay_buffer

    # 1 -- the buffer is a HER buffer, configured as specified
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

    check(
        "replay buffer is HerReplayBuffer with the specified HER config",
        isinstance(rb, HerReplayBuffer)
        and rb.n_sampled_goal == 4
        and rb.goal_selection_strategy == GoalSelectionStrategy.FUTURE,
        f"{type(rb).__name__}, n_sampled_goal={rb.n_sampled_goal}, "
        f"strategy={rb.goal_selection_strategy}, her_ratio={rb.her_ratio:.2f}",
    )

    # 2 -- it actually populated, with completed episodes
    n_complete = int((rb.ep_length > 0).sum())
    check(
        "HER buffer populates with COMPLETED episodes",
        rb.size() > 0 and n_complete > 0,
        f"{rb.size():,} transitions, {n_complete:,} belong to completed episodes",
    )

    # 3 -- BLOCKING: relabeling is actually happening
    valid = np.flatnonzero(rb.ep_length > 0)
    bi, ei = np.unravel_index(valid[: min(512, len(valid))], rb.ep_length.shape)
    stored = rb.observations["desired_goal"][bi, ei, :].copy()
    virtual = rb._get_virtual_samples(bi, ei, None)
    relabelled = virtual.observations["desired_goal"].cpu().numpy()
    differs = ~np.all(np.isclose(stored, relabelled, atol=1e-8), axis=1)
    frac = float(differs.mean())
    check(
        "HER RELABELING IS HAPPENING (virtual desired_goal != stored desired_goal)",
        frac > 0.5,
        f"{frac:.1%} of {len(bi)} virtual transitions carry a different goal; "
        f"mean |delta| = {np.abs(stored - relabelled).sum(1).mean():.4f} m",
        blocking=True,
    )
    # the relabelled goal must be an ACHIEVED goal from the same episode's future
    ach = rb.next_observations["achieved_goal"][:, 0, :]
    hit = np.array(
        [np.isclose(relabelled[k], ach, atol=1e-6).all(1).any() for k in range(len(bi))]
    )
    check(
        "each relabelled goal is an achieved goal that occurred in the buffer",
        hit.mean() > 0.95,
        f"{hit.mean():.1%} matched an achieved cube position",
        blocking=True,
    )
    # and the recomputed reward must fire more often than the original sparse reward
    r_virtual = virtual.rewards.cpu().numpy().ravel()
    r_stored = rb.rewards[bi, ei]
    check(
        "relabeling makes the sparse reward fire (that is the point of HER)",
        float((r_virtual == 0.0).mean()) > float((r_stored == 0.0).mean()),
        f"reward==0 in {float((r_virtual == 0.0).mean()):.1%} of virtual vs "
        f"{float((r_stored == 0.0).mean()):.1%} of stored transitions",
        blocking=True,
    )

    # 4 -- truncation is plumbed through to the bootstrap (this is what FIX 1 buys)
    n_timeout = int(rb.timeouts.sum())
    n_done = int(rb.dones.sum())
    check(
        "budget expiry is recorded as a TIMEOUT, so SB3 still bootstraps",
        n_timeout > 0,
        f"{n_timeout:,} timeouts vs {n_done:,} dones "
        f"({100 * n_timeout / max(n_done, 1):.0f}% of dones are timeouts)",
    )
    check(
        "episodes end at or under the 50-step budget",
        bool((rb.ep_length[rb.ep_length > 0] <= PUSH_LIMIT).all()),
        f"max episode length {int(rb.ep_length.max())} <= {PUSH_LIMIT}",
    )

    # 5 -- losses finite
    log = model.logger.name_to_value
    losses = {k: float(log.get(f"train/{k}", float("nan")))
              for k in ("actor_loss", "critic_loss", "ent_coef")}
    check(
        "actor / critic / ent_coef are finite, no NaN",
        all(np.isfinite(v) for v in losses.values()),
        ", ".join(f"{k}={v:+.4f}" for k, v in losses.items()),
    )
    params = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.policy.parameters()])
    check(
        "no NaN or inf in any policy parameter",
        bool(np.all(np.isfinite(params))),
        f"{params.size:,} parameters",
    )

    # 6 -- checkpoint round-trips
    path = out_dir / "sanity_model.zip"
    model.save(path)
    # HerReplayBuffer needs the env at load time to recompute virtual rewards
    reloaded = TQC.load(path, env=env, device=model.device)
    obs = env.reset()
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = reloaded.predict(obs, deterministic=True)
    check(
        "checkpoint saves and reloads to identical deterministic actions",
        bool(np.allclose(a1, a2, atol=1e-6)),
        f"max |delta| = {np.abs(np.asarray(a1) - np.asarray(a2)).max():.2e}",
    )

    n_fail = sum(1 for _, ok, _, _ in res if not ok)
    n_block = sum(1 for _, ok, _, b in res if not ok and b)
    print(f"\n  {len(res) - n_fail}/{len(res)} sanity checks passed"
          f"{f'  ({n_block} BLOCKING failures)' if n_block else ''}", flush=True)
    return n_fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    steps = 5_000 if a.sanity else a.steps
    name = "sanity" if a.sanity else f"seed{a.seed}"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PushOnly expert -- TQC + HER  |  {name}  |  {steps:,} steps  |  device {a.device}")
    env, model = build(a.seed, a.device, out_dir)
    print(f"  actor parameters: {actor_params(model):,}  "
          f"(total saved policy incl. critics: {total_params(model):,})")
    print(f"  train_freq {model.train_freq}  gradient_steps {model.gradient_steps}  "
          f"learning_starts {HYPERPARAMS['learning_starts']:,}", flush=True)

    eval_freq = min(a.eval_freq, max(1_000, steps // 4))
    cb = SeededEval(out_dir, eval_freq, 20 if a.sanity else a.n_eval,
                    HYPERPARAMS["learning_starts"])
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=cb, log_interval=4, progress_bar=False)
    wall = time.perf_counter() - t0
    model.save(out_dir / "final_model.zip")

    print(f"\n  wall-clock {wall / 60:.1f} min  |  overall {steps / wall:.1f} steps/s")
    post = [h for h in cb.history if h["after_learning_starts"]]
    if post:
        f = np.array([h["fps_interval"] for h in post])
        print(f"  steady-state fps AFTER the buffer fills: median {np.median(f):.1f} "
              f"(min {f.min():.1f}, max {f.max():.1f}) over {len(f)} intervals -- MEASURED, not extrapolated")

    ok = True
    if a.sanity:
        print("\nStep 2a sanity checks")
        ok = sanity(model, env, out_dir, steps)

    rows, inf_s = evaluate(model, 20 if a.sanity else 200)
    final = summarise(rows, inf_s, actor_params(model), f"{name}_final", total_params(model))
    final.update(total_timesteps=steps, wall_clock_s=wall, overall_fps=steps / wall,
                 steady_fps_median=(float(np.median([h["fps_interval"] for h in post])) if post else None))
    (out_dir / "push_eval.json").write_text(json.dumps(final, indent=2))
    print(f"\n  FINAL eval ({final['n_episodes']} held-out seeds): success "
          f"{final['success']:.3f} CI95 [{final['success_ci95'][0]:.3f}, {final['success_ci95'][1]:.3f}]"
          f"  ep_len {final['mean_episode_len']:.1f}")
    print(f"  artifacts -> {out_dir.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
