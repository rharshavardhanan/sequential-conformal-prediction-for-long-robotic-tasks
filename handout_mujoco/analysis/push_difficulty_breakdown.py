"""Does push success degrade with initial cube->goal distance?

Two later phases depend on the answer, which is why this runs now rather than after
Phase 9 discovers it:

  Phase 10 splits the evaluation seeds into EASY/HARD on reset geometry alone
  (||cube - task_goal|| below/above median) and asks whether a conformal threshold
  holds its firing rate across that shift. If push success does NOT degrade with
  distance, the median split is not a distribution shift in any sense the monitor can
  see, and Phase 10 has nothing to measure. Better to know before building the arbiter
  than after.

  Gate 5 wants the cheap policy in [0.50, 0.75]. A cheap policy distilled from an expert
  that is uniformly strong behaves differently from one distilled from an expert that is
  strong only on short pushes: in the latter case the cheap policy inherits a
  difficulty-dependent failure profile, which is what gives an escalation trigger
  something to fire on.

Uses the EXISTING seed-0 checkpoint. Trains nothing.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python analysis/push_difficulty_breakdown.py
"""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))
sys.path.insert(0, str(ROOT / "training"))

from envs import PushOnly
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

CKPT = ROOT / "results" / "mujoco_train" / "push_expert" / "seed0" / "best_model.zip"
OUT = ROOT / "results" / "mujoco_train" / "push_difficulty_breakdown"
EVAL_BASE, N_EVAL = 400_000, 200


def wilson(k, n, z=1.96):
    """Wilson score interval. Correct near 0 and 1, where the normal interval is not --
    and quartile bins of 50 episodes at success ~0.95 are exactly that regime."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rollout(model, n=N_EVAL, seed_base=EVAL_BASE):
    env = PushOnly()
    rows = []
    for i in range(n):
        obs, _ = env.reset(seed=seed_base + i)
        d0 = float(np.linalg.norm(env.sim.cube() - env.sim.task_goal))
        term = trunc = False
        steps = 0
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            steps += 1
        rows.append(
            {
                "seed": seed_base + i,
                "d0": d0,
                "success": bool(info["is_success"]),
                "steps": steps,
                "cube_fell": bool(info["cube_fell"]),
                "d_final": float(info["d_push"]),
            }
        )
    env.close()
    return rows


def main():
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint: {CKPT}")
    # HerReplayBuffer asserts an env at load time even for inference-only use
    model = TQC.load(CKPT, env=DummyVecEnv([PushOnly]), device="cpu")
    rows = rollout(model)

    d0 = np.array([r["d0"] for r in rows])
    ok = np.array([r["success"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    fell = np.array([r["cube_fell"] for r in rows])
    dfin = np.array([r["d_final"] for r in rows])

    edges = np.quantile(d0, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        m = (d0 >= lo) & (d0 <= hi) if q == 3 else (d0 >= lo) & (d0 < hi)
        n, k = int(m.sum()), int(ok[m].sum())
        bins.append(
            {
                "quartile": q + 1,
                "d0_range": [float(lo), float(hi)],
                "n": n,
                "success": float(k / n) if n else 0.0,
                "wilson_ci95": list(wilson(k, n)),
                "mean_ep_len": float(steps[m].mean()) if n else 0.0,
                "frac_cube_fell": float(fell[m].mean()) if n else 0.0,
                "mean_final_d_push": float(dfin[m].mean()) if n else 0.0,
            }
        )

    r_pb = float(np.corrcoef(d0, ok.astype(float))[0, 1])
    med = float(np.median(d0))
    easy, hard = d0 < med, d0 >= med
    out = {
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "n_episodes": len(rows),
        "eval_seeds": [EVAL_BASE, EVAL_BASE + N_EVAL - 1],
        "overall_success": float(ok.mean()),
        "overall_wilson_ci95": list(wilson(int(ok.sum()), len(ok))),
        "d0_stats": {"min": float(d0.min()), "median": med, "max": float(d0.max())},
        "quartiles": bins,
        "point_biserial_corr_d0_vs_success": r_pb,
        "median_split": {
            "easy": {"n": int(easy.sum()), "success": float(ok[easy].mean()),
                     "wilson_ci95": list(wilson(int(ok[easy].sum()), int(easy.sum()))),
                     "mean_ep_len": float(steps[easy].mean())},
            "hard": {"n": int(hard.sum()), "success": float(ok[hard].mean()),
                     "wilson_ci95": list(wilson(int(ok[hard].sum()), int(hard.sum()))),
                     "mean_ep_len": float(steps[hard].mean())},
        },
    }
    gap = out["median_split"]["easy"]["success"] - out["median_split"]["hard"]["success"]
    out["median_split"]["easy_minus_hard"] = float(gap)
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2))

    q = out["quartiles"]
    md = [
        "# Push expert: success vs initial cube→goal distance",
        "",
        f"Checkpoint `{out['checkpoint']}`, {out['n_episodes']} deterministic episodes on "
        f"evaluation seeds {EVAL_BASE:,}–{EVAL_BASE + N_EVAL - 1:,}. Generated by "
        "`analysis/push_difficulty_breakdown.py`.",
        "",
        f"Overall success **{out['overall_success']:.3f}** "
        f"[{out['overall_wilson_ci95'][0]:.3f}, {out['overall_wilson_ci95'][1]:.3f}] (Wilson).",
        "",
        "| quartile | ‖cube−goal‖ range (m) | n | success | Wilson CI95 | mean ep_len | cube fell | mean final d_push |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b in q:
        md.append(
            f"| Q{b['quartile']} | {b['d0_range'][0]:.3f} – {b['d0_range'][1]:.3f} | {b['n']} | "
            f"{b['success']:.3f} | [{b['wilson_ci95'][0]:.3f}, {b['wilson_ci95'][1]:.3f}] | "
            f"{b['mean_ep_len']:.1f} | {b['frac_cube_fell']:.3f} | {b['mean_final_d_push']:.4f} |"
        )
    e, h = out["median_split"]["easy"], out["median_split"]["hard"]
    md += [
        "",
        f"Point-biserial correlation between initial distance and success: **{r_pb:+.3f}**.",
        "",
        "## Median split (the Phase 10 regime definition, geometry only — no outcomes)",
        "",
        "| regime | n | success | Wilson CI95 | mean ep_len |",
        "|---|---|---|---|---|",
        f"| EASY (d₀ < {med:.3f}) | {e['n']} | {e['success']:.3f} | "
        f"[{e['wilson_ci95'][0]:.3f}, {e['wilson_ci95'][1]:.3f}] | {e['mean_ep_len']:.1f} |",
        f"| HARD (d₀ ≥ {med:.3f}) | {h['n']} | {h['success']:.3f} | "
        f"[{h['wilson_ci95'][0]:.3f}, {h['wilson_ci95'][1]:.3f}] | {h['mean_ep_len']:.1f} |",
        "",
        f"**EASY − HARD = {gap:+.3f}.**",
        "",
        "## Why this was run before the arbiter, not after",
        "",
        "Phase 10 defines EASY/HARD on reset geometry alone and asks whether a conformal",
        "threshold holds its firing rate across that shift. If success does not degrade with",
        "distance, the median split is not a shift the monitor can see and Phase 10 has",
        "nothing to measure. Gate 5 also depends on this: a cheap policy distilled from a",
        "uniformly-strong expert inherits a flat failure profile, which gives an escalation",
        "trigger far less to fire on than a difficulty-dependent one.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten: {OUT.relative_to(ROOT)}.json / .md")


if __name__ == "__main__":
    main()
