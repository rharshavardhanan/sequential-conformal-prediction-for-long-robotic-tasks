"""THE THESIS TEST: does a stage-aware threshold beat one global threshold?

  .venv-lerobot/bin/python -m scripts.ablation_global

`docs/technical-report.md:441` states the claim -- split CP on a trajectory score
is standard; calibrating a SEPARATE threshold per stage is the novel part. This
measures whether that separation buys detection power.

Two arms, both proper split conformal with a per-rollout guarantee:

    stage-aware  K=4  one threshold per phase, each at alpha/4 (Bonferroni)
    global       K=1  one threshold over all steps, at alpha

WHY THE COMPARISON IS AGAINST *ACHIEVED* FAR, NOT NOMINAL ALPHA
Bonferroni makes the stage-aware arm conservative: its realised false-alarm rate
lands below alpha while the global arm sits near it. Comparing detection at
unequal FAR reads two different points off an ROC curve and calls the higher one
better. So both arms are swept over alpha and plotted as detection vs the FAR
they actually achieve on held-out nominal rollouts. The question is which curve
DOMINATES, not which scores higher at one alpha.

Controls: paired splits (both arms see the identical partition per seed, so split
noise cancels in the difference) and many seeds (103 successes at cal_frac 0.5 is
~51 cal / 52 held-out -- one draw is far too noisy to conclude from).

This script does not modify cp_monitor.py; it imports its primitives.
"""

import argparse
import json

import numpy as np

from scripts.cp_monitor import conformal_quantile, load, phase_agg
from scripts.subphases import PHASES


def global_agg(r, how="max", use_mask=True):
    """K=1 counterpart of phase_agg: one number per rollout, phases ignored."""
    from scripts.cp_monitor import AGGS

    s = r["score"]
    if use_mask and r.get("motion_mask") is not None:
        m = r["motion_mask"]
        if m.any():
            s = s[m]
    return AGGS[how](s)


def alarms_stagewise(r, thresholds):
    """First step whose score exceeds its own phase's threshold."""
    hit = np.nonzero(r["score"] > thresholds[r["labels"]])[0]
    return int(hit[0]) if len(hit) else None


def alarms_global(r, q):
    hit = np.nonzero(r["score"] > q)[0]
    return int(hit[0]) if len(hit) else None


def run_split(cal_idx, test_ok, fail, alpha, pre):
    """Both arms on ONE calibration/held-out partition. Returns per-arm metrics.

    `pre` carries the per-rollout aggregates, precomputed once: they depend on the
    rollout alone, not on alpha or the split, so recomputing them inside the
    seed x alpha loop would repeat identical work 180 times.
    """
    out = {}

    # --- stage-aware, K=4, Bonferroni ---
    alpha_p = alpha / len(PHASES)
    cal_m = pre["phase"][cal_idx]
    th = np.full(len(PHASES), np.inf)
    feasible = True
    for i in range(len(PHASES)):
        s = cal_m[:, i]
        s = s[~np.isnan(s)]
        q, _, ok = conformal_quantile(s, alpha_p)
        th[i] = q
        feasible &= ok
    fa = [alarms_stagewise(r, th) for r in test_ok]
    det = [alarms_stagewise(r, th) for r in fail]
    lead = [len(r["score"]) - t for r, t in zip(fail, det) if t is not None]
    out["stage"] = {
        "far": sum(x is not None for x in fa) / max(1, len(fa)),
        "tpr": sum(x is not None for x in det) / max(1, len(fail)),
        "lead": float(np.median(lead)) if lead else None,
        "feasible": bool(feasible),
    }

    # --- global, K=1, no correction (one test, nothing to correct) ---
    cal_g = pre["glob"][cal_idx]
    qg, _, okg = conformal_quantile(cal_g, alpha)
    fa = [alarms_global(r, qg) for r in test_ok]
    det = [alarms_global(r, qg) for r in fail]
    lead = [len(r["score"]) - t for r, t in zip(fail, det) if t is not None]
    out["global"] = {
        "far": sum(x is not None for x in fa) / max(1, len(fa)),
        "tpr": sum(x is not None for x in det) / max(1, len(fail)),
        "lead": float(np.median(lead)) if lead else None,
        "feasible": bool(okg),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+",
                    default=["data/rollouts/task0_v10.hdf5",
                             "data/rollouts/task1_v10.hdf5",
                             "data/rollouts/task8_v10.hdf5"])
    ap.add_argument("--score", default="err_norm")
    ap.add_argument("--agg", default="max")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--out", default="data/results/ablation_global.json")
    args = ap.parse_args()

    results = {}
    for path in args.rollouts:
        rollouts, meta = load(path, args.score)
        succ = [r for r in rollouts if r["success"]]
        fail = [r for r in rollouts if not r["success"]]
        name = path.split("/")[-1].replace(".hdf5", "")
        print(f"\n{'=' * 74}\n{name}: {len(succ)} success, {len(fail)} failure"
              f"   (policy {str(meta.get('policy_sha256'))[:12]})\n{'=' * 74}")
        print(f"{'':>6s} {'STAGE-AWARE (K=4)':>26s} {'GLOBAL (K=1)':>26s}   paired")
        print(f"{'alpha':>6s} {'FAR':>8s}{'TPR':>9s}{'lead':>9s} "
              f"{'FAR':>8s}{'TPR':>9s}{'lead':>9s}   dTPR@FAR")

        # per-rollout aggregates: computed once, reused across every seed x alpha
        use_mask = not args.no_mask
        pre = {
            "phase": np.array([phase_agg(r, args.agg, use_mask) for r in succ]),
            "glob": np.array([global_agg(r, args.agg, use_mask) for r in succ]),
        }

        per_alpha = {}
        for alpha in args.alphas:
            rows = []
            for seed in range(args.seeds):
                rng = np.random.RandomState(seed)
                idx = rng.permutation(len(succ))
                ncal = max(1, int(len(succ) * args.cal_frac))
                cal_idx = idx[:ncal]
                test_ok = [succ[i] for i in idx[ncal:]]
                rows.append(run_split(cal_idx, test_ok, fail, alpha, pre))
            agg_arm = {}
            for arm in ("stage", "global"):
                agg_arm[arm] = {
                    k: float(np.mean([r[arm][k] for r in rows]))
                    for k in ("far", "tpr")
                }
                leads = [r[arm]["lead"] for r in rows if r[arm]["lead"] is not None]
                agg_arm[arm]["lead"] = float(np.mean(leads)) if leads else None
                agg_arm[arm]["tpr_sd"] = float(np.std([r[arm]["tpr"] for r in rows]))
                agg_arm[arm]["feasible_frac"] = float(
                    np.mean([r[arm]["feasible"] for r in rows]))
            # paired difference, seed by seed
            d = [r["stage"]["tpr"] - r["global"]["tpr"] for r in rows]
            agg_arm["paired_dtpr_mean"] = float(np.mean(d))
            agg_arm["paired_dtpr_sd"] = float(np.std(d))
            # per-seed points, needed for the matched-FAR comparison: the
            # same-alpha difference above is confounded by the arms achieving
            # different FAR, so it cannot answer the thesis question alone.
            agg_arm["per_seed"] = {
                arm: [[r[arm]["far"], r[arm]["tpr"]] for r in rows]
                for arm in ("stage", "global")
            }
            per_alpha[str(alpha)] = agg_arm

            s, g = agg_arm["stage"], agg_arm["global"]
            sl = f"{s['lead']:.0f}" if s["lead"] is not None else "-"
            gl = f"{g['lead']:.0f}" if g["lead"] is not None else "-"
            print(f"{alpha:6.2f} {s['far']:8.3f}{s['tpr']:9.3f}{sl:>9s} "
                  f"{g['far']:8.3f}{g['tpr']:9.3f}{gl:>9s}   "
                  f"{agg_arm['paired_dtpr_mean']:+.3f}")
        results[name] = {"n_success": len(succ), "n_fail": len(fail),
                         "alphas": per_alpha}

    meta_out = {"seeds": args.seeds, "cal_frac": args.cal_frac,
                "score": args.score, "agg": args.agg, "tasks": results}
    json.dump(meta_out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
