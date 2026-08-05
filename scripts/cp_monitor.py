"""Stage-aware split-conformal monitor over the planned-vs-executed signal.

  .venv-lerobot/bin/python -m scripts.cp_monitor --rollouts rollouts/task0.hdf5 --alpha 0.1

Two things this gets right that a naive implementation does not.

1. THE GUARANTEE IS PER-ROLLOUT, NOT PER-STEP.
   Split CP at level alpha bounds the error rate of ONE test. Checking a 600-step
   episode step-by-step is 600 tests: even at alpha=0.01 a nominal rollout alarms
   with probability ~1. Calibrating the per-rollout MAXIMUM score instead gives
   P(any alarm during a nominal rollout) <= alpha, which is the quantity the
   project actually promises. `--per-step` reproduces the broken version so the
   difference is visible rather than asserted.

2. THE FINITE-SAMPLE THRESHOLD CAN BE INFINITE.
   The conformal quantile is the ceil((n+1)(1-alpha))-th smallest of n
   calibration scores. If that index exceeds n there is no such value and the
   threshold is +inf -- an alarm that can never fire, while reporting perfect
   coverage. n >= ceil((n+1)(1-alpha)) requires n >= 1/alpha - 1, so alpha=0.1
   needs at least 9 calibration rollouts PER PHASE. This is checked and reported
   rather than silently producing a monitor that does nothing.
"""

import argparse
import json

import numpy as np

from scripts.subphases import PHASE_ID, PHASES, segment


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal threshold. Returns (q, rank, feasible)."""
    n = len(scores)
    if n == 0:
        return np.inf, 0, False
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return np.inf, rank, False
    return float(np.sort(scores)[rank - 1]), rank, True


def load(path, score_key):
    import h5py

    rollouts = []
    with h5py.File(path, "r") as f:
        meta = {k: f.attrs[k] for k in f.attrs}
        for k in sorted(x for x in f if x.startswith("rollout_")):
            g = f[k]
            labels, _ = segment(g["actions"][:, 6])
            rollouts.append(
                {
                    "name": k,
                    "success": bool(g.attrs["success"]),
                    "score": g[score_key][:],
                    "labels": labels,
                }
            )
    return rollouts, meta


def phase_max(r):
    """Max score within each phase; nan where the phase never occurs."""
    out = np.full(len(PHASES), np.nan)
    for p, i in PHASE_ID.items():
        m = r["labels"] == i
        if m.any():
            out[i] = r["score"][m].max()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--score", default="err_norm", choices=["err_norm", "err_raw"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--correction", choices=["none", "bonferroni"], default="bonferroni"
    )
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument(
        "--per-step",
        action="store_true",
        help="also report the naive per-step monitor, for contrast",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rollouts, meta = load(args.rollouts, args.score)
    succ = [r for r in rollouts if r["success"]]
    fail = [r for r in rollouts if not r["success"]]
    print(f"loaded {len(rollouts)} rollouts: {len(succ)} success, {len(fail)} failure")
    print(f"policy sha256: {str(meta.get('policy_sha256'))[:16]}...")
    print(f"score: {args.score}   alpha: {args.alpha}   correction: {args.correction}")

    if len(succ) < 2 or not fail:
        print("\nneed >=2 successes and >=1 failure; collect more rollouts")
        return

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(succ))
    ncal = max(1, int(len(succ) * args.cal_frac))
    cal = [succ[i] for i in idx[:ncal]]
    test_ok = [succ[i] for i in idx[ncal:]]

    alpha_p = (
        args.alpha / len(PHASES) if args.correction == "bonferroni" else args.alpha
    )
    print(
        f"\ncalibration: {len(cal)} nominal rollouts   held-out nominal: {len(test_ok)}"
    )
    print(
        f"per-phase alpha: {alpha_p:.4f}  -> needs n >= {int(np.ceil(1 / alpha_p - 1))} "
        f"calibration rollouts per phase\n"
    )

    cal_max = np.array([phase_max(r) for r in cal])
    print(f"{'phase':11s} {'n_cal':>6s} {'rank':>5s} {'threshold':>11s}  feasible")
    thresholds = np.full(len(PHASES), np.inf)
    feasible = {}
    for p, i in PHASE_ID.items():
        s = cal_max[:, i]
        s = s[~np.isnan(s)]
        q, rank, ok = conformal_quantile(s, alpha_p)
        thresholds[i] = q
        feasible[p] = ok
        print(
            f"{p:11s} {len(s):6d} {rank:5d} {q:11.4f}  {'yes' if ok else 'NO -- inf'}"
        )

    if not all(feasible.values()):
        need = int(np.ceil(1 / alpha_p - 1))
        print(f"\n[!] Some phases cannot be calibrated at alpha={alpha_p:.4f}.")
        print(f"    Need >= {need} calibration rollouts covering each phase.")
        print("    Those phases will never alarm -- the monitor reports perfect")
        print("    coverage while doing nothing. Collect more, raise alpha, or")
        print("    drop the Bonferroni correction.")

    def first_alarm(r):
        for t in range(len(r["score"])):
            if r["score"][t] > thresholds[r["labels"][t]]:
                return t
        return None

    fa = [first_alarm(r) for r in test_ok]
    det = [first_alarm(r) for r in fail]
    far = sum(x is not None for x in fa) / max(1, len(fa))
    tpr = sum(x is not None for x in det) / max(1, len(fail))
    lead = [len(r["score"]) - t for r, t in zip(fail, det) if t is not None]

    print("\n" + "=" * 62)
    print("  STAGE-AWARE MONITOR (per-rollout guarantee)")
    print("=" * 62)
    print(f"  false alarm rate (held-out nominal): {far:.2f}   target <= {args.alpha}")
    print(
        f"  detection rate (failures)          : {tpr:.2f}  ({sum(x is not None for x in det)}/{len(fail)})"
    )
    if lead:
        print(
            f"  median lead time                   : {int(np.median(lead))} steps "
            f"({np.median(lead) / 20:.1f} s before episode end)"
        )

    if args.per_step:
        ps = np.concatenate([r["score"] for r in cal])
        q_ps, _, ok = conformal_quantile(ps, args.alpha)
        far_ps = np.mean([bool((r["score"] > q_ps).any()) for r in test_ok])
        tpr_ps = np.mean([bool((r["score"] > q_ps).any()) for r in fail])
        print("\n" + "-" * 62)
        print("  NAIVE PER-STEP MONITOR (what not to do)")
        print("-" * 62)
        print(f"  threshold from pooled per-step scores: {q_ps:.4f}")
        print(
            f"  false alarm rate: {far_ps:.2f}   (alpha={args.alpha} bounds ONE test,"
        )
        print(f"  detection rate  : {tpr_ps:.2f}    not 600 of them per episode)")

    res = {
        "n_success": len(succ),
        "n_fail": len(fail),
        "alpha": args.alpha,
        "alpha_per_phase": alpha_p,
        "correction": args.correction,
        "score": args.score,
        "thresholds": dict(zip(PHASES, thresholds.tolist())),
        "feasible": feasible,
        "false_alarm_rate": far,
        "detection_rate": tpr,
        "median_lead_steps": float(np.median(lead)) if lead else None,
        "policy_sha256": str(meta.get("policy_sha256")),
    }
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
