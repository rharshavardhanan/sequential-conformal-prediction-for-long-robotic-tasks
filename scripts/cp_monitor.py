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
import sys

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
                    "motion_mask": g["motion_mask"][:] if "motion_mask" in g else None,
                    "split": g.attrs.get("split"),
                }
            )
    return rollouts, meta


AGGS = {
    "max": lambda v: v.max(),
    "p95": lambda v: np.percentile(v, 95),
    "p90": lambda v: np.percentile(v, 90),
    "p75": lambda v: np.percentile(v, 75),
    "mean": lambda v: v.mean(),
}


def phase_agg(r, how="max", use_mask=True):
    """D.10 aggregation per phase; nan where the phase never occurs.

    D.9: low-motion steps are excluded from SCORING via the stored mask, but they
    are never dropped from the data -- they are the stall signature.
    """
    fn = AGGS[how]
    out = np.full(len(PHASES), np.nan)
    for p, i in PHASE_ID.items():
        m = r["labels"] == i
        if use_mask and r.get("motion_mask") is not None:
            m = m & r["motion_mask"]
        if m.any():
            out[i] = fn(r["score"][m])
    return out


def dwell_maxima(r):
    """Longest contiguous run of each phase in one rollout; nan where absent.

    Calibrated the same way the deviation score is: the per-ROLLOUT maximum, not
    pooled per-segment durations. A rollout holds several segments per phase, so a
    per-segment quantile is not a per-rollout bound -- measured, it fires on 47%
    of successful rollouts at beta=0.10.
    """
    lab = np.asarray(r["labels"])
    out = np.full(len(PHASES), np.nan)
    if len(lab) == 0:
        return out
    edges = np.flatnonzero(np.diff(lab)) + 1
    for start, end in zip(np.r_[0, edges], np.r_[edges, len(lab)]):
        i = int(lab[start])
        out[i] = end - start if np.isnan(out[i]) else max(out[i], end - start)
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
    ap.add_argument("--agg", choices=list(AGGS), default="max",
                    help="D.10: per-phase aggregation. Default max until the "
                         "held-out comparison replicates on unseen data.")
    ap.add_argument("--no-mask", action="store_true",
                    help="ignore the stored motion mask when scoring")
    ap.add_argument("--alpha-esc", type=float, default=0.3,
                    help="risk budget for the escalation threshold; must exceed "
                         "--alpha so q_esc < q_safe and escalation precedes violation")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="stall risk budget, separate from alpha (D.10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rollouts, meta = load(args.rollouts, args.score)
    succ = [r for r in rollouts if r["success"]]
    fail = [r for r in rollouts if not r["success"]]
    print(f"loaded {len(rollouts)} rollouts: {len(succ)} success, {len(fail)} failure")
    print(f"policy sha256: {str(meta.get('policy_sha256'))[:16]}...")
    print(f"score: {args.score}   agg: {args.agg}   alpha: {args.alpha}   correction: {args.correction}")

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

    cal_max = np.array([phase_agg(r, args.agg, not args.no_mask) for r in cal])
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

    if args.alpha_esc <= args.alpha:
        raise SystemExit(f"[abort] --alpha-esc ({args.alpha_esc}) must exceed --alpha "
                         f"({args.alpha}); otherwise q_esc >= q_safe and escalation "
                         f"cannot precede violation")

    alpha_esc_p = (args.alpha_esc / len(PHASES)
                   if args.correction == "bonferroni" else args.alpha_esc)
    beta_p = args.beta / len(PHASES) if args.correction == "bonferroni" else args.beta

    esc = np.full(len(PHASES), np.inf)
    for p, i in PHASE_ID.items():
        s = cal_max[:, i]
        esc[i] = conformal_quantile(s[~np.isnan(s)], alpha_esc_p)[0]

    # FLAT CP: the same calibration with K=1 -- one threshold over the whole
    # episode, per-rollout max of every scorable step, no Bonferroni because
    # there is only one test. This is the baseline the per-phase thresholds are
    # meant to beat, and it is computed here so the viewer can display both
    # without ever recalibrating at demo time.
    flat_cal = np.array([r["score"][r["motion_mask"]].max()
                         if r.get("motion_mask") is not None else r["score"].max()
                         for r in cal])
    q_flat, _, flat_ok = conformal_quantile(flat_cal, args.alpha)

    cal_dwell = np.array([dwell_maxima(r) for r in cal])
    dwell = np.full(len(PHASES), np.inf)
    for p, i in PHASE_ID.items():
        s = cal_dwell[:, i]
        dwell[i] = conformal_quantile(s[~np.isnan(s)], beta_p)[0]

    print(f"\n{'phase':11s} {'q_esc':>11s} {'q_safe':>11s} {'d_hat':>9s}")
    for p, i in PHASE_ID.items():
        print(f"{p:11s} {esc[i]:11.4f} {thresholds[i]:11.4f} {dwell[i]:9.1f}")
    print(f"\nflat CP (K=1, one threshold for the whole episode): {q_flat:.4f}")
    print("note: grasp/release dwell is structurally 10 (the +/-5 window), so those")
    print("      two phases can never fire STUCK_TIMEOUT -- reach/transport only.")

    def first_alarm(r):
        """First scorable timestep exceeding this phase's threshold.

        The motion mask is applied HERE as well as in calibration. Calibrating
        with the mask and evaluating without it lets a low-motion step (mean
        err_norm 0.99, max 2.78 -- the extreme tail) raise an alarm calibration
        never saw.
        """
        mask = r.get("motion_mask")
        for t in range(len(r["score"])):
            if mask is not None and not mask[t]:
                continue
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
        "score": args.score, "agg": args.agg,
        "thresholds": dict(zip(PHASES, thresholds.tolist())),
        "feasible": feasible,
        "false_alarm_rate": far,
        "detection_rate": tpr,
        "median_lead_steps": float(np.median(lead)) if lead else None,
        "policy_sha256": str(meta.get("policy_sha256")),
    }
    res.update({
        "escalation_thresholds": dict(zip(PHASES, esc.tolist())),
        "dwell_limits": dict(zip(PHASES, dwell.tolist())),
        "alpha_esc": args.alpha_esc, "beta": args.beta,
        "cal_frac": args.cal_frac, "seed": args.seed,
        "flat_threshold": float(q_flat),
        "flat_feasible": bool(flat_ok),
        "phase_window": 5,
        "motion_floor_abs": float(meta.get("motion_floor_abs", 0.0017320508075688774)),
        "n_cal": len(cal),
        "command": " ".join([".venv-lerobot/bin/python", "-m", "scripts.cp_monitor"]
                            + sys.argv[1:]),
    })
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
