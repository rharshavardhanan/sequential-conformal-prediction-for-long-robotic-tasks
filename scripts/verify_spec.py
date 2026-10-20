"""Part A verifications (A1-A4, A6) against collected rollouts.

  .venv-lerobot/bin/python -m scripts.verify_spec --rollouts rollouts/task0.hdf5

Reports numbers, not verdicts. A failed criterion is a result.
"""

import argparse
import json

import h5py
import numpy as np

from scripts.subphases import PHASE_ID, PHASES, segment

POS_SCALE = 0.05
MAX_DELTA = POS_SCALE * np.sqrt(3)  # ||[1,1,1]*0.05||, largest commandable step


def auc(pos, neg):
    """P(failure score > success score), ties at 0.5. Mann-Whitney form."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def ks_2samp(a, b):
    """Two-sample KS statistic and asymptotic p-value (no scipy dependency)."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, "right") / n
    cdf_b = np.searchsorted(b, allv, "right") / m
    d = np.max(np.abs(cdf_a - cdf_b))
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(d), float(min(max(p, 0.0), 1.0))


def occurrences(labels, phase_id):
    """Contiguous runs of one phase type -> list of (start, end)."""
    m = labels == phase_id
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/rollouts/task0_v10.hdf5")
    ap.add_argument(
        "--floor-frac",
        type=float,
        default=0.02,
        help="A4: exclude steps with ||cmd|| below this fraction of max",
    )
    ap.add_argument("--out", default="data/results/verification_task0.json")
    args = ap.parse_args()

    R = []
    with h5py.File(args.rollouts, "r") as f:
        meta = {
            k: (f.attrs[k].item() if hasattr(f.attrs[k], "item") else f.attrs[k])
            for k in f.attrs
        }
        for k in sorted(x for x in f if x.startswith("rollout_")):
            g = f[k]
            lab, segs = segment(g["actions"][:, 6])
            R.append(
                {
                    "name": k,
                    "success": bool(g.attrs["success"]),
                    "init": int(g.attrs["init_state_index"]),
                    "err_norm": g["err_norm"][:],
                    "cmd_norm": g["cmd_norm"][:],
                    "labels": lab,
                    "segs": segs,
                    "grip": g["actions"][:, 6],
                }
            )
    S = [r for r in R if r["success"]]
    F = [r for r in R if not r["success"]]
    out = {"file": args.rollouts, "n": len(R), "n_success": len(S), "n_fail": len(F)}
    print(f"{args.rollouts}: {len(R)} rollouts, {len(S)} success, {len(F)} fail\n")

    # ---------------- A1 ----------------
    print("=" * 74)
    print("A1  sub-phase labelling on POLICY rollouts")
    print("=" * 74)

    def a1(group, name):
        counts = np.array([len(r["segs"]) for r in group])
        closes = np.array(
            [int(((r["grip"][1:] > 0) & (r["grip"][:-1] <= 0)).sum()) for r in group]
        )
        empty = [
            (r["name"], [p for p in PHASES if not (r["labels"] == PHASE_ID[p]).any()])
            for r in group
        ]
        empty = [(n, e) for n, e in empty if e]
        exact8 = float((counts == 8).mean()) if len(counts) else float("nan")
        print(f"\n{name} (n={len(group)}):")
        vals, cnt = np.unique(counts, return_counts=True)
        print(
            "  segment-count distribution: "
            + "  ".join(f"{v}:{c}" for v, c in zip(vals, cnt))
        )
        print(f"  exactly 8 segments        : {exact8:.2f}")
        print(
            f"  open->close transitions   : "
            f"mean={closes.mean():.2f} min={closes.min()} max={closes.max()}  "
            f"(>2 = regrasp: {int((closes > 2).sum())}/{len(closes)})"
        )
        print(f"  rollouts with empty phase : {len(empty)}")
        for n, e in empty[:6]:
            print(f"      {n}: missing {e}")
        return {
            "counts": counts.tolist(),
            "frac_exactly_8": exact8,
            "closes": closes.tolist(),
            "n_regrasp": int((closes > 2).sum()),
            "empty_phase": [{"rollout": n, "missing": e} for n, e in empty],
        }

    out["A1_success"] = a1(S, "SUCCESS")
    out["A1_fail"] = a1(F, "FAILURE")
    ok1 = (
        out["A1_success"]["frac_exactly_8"] >= 0.90
        and not out["A1_success"]["empty_phase"]
    )
    out["A1_pass"] = bool(ok1)
    print(
        f"\n  ACCEPTANCE (>=90% of successes give exactly 8 segments, no empty phase): "
        f"{'MET' if ok1 else 'NOT MET'}"
    )

    # ---------------- A4 (before A2: the floor changes A2's maxima) ----------------
    print("\n" + "=" * 74)
    print("A4  err_norm zero-motion instability")
    print("=" * 74)
    allcmd = np.concatenate([r["cmd_norm"] for r in R])
    pct = {p: float(np.percentile(allcmd, p)) for p in (1, 5, 25, 50)}
    print(f"  ||commanded delta|| percentiles (max possible {MAX_DELTA:.5f}):")
    for p, v in pct.items():
        print(f"    {p:2d}th: {v:.6f}   ({v / MAX_DELTA * 100:5.2f}% of max)")
    fracs = {}
    for t in (0.01, 0.02, 0.05):
        n = int((allcmd < t * MAX_DELTA).sum())
        fracs[t] = {"n": n, "frac": float(n / len(allcmd))}
        print(
            f"  steps below {t * 100:.0f}% of max: {n:6d}  ({n / len(allcmd) * 100:.2f}%)"
        )

    floor = args.floor_frac * MAX_DELTA
    tainted = 0
    for r in R:
        for p, i in PHASE_ID.items():
            m = r["labels"] == i
            if m.any():
                idx = np.flatnonzero(m)[np.argmax(r["err_norm"][m])]
                if r["cmd_norm"][idx] < floor:
                    tainted += 1
    print(
        f"\n  per-phase maxima originating below the {args.floor_frac * 100:.0f}% floor: "
        f"{tainted} / {len(R) * len(PHASES)}"
    )
    print(f"  floor applied for A2/A3: ||cmd|| >= {floor:.6f}")
    out["A4"] = {
        "percentiles": pct,
        "below": fracs,
        "floor": float(floor),
        "tainted_phase_maxima": tainted,
    }
    out["A4_pass"] = bool(tainted == 0)
    print(
        f"  ACCEPTANCE (no phase max from near-zero motion, after floor): "
        f"{'MET' if tainted == 0 else 'NOT MET -- floor is required'}"
    )

    def phase_max(r, i, use_floor=True):
        m = r["labels"] == i
        if use_floor:
            m = m & (r["cmd_norm"] >= floor)
        return r["err_norm"][m].max() if m.any() else np.nan

    # ---------------- A2 ----------------
    print("\n" + "=" * 74)
    print("A2  does the phase-MAX separate? (floored)")
    print("=" * 74)
    print(
        f"{'phase':11s} {'succ mean':>10s} {'sd':>7s} {'succ max':>9s} "
        f"{'fail mean':>10s} {'sd':>7s} {'fail min':>9s} {'AUC':>6s} {'overlap':>8s}"
    )
    a2 = {}
    for p, i in PHASE_ID.items():
        sv = np.array([phase_max(r, i) for r in S])
        sv = sv[~np.isnan(sv)]
        fv = np.array([phase_max(r, i) for r in F])
        fv = fv[~np.isnan(fv)]
        if not len(sv) or not len(fv):
            print(f"{p:11s}  insufficient data")
            continue
        n_below = int((fv < sv.max()).sum())
        a = auc(fv, sv)
        a2[p] = {
            "succ": [
                float(sv.mean()),
                float(sv.std()),
                float(sv.min()),
                float(sv.max()),
            ],
            "fail": [
                float(fv.mean()),
                float(fv.std()),
                float(fv.min()),
                float(fv.max()),
            ],
            "auc": float(a),
            "n_fail_below_max_succ": n_below,
        }
        print(
            f"{p:11s} {sv.mean():>10.3f} {sv.std():>7.3f} {sv.max():>9.3f} "
            f"{fv.mean():>10.3f} {fv.std():>7.3f} {fv.min():>9.3f} {a:>6.2f} "
            f"{n_below:>4d}/{len(fv):<3d}"
        )
    out["A2"] = a2
    best = max(a2.items(), key=lambda kv: kv[1]["auc"]) if a2 else None
    ok2 = bool(best and best[1]["auc"] >= 0.90)
    out["A2_pass"] = ok2
    if best:
        print(
            f"\n  best phase: {best[0]} (AUC {best[1]['auc']:.2f}, "
            f"{best[1]['n_fail_below_max_succ']} failures below max success)"
        )
    print(
        f"  ACCEPTANCE (>=1 phase with clean/near-clean max separation): "
        f"{'MET' if ok2 else 'NOT MET'}"
    )

    # ---------------- A3 ----------------
    print("\n" + "=" * 74)
    print("A3  is pooling occurrence 1 and 2 of each stage justified? (successes only)")
    print("=" * 74)
    print(
        f"{'phase':11s} {'n pairs':>8s} {'occ1 mean':>10s} {'occ2 mean':>10s} "
        f"{'KS D':>7s} {'p':>8s}  pooling"
    )
    a3 = {}
    for p, i in PHASE_ID.items():
        o1, o2 = [], []
        for r in S:
            occ = occurrences(r["labels"], i)
            vals = []
            for s, e in occ:
                seg = slice(s, e)
                m = r["cmd_norm"][seg] >= floor
                vals.append(r["err_norm"][seg][m].max() if m.any() else np.nan)
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) >= 2:
                o1.append(vals[0])
                o2.append(vals[1])
        if len(o1) < 3:
            print(f"{p:11s} {len(o1):>8d}   too few paired occurrences")
            a3[p] = {"n": len(o1), "ks_d": None, "p": None}
            continue
        d, pv = ks_2samp(o1, o2)
        a3[p] = {
            "n": len(o1),
            "occ1_mean": float(np.mean(o1)),
            "occ2_mean": float(np.mean(o2)),
            "ks_d": d,
            "p": pv,
        }
        print(
            f"{p:11s} {len(o1):>8d} {np.mean(o1):>10.3f} {np.mean(o2):>10.3f} "
            f"{d:>7.3f} {pv:>8.3f}  {'OK' if pv > 0.05 else 'DIFFERENT'}"
        )
    out["A3"] = a3
    tested = [v for v in a3.values() if v.get("p") is not None]
    ok3 = bool(tested) and all(v["p"] > 0.05 for v in tested)
    out["A3_pass"] = ok3
    print(
        f"\n  ACCEPTANCE (KS p>0.05 for every stage type -> keep K=4): "
        f"{'MET' if ok3 else 'NOT MET -- consider K=8'}"
    )

    # ---------------- A6 ----------------
    print("\n" + "=" * 74)
    print("A6  init-state disjointness")
    print("=" * 74)
    inits = [r["init"] for r in R]
    dup = len(inits) - len(set(inits))
    si = sorted(r["init"] for r in S)
    fi = sorted(r["init"] for r in F)
    print(
        f"  rollouts={len(inits)}  distinct init states={len(set(inits))}  repeats={dup}"
    )
    print(f"  success init indices: {si}")
    print(f"  failure init indices: {fi}")
    print(f"  success/failure init overlap: {len(set(si) & set(fi))}")
    print("\n  NOTE: a calibration/evaluation split is not yet recorded in the HDF5.")
    print(
        "  Successes must be split into cal/eval by init index and that split stored."
    )
    out["A6"] = {
        "n_distinct": len(set(inits)),
        "repeats": dup,
        "success_inits": si,
        "fail_inits": fi,
        "split_recorded_in_hdf5": False,
    }
    out["A6_pass"] = bool(dup == 0)
    print(
        f"  ACCEPTANCE (zero repeats; split recorded): "
        f"{'PARTIAL -- no repeats, but split not yet stored' if dup == 0 else 'NOT MET'}"
    )

    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
