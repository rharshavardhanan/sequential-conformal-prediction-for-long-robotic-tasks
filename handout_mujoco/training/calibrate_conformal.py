"""Split-conformal thresholds for the escalation trigger, K=1 and K=2.

Calibration uses ONLY seeds 300,000-300,999 and only SUCCESSFUL cheap episodes: the
thresholds describe the nominal behaviour distribution, so an episode that already failed
is not part of it.

Per-episode statistic:  S_k = max score over phase k        (a causal, whole-phase max --
for calibration that is legitimate, because it is the distribution of the monitored
statistic under nominal operation, not a prediction made at decision time.)

Threshold: the order statistic  q = S_(ceil((n+1)(1-alpha)))  of the n calibration values,
the standard split-conformal quantile with finite-sample coverage.

K=2 applies a Bonferroni split (alpha_k = alpha/2) because two phase-conditioned
thresholds are checked within one episode. K=1 pools both phases under one threshold with
no correction. Both are reported because the reach phase turns out to be near-degenerate
under a scripted P-servo reach stage -- see the phase sample counts.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python training/calibrate_conformal.py
"""

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))
sys.path.insert(0, str(ROOT / "analysis"))

from composed import PUSH, REACH
from score_informativeness import CAL_HI, CAL_LO, collect

OUT = ROOT / "results" / "mujoco_train" / "calibration"
ALPHA = 0.10
ALPHA_ESC = (0.20, 0.30, 0.40)
EVAL_LO, EVAL_HI = 400_000, 400_200
PRIMARY = "b_ensemble_disagreement"      # selected by Gate 6 on held-out AUC
SECONDARY = "a_compliance_residual"


def conformal_q(values, alpha):
    """Split-conformal upper quantile via the order statistic. None if infeasible."""
    v = np.sort(np.asarray([x for x in values if np.isfinite(x)], float))
    n = len(v)
    if n == 0:
        return None, n, None
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return None, n, rank          # infeasible: n too small for this alpha
    return float(v[rank - 1]), n, rank


def phase_stats(episodes, cand, phase):
    out = []
    for e in episodes:
        v = [s[cand] for s in e["steps"] if s["phase"] == phase and np.isfinite(s[cand])]
        if v:
            out.append(max(v))
    return out


def main():
    assert CAL_HI <= EVAL_LO, "calibration and evaluation seed ranges must be disjoint"
    print(f"seed-disjointness: calibration [{CAL_LO:,}, {CAL_HI - 1:,}] and evaluation "
          f"[{EVAL_LO:,}, {EVAL_HI - 1:,}] are disjoint by construction -- "
          f"{CAL_HI - 1:,} < {EVAL_LO:,}")
    episodes = collect(list(range(CAL_LO, CAL_HI)))
    kept = [e for e in episodes if not e["failed"]]
    print(f"calibration episodes {len(episodes):,}, successful (kept) {len(kept):,}")

    # motion-floor: how often the compliance residual is undefined
    tot = sum(len(e["steps"]) for e in kept)
    unscorable = sum(1 for e in kept for s in e["steps"]
                     if not np.isfinite(s[SECONDARY]))
    out = {
        "cheap_policy": "cheap_policy_16x16.pt",
        "primary_score": PRIMARY,
        "secondary_score": SECONDARY,
        "alpha": ALPHA,
        "calibration_seeds": [CAL_LO, CAL_HI - 1],
        "evaluation_seeds": [EVAL_LO, EVAL_HI - 1],
        "seed_disjoint": True,
        "n_cal_episodes": len(episodes),
        "n_kept_successful": len(kept),
        "motion_floor_fraction_secondary": unscorable / max(tot, 1),
        "thresholds": {},
    }

    for cand in (PRIMARY, SECONDARY):
        block = {"K2": {}, "K1": {}}
        # ---- K = 2, Bonferroni
        for name, ph in (("reach", REACH), ("push", PUSH)):
            S = phase_stats(kept, cand, ph)
            q_hat, n, rank = conformal_q(S, ALPHA / 2)
            entry = {"n": n, "rank": rank, "alpha_k": ALPHA / 2, "q_hat": q_hat,
                     "q_esc": {}}
            for ae in ALPHA_ESC:
                qe, _, r = conformal_q(S, ae)
                entry["q_esc"][f"{ae:.2f}"] = {"q": qe, "rank": r}
            entry["q_esc_below_q_hat"] = {
                k: (v["q"] is not None and q_hat is not None and v["q"] < q_hat)
                for k, v in entry["q_esc"].items()
            }
            block["K2"][name] = entry
        # ---- K = 1, pooled, no correction
        S = phase_stats(kept, cand, REACH) + phase_stats(kept, cand, PUSH)
        q_hat, n, rank = conformal_q(S, ALPHA)
        entry = {"n": n, "rank": rank, "alpha": ALPHA, "q_hat": q_hat, "q_esc": {}}
        for ae in ALPHA_ESC:
            qe, _, r = conformal_q(S, ae)
            entry["q_esc"][f"{ae:.2f}"] = {"q": qe, "rank": r}
        entry["q_esc_below_q_hat"] = {
            k: (v["q"] is not None and q_hat is not None and v["q"] < q_hat)
            for k, v in entry["q_esc"].items()
        }
        block["K1"]["pooled"] = entry
        out["thresholds"][cand] = block

    # the inequality the demo depends on
    checks = []
    for cand, blk in out["thresholds"].items():
        for K in ("K2", "K1"):
            for ph, e in blk[K].items():
                for ae, ok in e["q_esc_below_q_hat"].items():
                    checks.append((cand, K, ph, ae, ok))
    out["all_q_esc_below_q_hat"] = all(ok for *_, ok in checks)

    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2))

    md = [
        "# Conformal calibration — escalation and safety thresholds",
        "",
        f"Generated by `training/calibrate_conformal.py`. Calibration seeds "
        f"**{CAL_LO:,}–{CAL_HI - 1:,}**, evaluation seeds **{EVAL_LO:,}–{EVAL_HI - 1:,}**, "
        f"disjoint by construction ({CAL_HI - 1:,} < {EVAL_LO:,}).",
        "",
        f"{len(episodes):,} calibration episodes, **{len(kept):,} successful and kept**. "
        f"α = {ALPHA}. Motion-floor fraction (steps where the compliance residual is "
        f"undefined): **{out['motion_floor_fraction_secondary']:.1%}**.",
        "",
    ]
    for cand, blk in out["thresholds"].items():
        star = " — **PRIMARY, selected by Gate 6**" if cand == PRIMARY else ""
        md += [f"## `{cand}`{star}", "",
               "| K | phase | n | rank | α_k | q̂ (safety) | q_esc 0.20 | 0.30 | 0.40 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for K in ("K2", "K1"):
            for ph, e in blk[K].items():
                a_k = e.get("alpha_k", e.get("alpha"))
                g = lambda k: (f"{e['q_esc'][k]['q']:.4f}" if e["q_esc"][k]["q"] is not None else "—")
                qh = f"{e['q_hat']:.4f}" if e["q_hat"] is not None else "INFEASIBLE"
                md.append(f"| {K} | {ph} | {e['n']} | {e['rank']} | {a_k:.3f} | {qh} | "
                          f"{g('0.20')} | {g('0.30')} | {g('0.40')} |")
        md.append("")
    md += [
        f"**q_esc < q̂ in every phase, every K, every α_esc: "
        f"{'YES' if out['all_q_esc_below_q_hat'] else 'NO'}.** This is the leading-indicator "
        "property the demo depends on: escalation fires strictly before the safety "
        "threshold would be crossed.",
        "",
        "The bounded quantity is the spurious-escalation rate on nominal episodes. It is "
        "not a safety guarantee for an arbitrated trajectory.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten: {OUT.relative_to(ROOT)}.json / .md")


if __name__ == "__main__":
    main()
