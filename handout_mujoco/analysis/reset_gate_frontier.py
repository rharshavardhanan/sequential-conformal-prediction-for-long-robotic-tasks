"""An episode-level arbiter that decides at reset, evaluated exactly, offline.

Two facts from the recorded data reframe the problem:

  * The compliance score has AUC 0.543 at the step the arbiter fires. It cannot
    beat random because it carries no information at decision time.
  * Reset geometry -- cube-to-goal distance and whether the gripper is on the
    pushing side -- has AUC 0.744 / 0.789 BEFORE the first action.

So gate at reset instead of mid-episode. That choice buys exact offline
evaluation: if the gate escalates at t=0 the whole episode is the expert's, and
the expert's outcome for that exact seed is already recorded. No simulator, no
counterfactual assumption, no circularity.

    success(threshold) = mean( expert_success if gated else cheap_success )
    cost(threshold)    = fraction gated

The gate must also be judged on the right target. Escalation is only worth
anything on RESCUABLE episodes (cheap fails, expert succeeds). Escalating where
both fail wastes the expert; escalating where cheap already succeeds can lose a
win -- 4.6% of seeds here are exactly that.

    python analysis/reset_gate_frontier.py
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
NPZ = HERE / "data" / "replay_trajectories_a0.30.npz"
import sys

sys.path.insert(0, str(HERE / "analysis"))
from early_score_search import auc, features, logistic  # noqa: E402

# measured, results/efficiency_summary.json
MS_CHEAP, MS_EXPERT = 0.0845, 0.4136


def frontier(score, cheap, expert, rates):
    """Escalate the top-`rate` fraction by score; report exact success and cost."""
    out = []
    order = np.argsort(-score)  # highest score escalates first
    n = len(score)
    for r in rates:
        k = int(round(r * n))
        gate = np.zeros(n, bool)
        gate[order[:k]] = True
        succ = np.where(gate, expert, cheap)
        out.append(
            {"rate": float(k / n), "success": float(succ.mean()), "n_escalated": int(k)}
        )
    return out


def main():
    z = np.load(NPZ)
    seeds = sorted(
        {int(k.split("/")[1][4:]) for k in z.files if k.startswith("cheap/")}
    )
    cheap = np.array([bool(z[f"cheap/seed{s}/success"]) for s in seeds])
    expert = np.array([bool(z[f"expert/seed{s}/success"]) for s in seeds])
    resc = (~cheap) & expert
    n = len(seeds)

    rows = [features(z, s, 0) for s in seeds]
    names = [k for k in rows[0] if np.isfinite(rows[0][k])]
    X = np.nan_to_num(np.array([[r[k] for k in names] for r in rows], float))

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2 :]

    # Two gates, both fit ONLY on the training half, both scored on the held-out half.
    #   A) predict cheap failure          -- the obvious target
    #   B) predict RESCUABILITY           -- the target that actually pays
    gates = {}
    for name, y in (
        ("predict cheap failure", (~cheap).astype(float)),
        ("predict rescuable", resc.astype(float)),
    ):
        pred, w, mu, sd = logistic(X[tr], y[tr], X[te])
        gates[name] = pred
    gates["cube-goal distance alone"] = X[te][:, names.index("d_cube_goal_0")]
    gates["random"] = rng.random(len(te))

    ch, ex, rs = cheap[te], expert[te], resc[te]
    print(
        f"held-out half: {len(te)} seeds   cheap {ch.mean():.3f}   expert {ex.mean():.3f}"
    )
    print(f"rescuable {rs.mean():.1%}   spoilable {(ch & ~ex).mean():.1%}")
    print(f"ORACLE ceiling {(ch | rs).mean():.3f} at {rs.mean():.1%} escalated\n")

    print("AUC on the held-out half, against each possible target:")
    print(f"{'gate':<28}{'vs cheap-fail':>15}{'vs rescuable':>14}")
    for k, v in gates.items():
        if k == "random":
            continue
        print(f"  {k:<26}{auc(v, ~ch):>15.3f}{auc(v, rs):>14.3f}")

    rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
    print(
        f"\n{'=' * 78}\nSUCCESS vs ESCALATION RATE (exact: gated episodes use the "
        f"recorded expert outcome)\n{'=' * 78}"
    )
    hdr = "  ".join(f"{r:>5.0%}" for r in rates)
    print(f"{'gate':<28}{hdr}")
    res = {}
    for k, v in gates.items():
        f = frontier(v, ch, ex, rates)
        res[k] = f
        print(f"  {k:<26}" + "  ".join(f"{p['success']:>5.3f}" for p in f))
    orac = frontier(rs.astype(float) + 1e-6 * rng.random(len(te)), ch, ex, rates)
    res["oracle"] = orac
    print(
        f"  {'oracle (knows rescuable)':<26}"
        + "  ".join(f"{p['success']:>5.3f}" for p in orac)
    )

    # compute cost: expert steps are 4.9x a cheap step
    print(
        f"\n{'=' * 78}\nCOMPUTE (expert step = {MS_EXPERT / MS_CHEAP:.1f}x cheap step)\n{'=' * 78}"
    )
    best = "predict rescuable"
    print(
        f"{'rate':>6}{'success':>10}{'vs cheap':>10}{'vs expert':>11}"
        f"{'rel. compute':>14}{'success/compute':>17}"
    )
    base_cost = 1.0
    for p in res[best]:
        r = p["rate"]
        cost = (1 - r) * 1.0 + r * (MS_EXPERT / MS_CHEAP)
        print(
            f"{r:>6.0%}{p['success']:>10.3f}{p['success'] - ch.mean():>+10.3f}"
            f"{p['success'] - ex.mean():>+11.3f}{cost:>13.2f}x{p['success'] / cost:>17.3f}"
        )
    exp_cost = MS_EXPERT / MS_CHEAP
    print(
        f"{'100%':>6}{ex.mean():>10.3f}{ex.mean() - ch.mean():>+10.3f}{0.0:>+11.3f}"
        f"{exp_cost:>13.2f}x{ex.mean() / exp_cost:>17.3f}   (expert only)"
    )

    json.dump(
        {
            "n_heldout": len(te),
            "cheap": float(ch.mean()),
            "expert": float(ex.mean()),
            "rescuable": float(rs.mean()),
            "spoilable": float((ch & ~ex).mean()),
            "oracle_ceiling": float((ch | rs).mean()),
            "frontiers": res,
            "expert_step_cost_ratio": MS_EXPERT / MS_CHEAP,
        },
        open(HERE / "results" / "reset_gate_frontier.json", "w"),
        indent=2,
        default=float,
    )
    print(f"\nwrote results/reset_gate_frontier.json")


if __name__ == "__main__":
    main()
