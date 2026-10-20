"""Is there ANY signal available early enough for escalation to act on?

The arbiter's compliance score has AUC 0.543 at step 4, where it fires. That is why
conformal escalation ties with random (p = 0.24): calibration controls the firing
RATE, it cannot create signal that is not there at decision time.

This searches the recorded state for a statistic that separates failures from
successes EARLY. Every candidate obeys two rules:

  * **causal**  - at decision step t it uses only rows 0..t. No lookahead.
  * **deployable** - it uses only what a running policy can see: end-effector pose,
    cube pose, joint angles, its own commanded actions, and the goals (which are
    given at reset, in the observation). Nothing derived from the outcome.

Statistics like final cube-to-goal distance (AUC 1.000) or episode length (0.999)
are excluded by construction: they ARE the success criterion.

    python analysis/early_score_search.py
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
NPZ = HERE / "data" / "replay_trajectories_a0.30.npz"
DECISION_STEPS = (0, 2, 4, 8)


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if label.all() or not label.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score), float)
    r[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = int(label.sum())
    n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0


def features(z, seed, t):
    """Everything a running policy could compute at decision step t."""
    g = lambda f: np.asarray(z[f"cheap/seed{seed}/{f}"], float)
    ee, obj, act = g("ee"), g("obj_pos"), g("actions")
    sc, lab, qp = g("score"), np.asarray(z[f"cheap/seed{seed}/labels"]), g("qpos")
    goal, rgoal = g("task_goal"), g("reach_goal")
    T = len(sc)
    t = min(t, T - 1)

    o0, ot = obj[0], obj[min(t, len(obj) - 1)]
    e0, et = ee[0], ee[min(t, len(ee) - 1)]
    d_goal_0 = np.linalg.norm(o0 - goal)
    d_goal_t = np.linalg.norm(ot - goal)

    # geometry: is the gripper on the far side of the cube from the goal, so that
    # pushing the cube moves it TOWARD the goal? cos +1 = well placed, -1 = wrong side.
    push_dir = unit(goal - ot)
    approach = unit(ot - et)
    align = float(np.dot(approach, push_dir))
    align0 = float(np.dot(unit(o0 - e0), unit(goal - o0)))

    f = {
        # --- pure reset geometry, available at t = 0 before any action ---
        "d_cube_goal_0": float(d_goal_0),
        "d_ee_cube_0": float(np.linalg.norm(o0 - e0)),
        "d_ee_reachgoal_0": float(np.linalg.norm(e0 - rgoal)),
        "align_0": align0,
        "goal_behind_cube_0": float(np.dot(unit(goal - o0), unit(o0 - e0))),
        "cube_goal_vs_reach_0": float(np.linalg.norm(goal - rgoal)),
        # --- accumulated up to t ---
        "d_cube_goal_t": float(d_goal_t),
        "cube_progress_t": float(d_goal_0 - d_goal_t),
        "cube_moved_t": float(np.linalg.norm(ot - o0)),
        "d_ee_cube_t": float(np.linalg.norm(ot - et)),
        "align_t": align,
        "ee_speed_t": float(
            np.linalg.norm(ee[min(t, len(ee) - 1)] - ee[max(0, t - 1)])
        ),
        "cube_speed_t": float(np.linalg.norm(ot - obj[max(0, t - 1)])),
    }
    if t >= 1:
        a = act[:t]
        s = sc[:t][np.isfinite(sc[:t])]
        f["score_max_t"] = float(s.max()) if s.size else np.nan
        f["score_mean_t"] = float(s.mean()) if s.size else np.nan
        f["act_sat_t"] = float((np.abs(a) > 0.95).mean())
        f["act_norm_t"] = float(np.linalg.norm(a, axis=1).mean())
        # how much of the commanded motion actually happened
        cmd = np.linalg.norm(a, axis=1).sum() * 0.05
        got = np.linalg.norm(np.diff(ee[: t + 1], axis=0), axis=1).sum()
        f["motion_efficiency_t"] = float(got / cmd) if cmd > 1e-9 else np.nan
    else:
        for k in (
            "score_max_t",
            "score_mean_t",
            "act_sat_t",
            "act_norm_t",
            "motion_efficiency_t",
        ):
            f[k] = np.nan
    return f


def logistic(Xtr, ytr, Xte, iters=4000, lr=0.2, l2=1e-3):
    """Plain gradient-descent logistic regression, standardised, numpy only."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A = np.nan_to_num(A)
    B = np.nan_to_num(B)
    w = np.zeros(A.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w + b, -30, 30)))
        gw = A.T @ (p - ytr) / len(ytr) + l2 * w
        gb = float((p - ytr).mean())
        w -= lr * gw
        b -= lr * gb
    return B @ w + b, w, mu, sd


def main():
    z = np.load(NPZ)
    seeds = sorted(
        {int(k.split("/")[1][4:]) for k in z.files if k.startswith("cheap/")}
    )
    failed = np.array([not bool(z[f"cheap/seed{s}/success"]) for s in seeds])
    print(f"{len(seeds)} cheap episodes, {failed.sum()} failures ({failed.mean():.1%})")
    print("all features are causal (rows 0..t only) and deployable (no outcome info)\n")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(seeds))
    tr, te = perm[: len(perm) // 2], perm[len(perm) // 2 :]
    out = {}

    for t in DECISION_STEPS:
        rows = [features(z, s, t) for s in seeds]
        names = [k for k in rows[0]]
        X = np.array([[r[k] for k in names] for r in rows], float)
        print(
            f"{'=' * 74}\nDECISION STEP t = {t}"
            f"{'   (reset geometry only, before any action)' if t == 0 else ''}\n{'=' * 74}"
        )
        singles = {}
        for j, k in enumerate(names):
            a = auc(X[:, j], failed)
            if np.isfinite(a):
                singles[k] = a
        for k, a in sorted(singles.items(), key=lambda x: -abs(x[1] - 0.5))[:8]:
            lift = abs(a - 0.5)
            tag = (
                "none"
                if lift < 0.03
                else "weak"
                if lift < 0.08
                else "moderate"
                if lift < 0.15
                else "STRONG"
            )
            print(f"  {k:<24}{a:>7.3f}   {tag}")
        # combined, fit on half the seeds and scored on the other half
        keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).mean() > 0.9]
        Xc = np.nan_to_num(X[:, keep])
        pred, w, _, _ = logistic(Xc[tr], failed[tr].astype(float), Xc[te])
        a_comb = auc(pred, failed[te])
        best = max(singles.items(), key=lambda x: abs(x[1] - 0.5))
        out[f"t{t}"] = {
            "singles": singles,
            "combined_heldout_auc": a_comb,
            "best_single": best[0],
            "best_single_auc": best[1],
        }
        print(f"  {'-' * 70}")
        print(
            f"  COMBINED (logistic, fit on 250 seeds, scored on the held-out 250)"
            f"  AUC {a_comb:.3f}"
        )
        top = sorted(zip([names[j] for j in keep], w), key=lambda x: -abs(x[1]))[:4]
        print(f"  strongest weights: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    base = 0.543  # the arbiter's own score at step 4, measured
    for t in DECISION_STEPS:
        c = out[f"t{t}"]["combined_heldout_auc"]
        print(
            f"  t={t}: best single {out[f't{t}']['best_single_auc']:.3f} "
            f"({out[f't{t}']['best_single']}),  combined {c:.3f}"
        )
    print(f"\n  the arbiter's current trigger at its firing step: AUC {base:.3f}")
    (HERE / "results" / "early_score_search.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/early_score_search.json")


if __name__ == "__main__":
    main()
