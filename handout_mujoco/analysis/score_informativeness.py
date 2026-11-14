"""Does any candidate score predict EPISODE FAILURE at the moment a trigger would fire?

This is the gate that decides whether an arbiter is worth building. In the prior PyBullet
system the compliance residual had AUC 0.543 at the firing step, and CP arbitration came
out statistically indistinguishable from random escalation (+0.026, CI -0.014..+0.066,
p = 0.24). An overlay built on an uninformative score shows escalations at essentially
random moments while looking convincing.

METHOD -- causal by construction:
  For each episode and phase k, the statistic at horizon h is
        S = max over the first h steps of phase k   (of the candidate, where defined)
  which is exactly what a trigger watching the first h steps could have seen. No
  hindsight, no whole-episode aggregate, no quantity that depends on the outcome.
  AUC is then computed for S against "this episode ultimately FAILED".

CIRCULAR FEATURES, EXCLUDED AND WHY:
  episode length      -- a failed push episode runs to the 50-step budget by definition,
                         so length is a restatement of the label, not a predictor
  final d_push        -- the success test is a threshold on this exact quantity
  any success flag    -- the label itself
  cube_fell           -- a terminal condition, known only once it has happened

SELECTION HYGIENE: four candidates are screened, so the best-of-four AUC on the screening
data is optimistically biased. Calibration seeds are split in half: candidates are ranked
on 300,000-300,499 and the selected candidate's AUC is RE-REPORTED on the untouched half
300,500-300,999. The gate is judged on the held-out half.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python analysis/score_informativeness.py
"""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))

from cheap_policy import load as load_cheap
from composed import PUSH, REACH, ComposedEnv

TRAIN = ROOT / "results" / "mujoco_train"
OUT = TRAIN / "score_informativeness"
CAL_LO, CAL_HI = 300_000, 301_000
SPLIT = 300_500          # [CAL_LO, SPLIT) screening | [SPLIT, CAL_HI) held out
HORIZONS = (2, 4, 8, None)   # None = the whole phase

CANDIDATES = {
    "a_compliance_residual": (
        "||cmd_disp - achieved_disp|| / ||cmd_disp||, motion-floor masked",
        "Causal: both terms are known one step after the command is issued. Deployable: "
        "needs only the commanded action and the measured EE displacement.",
    ),
    "b_ensemble_disagreement": (
        "mean over action dims of the std across a 3-member cheap-MLP ensemble",
        "Causal: computed from the observation alone, before the action is taken. "
        "Deployable: 3 forward passes of a 755-parameter net, 0.02 ms each.",
    ),
    "c_predicted_action_std": (
        "mean predicted per-dimension std from a heteroscedastic (Gaussian-head) cheap policy",
        "Causal: a property of the current observation, available before acting. "
        "Deployable: one forward pass, no ensemble.",
    ),
    "d_hand_to_cube_distance": (
        "||ee - cube||",
        "Causal: a measured geometric quantity at the current step. Deployable trivially. "
        "Meaningful in PUSH only -- in REACH the arm is not supposed to be at the cube.",
    ),
}


def auc(scores, labels):
    """Mann-Whitney AUC. labels: 1 = episode FAILED. Higher score should mean failure."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), n1, n0
    r = np.empty(len(s), float)
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    i = 0
    while i < len(sr):                      # average ranks within ties
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), n1, n0


def collect(seeds, ens_tags=None, mask_dims=None):
    """Run the cheap policy and record every candidate at every step, with the phase.

    `ens_tags` swaps the ensemble used for the disagreement score without changing the
    ACTING policy -- the trigger is a monitor, the controller stays the same.
    """
    primary, _ = load_cheap(TRAIN / "cheap_policy_16x16.pt")
    if ens_tags is None:
        ens = [primary] + [load_cheap(TRAIN / f"cheap_policy_ens{i}.pt")[0] for i in (1, 2)]
    else:
        ens = [load_cheap(TRAIN / f"cheap_policy_{t}.pt")[0] for t in ens_tags]
    gauss, _ = load_cheap(TRAIN / "cheap_policy_gauss.pt")
    env = ComposedEnv()
    episodes = []
    for sd in seeds:
        env.reset(sd)
        steps = []
        info = None
        for _ in range(100):
            o = env.obs26()
            a = primary.act(o)                      # the ACTING policy, never masked
            o_trig = o.copy()
            if mask_dims:
                o_trig[list(mask_dims)] = 0.0       # what the TRIGGER is allowed to see
            acts = np.stack([m.act(o_trig) for m in ens])
            _, g_std = gauss.act_and_std(o)
            phase = env.phase
            _, term, trunc, info = env.step(a)
            steps.append(
                {
                    "phase": int(phase),
                    "a_compliance_residual": info["score"] if info["scorable"] else np.nan,
                    "b_ensemble_disagreement": float(acts.std(axis=0).mean()),
                    "c_predicted_action_std": float(g_std),
                    "d_hand_to_cube_distance": float(info["d_hand"]),
                }
            )
            if term or trunc:
                break
        episodes.append({"seed": sd, "failed": not bool(info["is_success"]),
                         "reached": bool(info["reached"]), "d0": float(info["d0"]),
                         "steps": steps})
    return episodes


def statistic(ep, cand, phase, h):
    """max of `cand` over the first h steps of `phase` -- what a trigger could have seen."""
    v = [s[cand] for s in ep["steps"] if s["phase"] == phase]
    v = v if h is None else v[:h]
    v = [x for x in v if np.isfinite(x)]
    return max(v) if v else np.nan


def table(episodes, phase):
    rows = {}
    eps = [e for e in episodes if any(s["phase"] == phase for s in e["steps"])]
    for cand in CANDIDATES:
        rows[cand] = {}
        for h in HORIZONS:
            S = [statistic(e, cand, phase, h) for e in eps]
            y = [e["failed"] for e in eps]
            a, n_fail, n_ok = auc(S, y)
            rows[cand][str(h) if h else "all"] = {
                "auc": a, "n_failed": n_fail, "n_succeeded": n_ok,
                "n_defined": int(np.isfinite(np.asarray(S, float)).sum()),
            }
    return rows, len(eps)


def main():
    seeds = list(range(CAL_LO, CAL_HI))
    print(f"running the cheap policy on calibration seeds {CAL_LO:,}-{CAL_HI - 1:,} ...")
    episodes = collect(seeds)
    screen = [e for e in episodes if e["seed"] < SPLIT]
    held = [e for e in episodes if e["seed"] >= SPLIT]

    out = {
        "cheap_policy": "cheap_policy_16x16.pt (755 actor params)",
        "calibration_seeds": [CAL_LO, CAL_HI - 1],
        "screening_seeds": [CAL_LO, SPLIT - 1],
        "heldout_seeds": [SPLIT, CAL_HI - 1],
        "n_episodes": len(episodes),
        "overall_failure_rate": float(np.mean([e["failed"] for e in episodes])),
        "reach_success_rate": float(np.mean([e["reached"] for e in episodes])),
        "candidates": {k: {"definition": v[0], "causality": v[1]} for k, v in CANDIDATES.items()},
        "excluded_circular": {
            "episode_length": "a failed push runs to the 50-step budget by definition",
            "final_d_push": "the success test is a threshold on this exact quantity",
            "success_flag": "the label itself",
            "cube_fell": "a terminal condition, known only after it happens",
        },
        "gate": "selected score must reach AUC >= 0.65 within the first 8 steps of the "
                "phase where escalation actually fires",
    }
    for name, phase in (("reach", REACH), ("push", PUSH)):
        rows, n = table(screen, phase)
        out[f"{name}_screening"] = {"n_episodes": n, "auc": rows}
        rows_h, n_h = table(held, phase)
        out[f"{name}_heldout"] = {"n_episodes": n_h, "auc": rows_h}

    # the phase where escalation actually fires is the one that produces the failures
    push_fail = out["push_screening"]["auc"]["a_compliance_residual"]["8"]["n_failed"]
    reach_fail = out["reach_screening"]["auc"]["a_compliance_residual"]["8"]["n_failed"]
    out["failures_by_phase_screening"] = {"reach_phase_episodes_failed": reach_fail,
                                          "push_phase_episodes_failed": push_fail}

    best, best_auc = None, -1.0
    for cand in CANDIDATES:
        v = out["push_screening"]["auc"][cand]["8"]["auc"]
        if np.isfinite(v) and v > best_auc:
            best, best_auc = cand, v
    held_auc = out["push_heldout"]["auc"][best]["8"]["auc"]
    out["selected_on_screening"] = {"candidate": best, "push_auc_at_8": best_auc}
    out["selected_heldout_push_auc_at_8"] = held_auc
    out["gate_passed"] = bool(np.isfinite(held_auc) and held_auc >= 0.65)

    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2))

    def show(title, blk):
        print(f"\n{title}  (n={blk['n_episodes']})")
        print(f"  {'candidate':<28} {'h=2':>8} {'h=4':>8} {'h=8':>8} {'all':>8}")
        for cand in CANDIDATES:
            r = blk["auc"][cand]
            f = lambda k: (f"{r[k]['auc']:.3f}" if np.isfinite(r[k]['auc']) else "  n/a")
            print(f"  {cand:<28} {f('2'):>8} {f('4'):>8} {f('8'):>8} {f('all'):>8}")

    print(f"\ncheap policy failure rate {out['overall_failure_rate']:.3f}, "
          f"reach success {out['reach_success_rate']:.3f}")
    print(f"episodes failing with the failure in reach phase: {reach_fail}, in push phase: {push_fail}")
    show("REACH phase, screening half", out["reach_screening"])
    show("PUSH phase, screening half", out["push_screening"])
    show("PUSH phase, HELD-OUT half (what the gate is judged on)", out["push_heldout"])
    print(f"\nselected on screening: {best}  (push AUC@8 = {best_auc:.3f})")
    print(f"HELD-OUT push AUC@8   = {held_auc:.3f}")
    print(f"GATE 6 (>= 0.65): {'PASS' if out['gate_passed'] else 'FAIL'}")
    print(f"\nwritten: {OUT.relative_to(ROOT)}.json")


if __name__ == "__main__":
    main()
