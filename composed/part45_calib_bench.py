"""PARTS 4-5 -- calibration and four-way benchmark on a composed task (round-1 methodology, unchanged).

Refuses to run unless part3_horizon_rescue.json says proceed_to_part4 == True.

Part 4: phases = the composition's two phases (K = 2, reach / phase-B task), defined causally
by the phase switch.  Score s(t) = ||clip(a)*0.05 - (ee(t+1)-ee(t))|| / ||clip(a)*0.05||,
MOTION_FLOOR mask (scp.scoring).  Calibration on successful, non-degenerate CHEAP episodes;
S_k = max scorable s in phase k; thresholds via scp.calibrate.calibrate_phases (Bonferroni).
Part 5: 500 held-out paired seeds; cheap, expert, arbitrated (scp.arbitration hysteresis,
hold 8, chunk 1), random escalation re-tuned on separate tune seeds to the arbiter's realised
expensive fraction.  Wilson CIs, paired bootstrap, exact McNemar.  alpha_esc 0.30 primary.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

import compose as K
import common as C
from scp.calibrate import calibrate_phases, feasibility_report

P3 = json.load(open(os.path.join(K.RESULTS, "part3_horizon_rescue.json")))
OVERRIDE = os.environ.get("GATE_OVERRIDE") == "1"
if not P3["proceed_to_part4"] and not OVERRIDE:
    sys.exit("Part 3 gate did not pass for %s -- calibration/benchmark not run." % K.TASK)
GATE_NOTE = None if P3["proceed_to_part4"] else (
    "USER-AUTHORISED GATE OVERRIDE (deterministic-reset Part 3): G3.1 %s (median tau_irr %.1f, IQR %.1f); "
    "G3.2 %s (checks %s).  Parts 4-5 run with this override disclosed; the user authorised running them "
    "after the earlier (shared-env, noisy) Part 3 missed G3.1 at the boundary." % (
        "passed" if P3["G3_1"]["pass"] else "FAILED", P3["G3_1"]["tau_irr"]["median"], P3["G3_1"]["tau_irr_iqr"],
        "passed" if P3["G3_2"]["pass"] else "FAILED", P3["G3_2"]["checks"]))
print("GATE:", GATE_NOTE or "passed without override")

SEEDS = {"calib": 210_000, "tune": 220_000, "bench": 230_000}
N_CAL, N_TUNE, N_BENCH = int(os.environ.get("N_CAL", 600)), int(os.environ.get("N_TUNE", 200)), int(os.environ.get("N_BENCH", 500))
ALPHA, ALPHA_ESC = 0.10, (0.30, 0.20, 0.40)
D = json.load(open(os.path.join(K.RESULTS, "part4a_distill.json")))
cheap = K.Cheap(os.path.join(K.MODELS, D["budgets"][str(D["chosen_budget"])]["path"]))
env = K.Composed()
ex = K.Experts()


def run(rule, seeds, kw=None, rng_seed=0, record=False):
    rng = np.random.default_rng(rng_seed)
    return [K.run_episode(env, s, cheap, ex, rule, kw, rng, record=record) for s in seeds]


REPLAY = {}


def keep_replay(name, recs):
    """Per-episode kinematic replay data + arbitration trace, for the MuJoCo animation handout."""
    for r in recs:
        p = "%s/seed%d/" % (name, r["seed"])
        for k, v in r["traj"].items():
            REPLAY[p + k] = v
        for k in ("modes", "labels", "score", "flags", "actions"):
            REPLAY[p + k] = np.asarray(r[k])
        REPLAY[p + "success"] = np.array(r["success"])


# ------------------------------------------------------------------ Part 4
cal = run("cheap", [SEEDS["calib"] + i for i in range(N_CAL)])
succ = [r for r in cal if r["success"]]
agg = []
for r in succ:
    a = {}
    for k, p in enumerate(K.PHASES):
        sel = (r["labels"] == k) & r["mask"]
        a[p] = float(r["score"][sel].max()) if sel.any() else float("nan")
    agg.append(a)
kept = [a for a in agg if all(np.isfinite(v) for v in a.values())]
ps = {p: [a[p] for a in kept] for p in K.PHASES}
safe = calibrate_phases(ps, ALPHA)
part4 = {"n_cal_episodes": N_CAL, "cheap_successes": len(succ), "degenerate_empty_phase": len(agg) - len(kept),
         "kept": len(kept), "empty_phase_after_filter": 0, "feasibility": feasibility_report(ALPHA, 2, len(kept)),
         "frac_steps_below_floor": float(1 - np.mean(np.concatenate([r["mask"] for r in cal]))),
         "q_hat": {p: {"value": t.value, "rank": t.rank, "n": t.n, "feasible": t.feasible, "at_sample_max": t.rank == t.n}
                   for p, t in safe.thresholds.items()}, "escalation": {}}
for ae in ALPHA_ESC:
    ce = calibrate_phases(ps, ae)
    part4["escalation"][str(ae)] = {"q_esc": {p: t.value for p, t in ce.thresholds.items()},
                                    "feasible": ce.feasible,
                                    "q_esc_below_q_hat": all(ce.thresholds[p].value < safe.thresholds[p].value for p in K.PHASES)}
print(json.dumps(part4, indent=1, default=bool))
assert safe.feasible


# ------------------------------------------------------------------ Part 5
def summary(recs):
    k, n = int(sum(r["success"] for r in recs)), len(recs)
    first = [int(np.flatnonzero(r["modes"])[0]) for r in recs if r["modes"].any()]
    steps = sum(r["T"] for r in recs)
    return {"n": n, "success": k, "success_rate": k / n, "wilson95": C.wilson(k, n),
            "frac_expensive_mean_over_episodes": float(np.mean([r["frac_expensive"] for r in recs])),
            "frac_expensive_pooled_steps": float(sum(r["modes"].sum() for r in recs) / steps),
            "mean_episode_len": steps / n, "frac_episodes_escalated": len(first) / n,
            "first_escalation_step_median": float(np.median(first)) if first else None,
            "switches_mean": float(np.mean([r["switches"] for r in recs])),
            "mean_episode_score": float(np.mean([r["score"][r["mask"]].mean() for r in recs if r["mask"].any()]))}


def paired(a, b):
    sa, sb = np.array([r["success"] for r in a]), np.array([r["success"] for r in b])
    n10, n01 = int(np.sum(sa & ~sb)), int(np.sum(~sa & sb))
    d = sa.astype(float) - sb.astype(float)
    rng = np.random.default_rng(123)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return {"diff": float(d.mean()), "boot95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "a_only": n10, "b_only": n01, "mcnemar_exact_p": float(binomtest(n10, n10 + n01, .5).pvalue) if n10 + n01 else 1.0}


bench = [SEEDS["bench"] + i for i in range(N_BENCH)]
tune = [SEEDS["tune"] + i for i in range(N_TUNE)]
cheap_r, exp_r = run("cheap", bench, record=True), run("expert", bench, record=True)
keep_replay("cheap", cheap_r); keep_replay("expert", exp_r)
res = {"doc": __doc__, "gate_override": GATE_NOTE, "part4": part4, "conditions": {"1_cheap": summary(cheap_r), "2_expert": summary(exp_r)},
       "comparisons": {}, "tuning": {}}
print("cheap %.3f expert %.3f" % (res["conditions"]["1_cheap"]["success_rate"], res["conditions"]["2_expert"]["success_rate"]))
for ae in ALPHA_ESC:
    tag = "a%.2f" % ae
    kw = {"q_esc": part4["escalation"][str(ae)]["q_esc"], "hold": C.arb.DEFAULT_HOLD}
    arb_r = run("arb", bench, kw, record=(ae == 0.30))
    s_arb = summary(arb_r)
    target = s_arb["frac_expensive_mean_over_episodes"]
    lo, hi, hist = 1e-4, 1.0, []
    for it in range(9):
        mid = float(np.sqrt(lo * hi))
        f = float(np.mean([r["frac_expensive"] for r in run("random", tune, dict(kw, rate=mid), 1000 + it)]))
        hist.append({"rate": mid, "frac": f})
        lo, hi = (mid, hi) if f < target else (lo, mid)
    rate = min(hist, key=lambda h: abs(h["frac"] - target))["rate"]
    rnd_r = run("random", bench, dict(kw, rate=rate), 7, record=(ae == 0.30))
    if ae == 0.30:
        keep_replay("arbitrated_a0.30", arb_r); keep_replay("random_a0.30", rnd_r)
        np.savez_compressed(os.path.join(K.RESULTS, "replay_trajectories_a0.30.npz"), **REPLAY)
    s_rnd = summary(rnd_r)
    res["conditions"]["3_arbitrated_" + tag] = s_arb
    res["conditions"]["4_random_" + tag] = s_rnd
    res["tuning"][tag] = {"target": target, "rate": rate, "bisection": hist}
    c = {"arb_vs_cheap": paired(arb_r, cheap_r), "arb_vs_expert": paired(arb_r, exp_r), "arb_vs_random": paired(arb_r, rnd_r),
         "cost_match": {"arb": target, "random": s_rnd["frac_expensive_mean_over_episodes"]}}
    c["beats_random_significantly"] = bool(c["arb_vs_random"]["boot95"][0] > 0 and c["arb_vs_random"]["mcnemar_exact_p"] < 0.05)
    res["comparisons"][tag] = c
    print(tag, "arb %.3f (fexp %.3f)  random %.3f (fexp %.3f)" % (s_arb["success_rate"], target, s_rnd["success_rate"],
                                                                s_rnd["frac_expensive_mean_over_episodes"]),
          json.dumps(c["arb_vs_random"]), flush=True)
    json.dump(res, open(os.path.join(K.RESULTS, "part45_benchmark.json"), "w"), indent=2, default=bool)
json.dump({c: [{k: r[k] for k in ("seed", "success", "T", "frac_expensive", "switches", "fail_phase")} for r in recs]
           for c, recs in (("cheap", cheap_r), ("expert", exp_r))}, open(os.path.join(K.RESULTS, "part45_episodes_baselines.json"), "w"))
print("done")


