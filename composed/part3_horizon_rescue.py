"""PART 3 -- horizon diagnostic + rescue curve on the composed task, BEFORE any calibration.

Pre-registered gates (written before running):

  Rescue curve: 100 seeds on which the cheap policy fails (first 100 failures among the 500
  horizon-diagnostic episodes, seeds 160000..).  Expert pair (unmodified) takes permanent
  control from step t, t in {0, 5, ..., 95}; phase budgets unchanged.  R(t) = success rate.

  Per-seed irreversibility step tau_irr = first grid t at which handover no longer rescues,
  over seeds rescued at t=0 (never-failing seeds censored at 100).

  G3.1 (horizon): PASS iff median(tau_irr) > 15 (15% of the 100-step horizon)
                  AND IQR(tau_irr) >= 10 steps.
  G3.2 (rescue):  with D = R(0) - R(95):
                  R(0) >= 0.8 x composed-expert rate (slide: 0.70; push: computed from part2_compose.json;
                        a flat 0.80 would test the expert, not the window)
                  AND D >= 0.30
                  AND the largest drop between consecutive grid points <= 0.5 * D
                  AND >= 3 grid points with t >= 10 inside the middle band
                      [R(95) + 0.25 D, R(0) - 0.25 D].
                  (Round 1's curve 0.96/0.60/0.63/.../0.50 fails: one 0.36 drop of D=0.46.)
  Proceed to calibration/benchmark only if BOTH pass.
"""
import json
import os

import numpy as np

import compose as K
import common as C

N_DIAG = 500
SEED0 = 160_000
TS = tuple(range(0, 100, 5))

D = json.load(open(os.path.join(K.RESULTS, "part4a_distill.json")))
cheap = K.Cheap(os.path.join(K.MODELS, D["budgets"][str(D["chosen_budget"])]["path"]))
env = K.Composed()
ex = K.Experts()

# ---------------------------------------------------------------- 3.1 horizon diagnostic
diag = [K.run_episode(env, SEED0 + i, cheap, None, "cheap") for i in range(N_DIAG)]
succ = [r for r in diag if r["success"]]
fail = [r for r in diag if not r["success"]]


def dist(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.quantile(v, .1)), "p90": float(np.quantile(v, .9)), "min": float(v.min()),
            "max": float(v.max())} if v.size else {"n": 0}


horizon = {
    "n": N_DIAG, "cheap_success_rate": len(succ) / N_DIAG, "wilson95": C.wilson(len(succ), N_DIAG),
    "phaseA_success_rate": float(np.mean([r["phaseA_success"] for r in diag])),
    "first_success_step_phaseA": dist([r["t_switch"] for r in diag if r["t_switch"]]),
    "episode_success_step": dist([r["T"] for r in succ]),
    "phaseB_steps_to_success": dist([r["T"] - r["t_switch"] for r in succ]),
    "failure_terminal_step": dist([r["T"] for r in fail]),
    "failure_phase_counts": {p: int(sum(r["fail_phase"] == p for r in fail)) for p in K.PHASES},
    "episode_length_all": dist([r["T"] for r in diag]),
}
print(json.dumps(horizon, indent=1))

# ---------------------------------------------------------------- 3.2 rescue curve
pool = [r["seed"] for r in fail][:100]
res = {t: np.array([K.run_episode(env, s, cheap, ex, "handover", {"t": t})["success"] for s in pool]) for t in TS}
R = {t: float(v.mean()) for t, v in res.items()}
for t in TS:
    print("t=%2d R=%.2f" % (t, R[t]))

M = np.stack([res[t] for t in TS], 1)          # seeds x grid
rescued0 = M[:, 0]
tau = []
for row in M[rescued0]:
    idx = np.flatnonzero(~row)
    tau.append(TS[idx[0]] if idx.size else 100)
tau = np.array(tau, float)
g31 = {"tau_irr": dist(tau), "tau_irr_iqr": float(np.quantile(tau, .75) - np.quantile(tau, .25)) if tau.size else None,
       "n_rescued_at_t0": int(rescued0.sum()), "frac_non_monotone_seeds": float(np.mean(
           [np.any(np.diff(row.astype(int)) > 0) for row in M]))}
g31["pass"] = bool(tau.size and np.median(tau) > 15 and g31["tau_irr_iqr"] >= 10)

R0, Rl = R[0], R[95]
Dd = R0 - Rl
EXP_RATE = json.load(open(os.path.join(K.RESULTS, "part2_compose.json")))["composed_success_rate"]
R0_MIN = round(0.8 * EXP_RATE, 2)   # slide: 0.8 x 0.865 = 0.69 (reported as 0.70); same rule for every composition
drops = [R[TS[i]] - R[TS[i + 1]] for i in range(len(TS) - 1)]
band = (Rl + 0.25 * Dd, R0 - 0.25 * Dd)
mid_pts = [t for t in TS if t >= 10 and band[0] <= R[t] <= band[1]]
g32 = {"R0": R0, "R95": Rl, "D": Dd, "max_consecutive_drop": float(max(drops)),
       "max_drop_at": f"{TS[int(np.argmax(drops))]}->{TS[int(np.argmax(drops)) + 1]}",
       "middle_band": band, "middle_points_t>=10": mid_pts,
       "R0_min": R0_MIN if K.TASK != "slide" else 0.70,
       "checks": {"R0>=R0_min": R0 >= (R0_MIN if K.TASK != "slide" else 0.70), "D>=0.30": Dd >= 0.30, "max_drop<=0.5D": max(drops) <= 0.5 * Dd,
                  ">=3 middle points": len(mid_pts) >= 3}}
g32["pass"] = bool(all(g32["checks"].values()))

out = {"protocol": __doc__, "cheap_budget": D["chosen_budget"], "horizon_diagnostic": horizon,
       "rescue_pool_seeds": pool, "rescue_curve": {str(t): {"rate": R[t], "wilson95": C.wilson(int(res[t].sum()), len(pool))}
                                                   for t in TS},
       "G3_1": g31, "G3_2": g32, "proceed_to_part4": bool(g31["pass"] and g32["pass"])}
json.dump(out, open(os.path.join(K.RESULTS, "part3_horizon_rescue.json"), "w"), indent=2, default=bool)
np.savez_compressed(os.path.join(K.RESULTS, "part3_rescue_matrix.npz"), M=M, seeds=np.array(pool), ts=np.array(TS))
print("G3.1", json.dumps(g31, default=bool)); print("G3.2", json.dumps(g32, default=bool))
print("PROCEED:", out["proceed_to_part4"])

