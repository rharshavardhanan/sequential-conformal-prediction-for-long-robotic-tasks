"""Paired evaluation of all arms on identical evaluation seeds 400,000-400,199.

Arms: expert-only | cheap-only | arbitrated (alpha_esc 0.20/0.30/0.40) |
      random escalation at a MATCHED firing rate (the control that decides whether the
      TIMING of escalation carries any information, or only the amount of expert compute)

Every arm runs the same seeds, so success can be compared paired (McNemar on discordant
episodes) rather than as two independent proportions.

ALL TIMING ON CPU AT BATCH 1. Measured: CPU 0.106 ms vs MPS 2.658 ms p50 for this actor.
The device ranking inverts with network size, so timing the 541k-parameter expert on one
device and the 755-parameter cheap policy on another would turn a device switch into a
fake compute saving.
"""

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cheap_policy import load as load_cheap
from composed import ComposedEnv, ComposedExpert, rollout
from envs import PushOnly
from mujoco_arbiter import Arbiter, phase_q
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

TRAIN = ROOT / "results" / "mujoco_train"
OUT = TRAIN / "arbitration_eval"
EVAL_LO, EVAL_HI = 400_000, 400_200
ALPHAS = (0.20, 0.30, 0.40)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a, b):
    """Exact two-sided McNemar on paired binary outcomes."""
    from math import comb
    n01 = int(np.sum(~a & b))   # b succeeded, a failed
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"n01": n01, "n10": n10, "p_value": float(p)}


def run_arm(name, policy_factory, seeds, env):
    rows, t_all = [], []
    for sd in seeds:
        pol, stats_fn = policy_factory()
        t0 = time.perf_counter()
        info, steps = rollout(env, pol, sd)
        dt = time.perf_counter() - t0
        st = stats_fn() if stats_fn else {}
        rows.append({"seed": sd, "success": bool(info["is_success"]),
                     "reached": bool(info["reached"]), "steps": steps,
                     "wall_s": dt, **st})
        t_all.append(dt / max(steps, 1))
    ok = np.array([r["success"] for r in rows])
    reached = np.array([r["reached"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    out = {
        "arm": name, "n": len(rows),
        "composed_success": float(ok.mean()),
        "wilson_ci95": list(wilson(int(ok.sum()), len(ok))),
        "reach_success": float(reached.mean()),
        "push_success_given_reach": float(ok[reached].mean()) if reached.any() else 0.0,
        "mean_ep_len": float(steps.mean()),
        "expert_calls_per_ep": float(np.mean([r.get("expert_calls", 0) for r in rows])),
        "expert_step_pct": 100.0 * float(np.mean([r.get("expert_step_frac", 0.0) for r in rows])),
        "switches_per_ep": float(np.mean([r.get("switches", 0) for r in rows])),
        "first_escalation_step_median": float(np.median(
            [r["first_escalation_step"] for r in rows if r.get("first_escalation_step")]
        )) if any(r.get("first_escalation_step") for r in rows) else None,
        "trigger_fire_rate_among_cheap_steps": float(np.mean(
            [r.get("trigger_fire_rate_among_cheap_steps", 0.0) for r in rows])),
        "ms_per_step_cpu": 1000.0 * float(np.mean(t_all)),
        "total_compute_ms_per_ep": 1000.0 * float(np.mean([r["wall_s"] for r in rows])),
        "_success_vec": ok,
    }
    return out, rows


def main():
    seeds = list(range(EVAL_LO, EVAL_HI))
    cal = json.loads((TRAIN / "calibration.json").read_text())
    push_model = TQC.load(TRAIN / "push_expert" / "seed0" / "best_model.zip",
                          env=DummyVecEnv([PushOnly]), device="cpu")
    expert = ComposedExpert(push_model)
    cheap, _ = load_cheap(TRAIN / "cheap_policy_16x16.pt")
    ens = [cheap] + [load_cheap(TRAIN / f"cheap_policy_ens{i}.pt")[0] for i in (1, 2)]
    env = ComposedEnv()

    arms = {}
    print(f"evaluating on seeds {EVAL_LO:,}-{EVAL_HI - 1:,}, all timing CPU batch-1\n")

    a, _ = run_arm("expert_only", lambda: ((lambda e: expert.act(e)), None), seeds, env)
    arms["expert_only"] = a
    a, _ = run_arm("cheap_only", lambda: ((lambda e: (cheap.act(e.obs26()), "cheap")), None),
                   seeds, env)
    arms["cheap_only"] = a

    for al in ALPHAS:
        q = phase_q(cal, al)

        def factory(q=q):
            arb = Arbiter(cheap, ens, expert, q)
            return arb.act, arb.stats

        a, rows = run_arm(f"arbitrated_a{al:.2f}", factory, seeds, env)
        arms[f"arbitrated_a{al:.2f}"] = a

        # MATCH ON THE REALISED OPERATING POINT, not on the nominal knob. Matching the
        # per-step trigger probability leaves the two arms spending different amounts of
        # expert compute (measured: 37.6% vs 49.6% at alpha=0.30), and the comparison then
        # reports that gap rather than the value of the timing. Bisect the Bernoulli rate
        # until random spends the same expert-step share as the arbiter.
        target = a["expert_step_pct"]
        lo, hi, best, b = 0.0, 1.0, None, None
        for _ in range(9):
            mid = 0.5 * (lo + hi)

            def rfactory(rate=mid):
                arb = Arbiter(cheap, ens, expert, q, random_rate=rate,
                              rng=np.random.default_rng(1234))
                return arb.act, arb.stats

            cand, _ = run_arm(f"random_a{al:.2f}", rfactory, seeds, env)
            if best is None or abs(cand["expert_step_pct"] - target) < abs(best - target):
                best, b = cand["expert_step_pct"], cand
            if cand["expert_step_pct"] < target:
                lo = mid
            else:
                hi = mid
            if abs(cand["expert_step_pct"] - target) < 0.75:
                b = cand
                break
        b["matched_to_expert_step_pct"] = target
        b["match_error_pp"] = abs(b["expert_step_pct"] - target)
        arms[f"random_a{al:.2f}"] = b
        print(f"  alpha {al:.2f}: matched random to {target:.1f}% expert steps -> "
              f"achieved {b['expert_step_pct']:.1f}% (err {b['match_error_pp']:.2f} pp)")

    ex, ch = arms["expert_only"]["composed_success"], arms["cheap_only"]["composed_success"]
    out = {
        "eval_seeds": [EVAL_LO, EVAL_HI - 1],
        "timing_device": "cpu, batch 1 (see spec section 13)",
        "score": "b_ensemble_disagreement (Gate 6)",
        "K": 2,
        "handback_steps": 8,
        "arms": {}, "comparisons": {},
    }
    for k, v in arms.items():
        out["arms"][k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        if k.startswith("arbitrated"):
            out["arms"][k]["gain_recovered"] = (
                (v["composed_success"] - ch) / (ex - ch) if ex != ch else float("nan"))
    for al in ALPHAS:
        c, d = arms[f"arbitrated_a{al:.2f}"], arms[f"random_a{al:.2f}"]
        out["comparisons"][f"a{al:.2f}_arbitrated_vs_random"] = {
            **mcnemar(c["_success_vec"], d["_success_vec"]),
            "arbitrated_success": c["composed_success"],
            "random_success": d["composed_success"],
            "delta": c["composed_success"] - d["composed_success"],
            "matched_expert_step_pct": [c["expert_step_pct"], d["expert_step_pct"]],
        }
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2))

    hdr = f"{'arm':<20} {'success':>16} {'reach':>6} {'eplen':>6} {'calls':>6} {'exp%':>6} {'sw':>5} {'1stEsc':>7} {'ms/step':>8}"
    print(hdr); print("-" * len(hdr))
    for k, v in out["arms"].items():
        lo, hi = v["wilson_ci95"]
        fe = v["first_escalation_step_median"]
        print(f"{k:<20} {v['composed_success']:.3f}[{lo:.2f},{hi:.2f}] {v['reach_success']:>6.3f} "
              f"{v['mean_ep_len']:>6.1f} {v['expert_calls_per_ep']:>6.2f} {v['expert_step_pct']:>6.1f} "
              f"{v['switches_per_ep']:>5.2f} {(f'{fe:.0f}' if fe else '—'):>7} {v['ms_per_step_cpu']:>8.3f}")
    print(f"\nexpert {ex:.3f}   cheap {ch:.3f}")
    for k, v in out["arms"].items():
        if "gain_recovered" in v:
            print(f"  {k}: gain_recovered = {v['gain_recovered']:+.3f}")
    print("\nPAIRED arbitrated vs random-at-matched-rate (exact McNemar):")
    for k, c in out["comparisons"].items():
        print(f"  {k}: arb {c['arbitrated_success']:.3f} vs rand {c['random_success']:.3f}  "
              f"delta {c['delta']:+.3f}  discordant {c['n10']}/{c['n01']}  p = {c['p_value']:.3f}")
    print(f"\nwritten: {OUT.relative_to(ROOT)}.json")


if __name__ == "__main__":
    main()
