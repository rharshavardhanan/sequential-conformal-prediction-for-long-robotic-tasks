"""Re-run the transfer gap with the tuned controllers, using RECORDED EXPERT ACTIONS.

No policies are loaded. The recorded action sequence is an input; every state comes
from MuJoCo integrating it.
"""
import json, os
import numpy as np
from mj_env import MJLive, ACTION_SCALE, SUCCESS_RADIUS
from tune_controller import set_ctrl_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CONFIGS = [("baseline",          1, 0.100, 1),
           ("SELECTED kv.035",   1, 0.035, 10),
           ("A kp1 kv.02 f30",   1, 0.020, 30),
           ("C kp5 kv.05 f30",   5, 0.050, 30)]

def run(env, seed, cond="expert"):
    g = lambda f: np.asarray(env.z[f"{cond}/seed{seed}/{f}"])
    acts, ee_ref, obj_ref, succ_ref = g("actions"), g("ee"), g("obj_pos"), bool(g("success"))
    env.reset_from_seed(seed, cond)
    ee_e, ob_e, ratio = [], [], []
    for t in range(len(acts)):
        b = env.ee(); s = env.step(acts[t])
        c = np.linalg.norm(np.clip(acts[t], -1, 1)) * ACTION_SCALE
        if c > 1e-6: ratio.append(np.linalg.norm(s["ee"] - b) / c)
        ee_e.append(np.linalg.norm(s["ee"] - ee_ref[t+1]))
        ob_e.append(np.linalg.norm(s["cube"] - obj_ref[t+1]))
    return (np.array(ee_e), np.array(ob_e), np.array(ratio),
            env.success(), succ_ref, np.linalg.norm(env.cube() - env.task_goal))

def main():
    env = MJLive()
    base = {"kp": env.m.actuator_gainprm[:7, 0].copy(),
            "forcerange": env.m.actuator_forcerange[:7].copy()}
    seeds = [230044, 230011, 230120, 230375, 230001, 230002, 230003, 230004]
    out = {}
    print(f"recorded EXPERT actions replayed through live MuJoCo physics, n={len(seeds)} seeds")
    print(f"{'config':<18}{'ratio':>7}{'ee@2':>8}{'ee@10':>8}{'ee@20':>8}{'ee@end':>9}"
          f"{'cube@20':>9}{'cube@end':>10}{'succ':>7}")
    for name, kp, kv, fm in CONFIGS:
        set_ctrl_params(env, kp, kv, fm, base)
        E2=E10=E20=EE=C20=CE=0; R=[]; ok=0; ref_ok=0
        for sd in seeds:
            ee_e, ob_e, r, s_mj, s_ref, dfin = run(env, sd)
            f=lambda a,i: a[i] if i < len(a) else a[-1]
            E2+=f(ee_e,2); E10+=f(ee_e,10); E20+=f(ee_e,20); EE+=ee_e[-1]
            C20+=f(ob_e,20); CE+=ob_e[-1]; R+=list(r); ok+=s_mj; ref_ok+=s_ref
        n=len(seeds)
        out[name]={"ratio_median":float(np.median(R)),"ee2":E2/n,"ee10":E10/n,"ee20":E20/n,
                   "ee_end":EE/n,"cube20":C20/n,"cube_end":CE/n,
                   "mj_success":ok/n,"pybullet_success":ref_ok/n,
                   "params":{"kp_mult":kp,"kv_ratio":kv,"force_mult":fm}}
        print(f"{name:<18}{np.median(R):>7.3f}{E2/n:>8.4f}{E10/n:>8.4f}{E20/n:>8.4f}"
              f"{EE/n:>9.4f}{C20/n:>9.4f}{CE/n:>10.4f}{ok}/{n:<5}")
    print(f"\n  PyBullet reference success on these seeds: {ref_ok}/{n}")
    print(f"  success radius = {SUCCESS_RADIUS} m")
    json.dump(out, open(os.path.join(RESULTS,"transfer_gap_tuned.json"),"w"), indent=2, default=float)
    print(f"\nwrote results/transfer_gap_tuned.json")

if __name__ == "__main__":
    main()
