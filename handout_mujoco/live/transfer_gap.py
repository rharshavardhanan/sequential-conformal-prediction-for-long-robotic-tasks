"""How far does live MuJoCo drift from the recorded PyBullet run under the SAME actions?

This is the transfer question, answered before any policy exists. Feed each recorded
action into live MuJoCo physics and compare the resulting trajectory with the
recorded PyBullet one. Recorded actions are used as an INPUT SEQUENCE only -- every
state comes from MuJoCo integrating them. No state is ever copied across.

If the two track closely, PyBullet-trained policies have a chance in MuJoCo. If they
diverge in a handful of steps, retraining is certain and is better known now.
"""
import numpy as np
from mj_env import MJLive, ACTION_SCALE

def run(env, seed, cond, gain=1.0, n=None):
    g = lambda f: np.asarray(env.z[f"{cond}/seed{seed}/{f}"])
    acts, ee_ref, obj_ref = g("actions"), g("ee"), g("obj_pos")
    env.reset_from_seed(seed, cond)
    T = len(acts) if n is None else min(n, len(acts))
    ee_mj, obj_mj, ratio = [], [], []
    for t in range(T):
        before = env.ee()
        s = env.step(np.clip(acts[t] * gain, -1, 1))
        cmd = np.linalg.norm(np.clip(acts[t], -1, 1) * ACTION_SCALE)
        got = np.linalg.norm(s["ee"] - before)
        if cmd > 1e-6:
            ratio.append(got / cmd)
        ee_mj.append(s["ee"]); obj_mj.append(s["cube"])
    ee_mj, obj_mj = np.array(ee_mj), np.array(obj_mj)
    ee_err = np.linalg.norm(ee_mj - ee_ref[1:T+1], axis=1)
    obj_err = np.linalg.norm(obj_mj - obj_ref[1:T+1], axis=1)
    return ee_err, obj_err, np.array(ratio), T

def main():
    env = MJLive()
    seeds = [230044, 230011, 230120, 230375]
    print("tracking gain: achieved / commanded end-effector displacement per step")
    print(f"{'gain':>6}{'median ratio':>14}{'ee err @10':>12}{'ee err @end':>13}"
          f"{'cube err @end':>15}")
    for gain in (1.0, 2.0, 4.0, 1.0/0.21):
        E10, EE, OO, RR = [], [], [], []
        for sd in seeds:
            ee_err, obj_err, ratio, T = run(env, sd, "expert", gain=gain)
            if len(ee_err) > 10: E10.append(ee_err[10])
            EE.append(ee_err[-1]); OO.append(obj_err[-1]); RR.append(np.median(ratio))
        print(f"{gain:>6.2f}{np.mean(RR):>14.3f}{np.mean(E10):>12.4f}"
              f"{np.mean(EE):>13.4f}{np.mean(OO):>15.4f}")

    print("\nper-step divergence, seed 230044, expert actions, gain 1.0:")
    ee_err, obj_err, ratio, T = run(env, 230044, "expert")
    for t in (0, 2, 5, 10, 20, min(40, T-1), T-1):
        if t < len(ee_err):
            print(f"  step {t:3d}   ee error {ee_err[t]:.4f} m   cube error {obj_err[t]:.4f} m")
    print(f"\n  median achieved/commanded ratio {np.median(ratio):.3f}")
    print(f"  (PyBullet's controller reaches ~1.0; MuJoCo's position actuators do not")
    print(f"   converge inside one 0.04 s control period)")

if __name__ == "__main__":
    main()
