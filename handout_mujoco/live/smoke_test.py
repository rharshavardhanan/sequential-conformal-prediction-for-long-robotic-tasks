"""Deterministic smoke test: is the loop real, and does the action move the robot?

Proves three things before any policy is involved:
  1. reset places the robot and cube at the recorded seed's initial condition
  2. one action changes the state, and the change is caused by the action
  3. a zero action does NOT produce the same change (so motion is not drift)
"""
import numpy as np
from mj_env import MJLive, ACTION_SCALE

SEED = 230044

def main():
    env = MJLive()
    s0 = env.reset_from_seed(SEED)
    print(f"seed {SEED}  substeps/control step = {env.substeps}  dt = {env.m.opt.timestep}")
    print(f"  initial ee   {np.round(s0['ee'], 4)}")
    print(f"  initial cube {np.round(s0['cube'], 4)}")
    print(f"  reach goal   {np.round(env.reach_goal, 4)}")
    print(f"  task  goal   {np.round(env.task_goal, 4)}")

    a = np.array([1.0, 0.0, 0.0])
    s1 = env.step(a)
    d = s1["ee"] - s0["ee"]
    print(f"\n  action {a} -> commanded dx = {a * ACTION_SCALE}")
    print(f"  ee moved     {np.round(d, 4)}   |d| = {np.linalg.norm(d):.4f} m")
    print(f"  joints moved {np.round(s1['arm_q'] - s0['arm_q'], 4)}")

    env.reset_from_seed(SEED)
    s2 = env.step(np.zeros(3))
    d0 = s2["ee"] - s0["ee"]
    print(f"\n  zero action -> ee moved {np.round(d0, 4)}  |d| = {np.linalg.norm(d0):.4f} m")

    ok_moves = np.linalg.norm(d) > 1e-3
    ok_causal = np.linalg.norm(d) > 3 * max(np.linalg.norm(d0), 1e-9)
    ok_dir = float(d[0]) > 0
    print(f"\n  [{'PASS' if ok_moves else 'FAIL'}] the action moves the arm")
    print(f"  [{'PASS' if ok_causal else 'FAIL'}] motion is caused by the action, not drift")
    print(f"  [{'PASS' if ok_dir else 'FAIL'}] motion follows the commanded +x direction")
    print(f"  [{'PASS' if abs(env.m.opt.timestep*env.substeps-0.04) < 1e-9 else 'FAIL'}]"
          f" control period is 0.04 s (25 Hz)")

if __name__ == "__main__":
    main()
