"""Step 1: confirm the LIBERO action space and the delta-pose scaling factor.

The collection loop needs `planned_eef = current_eef + f(action)`. The raw policy
action is normalised to [-1, 1] and rescaled by the OSC controller before it ever
reaches the sim, so `current + action[:3]` is wrong by a constant factor. This
script reports the declared controller limits and then *measures* the realised
scaling by stepping saturated actions from a settled state.
"""

import json

import numpy as np
import robosuite as suite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_SUITE = "libero_10"
TASK_ID = 0


def main():
    suite_ = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite_.get_task(TASK_ID)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

    print(f"suite        : {TASK_SUITE}  ({suite_.n_tasks} tasks)")
    print(f"task[{TASK_ID}]      : {task.name}")
    print(f"language     : {task.language}")
    print(f"bddl         : {bddl}")

    # Rendering off: this check is pure kinematics, no pixels needed.
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.seed(0)
    obs = env.reset()

    robot = env.env.robots[0]
    ctrl = robot.controller

    print("\n" + "=" * 62)
    print("DECLARED CONTROLLER CONFIG")
    print("=" * 62)
    print(f"controller class : {type(ctrl).__name__}")
    print(f"name             : {ctrl.name}")
    print(
        f"control_freq     : {ctrl.control_freq} Hz  (dt = {1 / ctrl.control_freq:.4f} s)"
    )
    print(
        f"control_delta    : {getattr(ctrl, 'use_delta', 'n/a')}   <- True => action is a DELTA pose"
    )
    print(f"impedance_mode   : {getattr(ctrl, 'impedance_mode', 'n/a')}")
    print(f"uncouple_pos_ori : {getattr(ctrl, 'uncoupling', 'n/a')}")
    print(f"control_dim      : {ctrl.control_dim}")

    print(f"\ninput_min        : {np.asarray(ctrl.input_min)}")
    print(f"input_max        : {np.asarray(ctrl.input_max)}")
    print(f"output_min       : {np.asarray(ctrl.output_min)}")
    print(f"output_max       : {np.asarray(ctrl.output_max)}")
    print(f"action_scale     : {np.asarray(ctrl.action_scale)}")
    print(f"action_output_transform : {np.asarray(ctrl.action_output_transform)}")
    print(f"action_input_transform  : {np.asarray(ctrl.action_input_transform)}")

    low, high = env.env.action_spec
    print("\n" + "=" * 62)
    print("ACTION SPEC (what the policy emits)")
    print("=" * 62)
    print(
        f"action dim  : {low.shape[0]}   (= {ctrl.control_dim} arm + {low.shape[0] - ctrl.control_dim} gripper)"
    )
    print(f"action low  : {low}")
    print(f"action high : {high}")

    # ---- empirical check -------------------------------------------------
    # Hold a saturated +1 on one axis and integrate the realised EEF motion.
    # OSC is a compliant controller, so a single step undershoots its target;
    # what we want is the *commanded* delta, which is what scale_action gives.
    print("\n" + "=" * 62)
    print("SCALING: scale_action() on saturated input")
    print("=" * 62)
    for axis, name in enumerate("xyz"):
        a = np.zeros(ctrl.control_dim)
        a[axis] = 1.0
        scaled = ctrl.scale_action(a.copy())
        print(f"  action[{name}] = +1.0  ->  scaled delta = {scaled[axis]:+.5f} m")
    for axis, name in zip(range(3, 6), ("rx", "ry", "rz")):
        a = np.zeros(ctrl.control_dim)
        a[axis] = 1.0
        scaled = ctrl.scale_action(a.copy())
        print(f"  action[{name}]= +1.0  ->  scaled delta = {scaled[axis]:+.5f} rad")

    # ---- realised motion over a settled hold ------------------------------
    print("\n" + "=" * 62)
    print("REALISED EEF MOTION (20 steps of a saturated +x command)")
    print("=" * 62)
    start = robot.controller.ee_pos.copy()
    prev = start.copy()
    for i in range(20):
        act = np.zeros(low.shape[0])
        act[0] = 1.0
        obs, _, _, _ = env.step(act)
        now = robot.controller.ee_pos.copy()
        if i < 5 or i == 19:
            print(
                f"  step {i:2d}: per-step dx = {now[0] - prev[0]:+.5f} m   cumulative = {now[0] - start[0]:+.5f} m"
            )
        prev = now
    print(f"\n  mean per-step dx over 20 steps: {(prev[0] - start[0]) / 20:+.5f} m")
    print("  ^ steady-state per-step motion approaches the commanded delta once")
    print("    the OSC transient settles; the commanded delta is the number to use.")

    # ---- the formula ------------------------------------------------------
    om = np.asarray(ctrl.output_max)
    print("\n" + "=" * 62)
    print("FORMULA FOR THE COLLECTION LOOP")
    print("=" * 62)
    print("  current_pos = obs['robot0_eef_pos']            # 3-vector, world frame")
    print(f"  planned_pos = current_pos + action[:3] * {om[:3]}")
    print("  tracking_error = || planned_pos - next_obs['robot0_eef_pos'] ||")
    print(
        "\n  NOT `current + action[:3]` -- that is off by a factor of "
        f"{1.0 / om[0]:.0f}x on position."
    )

    print("\n" + "=" * 62)
    print("OBS KEYS AVAILABLE (state-only env)")
    print("=" * 62)
    for k in sorted(obs.keys()):
        v = np.asarray(obs[k])
        print(f"  {k:34s} {str(v.shape):10s} {v.dtype}")

    summary = {
        "suite": TASK_SUITE,
        "controller": type(ctrl).__name__,
        "control_freq": int(ctrl.control_freq),
        "control_delta": bool(getattr(ctrl, "use_delta", True)),
        "action_dim": int(low.shape[0]),
        "arm_dim": int(ctrl.control_dim),
        "output_max": np.asarray(ctrl.output_max).tolist(),
        "output_min": np.asarray(ctrl.output_min).tolist(),
        "input_max": np.asarray(ctrl.input_max).tolist(),
        "input_min": np.asarray(ctrl.input_min).tolist(),
        "pos_scale_m": float(om[0]),
        "rot_scale_rad": float(om[3]),
    }
    with open("action_space.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote action_space.json")

    env.close()


if __name__ == "__main__":
    main()
