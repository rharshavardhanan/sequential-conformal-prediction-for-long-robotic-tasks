"""How much of the planned-vs-executed gap is just OSC compliance?

The CP score is || planned - executed ||. OSC is a compliant impedance
controller: it will not reach a commanded setpoint within one 1/20 s step. If
the residual is a fixed *fraction* of the commanded delta, then the raw score is
mostly a proxy for "how fast is the policy asking to move" rather than an
anomaly signal -- which changes how the score should be defined.

This sweeps commanded magnitude and reports the realised/commanded gain.
"""

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SETTLE = 12  # steps to let the OSC transient decay before measuring
HOLD = 12  # steps averaged for the steady-state estimate


def make_env():
    suite_ = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite_.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.seed(0)
    return env


def main():
    env = make_env()
    ctrl = env.env.robots[0].controller
    print(f"position_limits  : {ctrl.position_limits}")
    print(f"kp               : {np.asarray(ctrl.kp)}")
    print(f"kd               : {np.asarray(ctrl.kd)}")
    print(f"interpolator_pos : {ctrl.interpolator_pos}")

    print("\n" + "=" * 74)
    print("STEADY-STATE TRACKING GAIN  (+x axis, constant hold)")
    print("=" * 74)
    print(
        f"{'norm act':>9} {'commanded':>12} {'realised':>12} {'gain':>8} {'residual':>12}"
    )
    print(f"{'':>9} {'(m/step)':>12} {'(m/step)':>12} {'':>8} {'(m)':>12}")
    print("-" * 74)

    rows = []
    for mag in (0.1, 0.25, 0.5, 0.75, 1.0):
        env.reset()
        act = np.zeros(7)
        act[0] = mag
        for _ in range(SETTLE):
            env.step(act)
        p0 = env.env.robots[0].controller.ee_pos.copy()
        for _ in range(HOLD):
            env.step(act)
        p1 = env.env.robots[0].controller.ee_pos.copy()

        commanded = mag * 0.05
        realised = (p1[0] - p0[0]) / HOLD
        gain = realised / commanded
        residual = commanded - realised
        rows.append((mag, commanded, realised, gain, residual))
        print(
            f"{mag:>9.2f} {commanded:>12.5f} {realised:>12.5f} {gain:>8.3f} {residual:>12.5f}"
        )

    gains = [r[3] for r in rows]
    print("-" * 74)
    print(
        f"gain mean={np.mean(gains):.3f}  sd={np.std(gains):.4f}  "
        f"min={min(gains):.3f}  max={max(gains):.3f}"
    )

    print("\nInterpretation:")
    if np.std(gains) < 0.05:
        print(
            f"  Gain is ~constant at {np.mean(gains):.2f} across the commanded range."
        )
        print(
            "  => residual ~= (1 - gain) * ||commanded delta||, i.e. the raw score is"
        )
        print(
            "     ~proportional to commanded speed. A healthy fast motion will outscore"
        )
        print("     an unhealthy slow one. Normalise the score by ||commanded delta||,")
        print(
            "     or calibrate per sub-phase so speed is held roughly constant within"
        )
        print(
            "     each calibration bucket (which is what the sequential CP design does)."
        )
    else:
        print(
            "  Gain varies with commanded magnitude; residual is not a simple scaling."
        )

    env.close()


if __name__ == "__main__":
    main()
