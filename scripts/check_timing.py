"""Time one rollout with and without rendering, to size Mac vs cloud.

Excludes policy inference -- this is the simulator+render floor only.
"""

import time

import numpy as np
from libero.libero import benchmark, get_libero_path

STEPS = 200


def build(use_camera_obs, **kw):
    from libero.libero.envs import OffScreenRenderEnv

    suite_ = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite_.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        has_renderer=False,
        has_offscreen_renderer=use_camera_obs,
        use_camera_obs=use_camera_obs,
        **kw,
    )
    env.seed(0)
    return env


def time_env(label, env):
    env.reset()
    rng = np.random.RandomState(0)
    # one untimed step so first-call lazy init isn't charged to the loop
    env.step(np.zeros(7))
    t0 = time.perf_counter()
    for _ in range(STEPS):
        env.step(rng.uniform(-1, 1, 7))
    dt = time.perf_counter() - t0
    hz = STEPS / dt
    print(
        f"{label:38s} {dt:7.2f} s / {STEPS} steps   {hz:7.1f} Hz   {1000 * dt / STEPS:6.2f} ms/step"
    )
    env.close()
    return dt / STEPS


def main():
    print(f"timing {STEPS} steps each (sim only, no policy inference)\n")
    s_state = time_env("state only (no rendering)", build(False))
    s_rgb = time_env(
        "rgb 128x128, agentview+eye_in_hand",
        build(
            True,
            camera_heights=128,
            camera_widths=128,
            camera_names=["agentview", "robot0_eye_in_hand"],
        ),
    )

    print("\n" + "=" * 74)
    print("EXTRAPOLATION  (600-step episodes, sim only -- add policy inference)")
    print("=" * 74)
    for label, per_step in (("state only", s_state), ("with 2x128px rgb", s_rgb)):
        ep = per_step * 600
        print(f"\n  {label}:  {ep:.1f} s/episode")
        for n_task, n_roll in ((4, 100), (10, 100)):
            total = ep * n_task * n_roll
            print(
                f"    {n_task:2d} tasks x {n_roll} rollouts = {n_task * n_roll:5d} eps"
                f"  ->  {total / 3600:6.1f} h  ({total / 86400:.1f} days)"
            )


if __name__ == "__main__":
    main()
