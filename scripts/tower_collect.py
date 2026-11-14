"""Layer 3, step 1: generate demonstrations that actually stack.

Route B: the observation includes the cube positions, so a policy trained on
this data can respond when a block moves -- which is precisely what the
proprioception-only RoboTurk policies cannot do.

    obs (32) = joint_pos(7) + joint_vel(7) + eef_pos(3) + eef_quat(4)
               + gripper_qpos(2) + cube_pos(3 x 3)
    act (8)  = 7 absolute joint targets (radians) + grasp bool

Same action shape as RoboTurk, so the arbitration machinery is unchanged.
"""

import argparse
import time

import h5py
import numpy as np

from scripts.tower_control import TowerExpert, controller_config, joint_action
from scripts.tower_env import TowerCreation
from scripts.tower_target import target_channels

N_CUBES = 3
# proprio(23) + absolute cubes(9) + eef-relative cubes(9) + target channels(9)
OBS_DIM = 7 + 7 + 3 + 4 + 2 + 3 * N_CUBES + 3 * N_CUBES + 3 + 3 + N_CUBES
ACT_DIM = 8


def observation(env):
    """Route B observation: proprioception AND the cube positions.

    The cube channels are the whole point -- a policy without them cannot react
    when a block moves, which is exactly the limitation of the RoboTurk
    state-only policies.
    """
    from robosuite.utils.transform_utils import mat2quat

    r, sim = env.robots[0], env.sim
    eef = sim.data.site_xpos[sim.model.site_name2id(r.controller.eef_name)]
    parts = [np.asarray(r._joint_positions),                       # 7
             np.asarray(r._joint_velocities),                      # 7
             np.asarray(eef),                                      # 3
             mat2quat(np.asarray(r.controller.ee_ori_mat)),        # 4
             np.asarray(sim.data.qpos[r._ref_gripper_joint_pos_indexes])]  # 2
    cubes = [np.array(sim.data.body_xpos[b]) for b in env.cube_body_ids]
    parts += cubes                                                    # 3 x 3 absolute
    # EEF-RELATIVE cube vectors. Absolute coordinates force the network to learn
    # a subtraction before it can answer "am I above the cube yet?"; handing it
    # the difference directly is what makes the targeting signal usable.
    # Measured without them: the gripper closed a median 84 mm from the nearest
    # cube, against the ~30 mm it needs, so it never picked anything up.
    parts += [c - np.asarray(eef) for c in cubes]                     # 3 x 3 relative
    # WHICH cube to pick up next. Without it the mapping is one-to-many: 44% of a
    # state's nearest neighbours carried a different target, so regression returned
    # the mean of the branches and the gripper landed between the cubes (measured
    # 12.5-24.7 cm against the ~3 cm a grasp needs). target_channels() is a pure
    # function of cube geometry -- no expert state -- so this identical call is what
    # runs at collection time and at runtime. See docs/closed-loop-diagnostic.md.
    parts.append(target_channels(cubes, env.table_offset[2], eef, N_CUBES))  # 3+3+3
    return np.concatenate(parts).astype(np.float32)


def make_env(seed=None, render=False, res=256):
    env = TowerCreation(
        robots="Panda", n_cubes=N_CUBES, controller_configs=controller_config(),
        has_renderer=False, has_offscreen_renderer=render, use_camera_obs=render,
        camera_names="frontview", camera_heights=res, camera_widths=res,
        control_freq=20, horizon=10 ** 6)
    if seed is not None:
        np.random.seed(seed)      # robosuite samples placements from np.random
    return env


def rollout(env, expert, max_steps=800, noise=0.0, rng=None):
    """One demonstration.

    `noise` implements DART: the EXECUTED joint target is perturbed, but the
    RECORDED label is what the expert wanted at the state it actually reached.
    Without it the expert is so smooth that the next joint target is almost
    perfectly predictable from the current arm pose alone -- so a network learns
    to ignore the cube channels entirely (measured: 0.005 rad response to a 5 cm
    cube move, versus 0.043 rad to a 5 cm arm move) and cannot stack. Perturbing
    the arm off-trajectory makes the recovery action depend on where the cube
    actually is, which is what forces the object channels to be used.
    """
    rng = rng or np.random
    env.reset()
    expert.reset()
    O, A = [], []
    for _ in range(max_steps):
        q, g = expert.act()
        O.append(observation(env))
        A.append(np.concatenate([q, [1.0 if g > 0 else 0.0]]).astype(np.float32))
        # Perturb only the TRANSIT stages. Noise during descend/grasp/place
        # destroys the grasp itself (measured: sigma 0.02 drops the expert from
        # 6/6 to 2/6), while noise during the approach is precisely what makes
        # the recovery action depend on where the cube is.
        transit = expert.stage in ("above", "lift", "carry", "retreat")
        q_exec = q + rng.normal(0, noise, 7) if (noise > 0 and transit) else q
        env.step(joint_action(env, q_exec, g))
        if env._check_success() and expert.idx >= len(expert.order or []):
            break
    return np.array(O), np.array(A), bool(env._check_success()), env.tower_height()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--out", default="data/tower/demos.hdf5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="DART: sigma (radians) of noise added to the EXECUTED "
                         "joint target. Labels stay clean. 0 = no perturbation.")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    env = make_env(seed=args.seed)
    expert = TowerExpert(env)
    rng = np.random.RandomState(args.seed + 1)
    ok, t0 = 0, time.perf_counter()
    with h5py.File(args.out, "w") as f:
        f.attrs.update(obs_dim=OBS_DIM, act_dim=ACT_DIM, n_cubes=N_CUBES,
                       dart_noise=args.noise,
                       robot="Panda", control_freq=20,
                       obs_layout="joint_pos7,joint_vel7,eef_pos3,eef_quat4,"
                                  "gripper2,cube_pos3x3,cube_rel3x3",
                       act_layout="joint_target7,grasp1")
        for i in range(args.n):
            O, A, succ, h = rollout(env, expert, args.max_steps,
                                    noise=args.noise, rng=rng)
            ok += succ
            g = f.create_group(f"demo_{i:04d}")
            g.create_dataset("obs", data=O, compression="gzip", compression_opts=4)
            g.create_dataset("act", data=A, compression="gzip", compression_opts=4)
            g.attrs.update(success=succ, steps=len(O), tower_height=h)
            if (i + 1) % 10 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1:4d}/{args.n}  success {ok}/{i+1} = {ok/(i+1):.2f}   "
                      f"{el/(i+1):.1f}s/ep   eta {(args.n-i-1)*el/(i+1)/60:.1f} min",
                      flush=True)
        f.attrs["n_demos"] = args.n
        f.attrs["n_success"] = ok
    env.close()
    print(f"\nwrote {args.out}   {ok}/{args.n} successful  "
          f"({time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
