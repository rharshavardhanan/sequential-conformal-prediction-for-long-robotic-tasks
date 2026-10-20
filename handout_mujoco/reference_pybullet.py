"""Ground-truth reference: replay the SAME recorded episode kinematically in panda-gym/PyBullet.

Needs the project venv (panda-gym 3.0.7 + PyBullet).  Produces reference/pybullet_<condition>_seed<seed>.mp4
so the MuJoCo animation can be compared against the simulator the data came from.
"""
import argparse
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import panda_gym  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "replay_trajectories_a0.30.npz")

ap = argparse.ArgumentParser()
ap.add_argument("--condition", default="arbitrated_a0.30")
ap.add_argument("--seed", type=int, default=230044)
a = ap.parse_args()

z = np.load(DATA)
p = "%s/seed%d/" % (a.condition, a.seed)
env = gym.make("PandaPush-v3", render_mode="rgb_array")
env.reset(seed=a.seed)
sim, robot = env.unwrapped.sim, env.unwrapped.robot
sim.set_base_pose("target", z[p + "task_goal"], np.array([0.0, 0.0, 0.0, 1.0]))
frames = []
for t in range(len(z[p + "qpos"])):
    robot.set_joint_angles(z[p + "qpos"][t])
    sim.set_base_pose("object", z[p + "obj_pos"][t], z[p + "obj_quat"][t])
    frames.append(env.render())
frames += [frames[-1]] * 25
os.makedirs(os.path.join(HERE, "reference"), exist_ok=True)
out = os.path.join(HERE, "reference", "pybullet_%s_seed%d.mp4" % (a.condition, a.seed))
imageio.mimsave(out, frames, fps=25, macro_block_size=1)
print("wrote", out)
