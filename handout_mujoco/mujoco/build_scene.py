"""Build a MuJoCo scene that mirrors the panda-gym Reach->Push layout, for KINEMATIC replay.

Robot: MuJoCo Menagerie `franka_emika_panda/panda.xml` (Apache-2.0), base moved to (-0.6, 0, 0)
       exactly like panda-gym.  Joint order is identical to panda-gym's recorded `qpos`:
       joint1..joint7, finger_joint1, finger_joint2.
Table: box 1.1 x 0.7 x 0.4 m, centre (-0.3, 0, -0.2)  -> top surface at z = 0 (panda-gym create_table).
Floor: z = -0.4.   Cube: 0.04 m, free joint (pose written every frame; no physics is simulated).
Goals: translucent cube = Push goal, translucent sphere = Reach goal (r = 0.05 = success radius).
Mode lamp: a sphere on the table corner coloured by which policy acted (grey = cheap, orange = expert).
TCP site: hand + 0.105 m along the hand z-axis = panda-gym's `panda_grasptarget` (ee_link 11).

Usage:  python build_scene.py            -> writes scene_pushdemo.xml next to panda.xml
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA_DIR = os.path.join(HERE, "..", "third_party", "mujoco_menagerie", "franka_emika_panda")
OUT = os.path.join(PANDA_DIR, "scene_pushdemo.xml")

EXTRA_ASSETS = """
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6" rgb2="0.05 0.05 0.1" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.2 0.2 0.22" rgb2="0.15 0.15 0.17" width="300" height="300"/>
    <material name="floor" texture="floor" texrepeat="6 6" reflectance="0.05"/>
"""

WORLD = """
    <light pos="0 -1 2.5" dir="0 0.4 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.4" material="floor"/>
    <geom name="table" type="box" size="0.55 0.35 0.2" pos="-0.3 0 -0.2" rgba="0.82 0.82 0.82 1"/>
    <body name="cube" pos="0 0 0.02">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="0.02 0.02 0.02" rgba="0.1 0.9 0.1 1" contype="0" conaffinity="0"/>
    </body>
    <body name="push_goal" mocap="true" pos="0 0 0.02">
      <geom type="box" size="0.02 0.02 0.02" rgba="0.1 0.9 0.1 0.3" contype="0" conaffinity="0"/>
    </body>
    <body name="reach_goal" mocap="true" pos="0 0 0.15">
      <geom type="sphere" size="0.05" rgba="0.2 0.4 1.0 0.25" contype="0" conaffinity="0"/>
    </body>
    <body name="mode_lamp" mocap="true" pos="0.18 0.28 0.04">
      <geom name="mode_lamp" type="sphere" size="0.035" rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0"/>
    </body>
"""


def main():
    src = open(os.path.join(PANDA_DIR, "panda.xml"), encoding="utf-8").read()
    s = src.replace('<mujoco model="panda">', '<mujoco model="panda_pushdemo">', 1)
    # robot base where panda-gym puts it
    s = s.replace('<body name="link0" childclass="panda">', '<body name="link0" childclass="panda" pos="-0.6 0 0">', 1)
    # TCP site = panda_grasptarget (hand + 0.105 m along hand z)
    s = re.sub(r'(<body name="hand"[^>]*>)', r'\1\n          <site name="tcp" pos="0 0 0.105" size="0.008" rgba="1 0 0 1"/>', s, count=1)
    s = s.replace("<asset>", "<asset>" + EXTRA_ASSETS, 1)
    s = s.replace("<worldbody>", "<worldbody>" + WORLD, 1)
    s = s.replace("</mujoco>", '  <visual><global offwidth="1280" offheight="720"/></visual>\n</mujoco>')
    assert 'name="tcp"' in s and 'pos="-0.6 0 0"' in s
    open(OUT, "w", encoding="utf-8").write(s)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()

