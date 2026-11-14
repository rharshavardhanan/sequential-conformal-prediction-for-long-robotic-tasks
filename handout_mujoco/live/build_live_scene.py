"""Build a MuJoCo scene that can actually SIMULATE the Reach->Push task.

This is deliberately NOT `mujoco/build_scene.py`. That scene exists for kinematic
replay and is wrong for live physics in two ways:

  * the cube carries `contype="0" conaffinity="0"` -- it cannot be touched, because
    nothing was ever simulated; its pose was written in frame by frame.
  * panda.xml's `home` keyframe lists 9 qpos values, but adding the cube's free
    joint makes nq = 16, so the model fails to load at all in MuJoCo 3.2.7.

Neither file in `mujoco/` is modified; the replay path and the PyBullet benchmark
stay exactly as shipped.

Geometry is copied from `data/meta.json` so the live scene and the recorded
episodes describe the same world: table top at z = 0, robot base at (-0.6, 0, 0),
cube edge 0.04 m, TCP = hand + 0.105 m along hand z.

    python live/build_live_scene.py     -> writes scene_live.xml next to panda.xml
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA_DIR = os.path.join(
    HERE, "..", "third_party", "mujoco_menagerie", "franka_emika_panda"
)
OUT = os.path.join(PANDA_DIR, "scene_live.xml")

ASSETS = """
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6" rgb2="0.05 0.05 0.1" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.2 0.2 0.22" rgb2="0.15 0.15 0.17" width="300" height="300"/>
    <material name="floor" texture="floor" texrepeat="6 6" reflectance="0.05"/>
"""

# The cube is a real dynamic body here: mass, friction, and collision enabled so the
# gripper can push it. Goal markers stay non-colliding mocap bodies.
WORLD = """
    <light pos="0 -1 2.5" dir="0 0.4 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.4" material="floor"/>
    <geom name="table" type="box" size="0.55 0.35 0.2" pos="-0.3 0 -0.2"
          rgba="0.82 0.82 0.82 1" friction="1.0 0.005 0.0001"/>
    <body name="cube" pos="0 0 0.02">
      <freejoint name="cube_free"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="1.3e-5 1.3e-5 1.3e-5"/>
      <geom name="cube" type="box" size="0.02 0.02 0.02" rgba="0.1 0.9 0.1 1"
            friction="1.0 0.005 0.0001" solref="0.004 1" condim="4"/>
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
    s = src.replace('<mujoco model="panda">', '<mujoco model="panda_live">', 1)
    s = s.replace(
        '<body name="link0" childclass="panda">',
        '<body name="link0" childclass="panda" pos="-0.6 0 0">',
        1,
    )
    s = re.sub(
        r'(<body name="hand"[^>]*>)',
        r'\1\n          <site name="tcp" pos="0 0 0.105" size="0.008" rgba="1 0 0 1"/>',
        s,
        count=1,
    )
    s = s.replace("<asset>", "<asset>" + ASSETS, 1)
    s = s.replace("<worldbody>", "<worldbody>" + WORLD, 1)
    # the shipped builder leaves this in place and the model then refuses to load:
    # 9 keyframe values against nq = 16 once the cube's free joint exists.
    s = re.sub(r"\s*<keyframe>.*?</keyframe>", "", s, flags=re.S)
    s = s.replace(
        "</mujoco>",
        '  <visual><global offwidth="1280" offheight="720"/></visual>\n</mujoco>',
    )
    assert 'name="tcp"' in s and 'pos="-0.6 0 0"' in s and "<keyframe>" not in s
    open(OUT, "w", encoding="utf-8").write(s)
    print("wrote", os.path.normpath(OUT))

    import mujoco

    m = mujoco.MjModel.from_xml_path(OUT)
    print(f"loads OK: nq={m.nq} nv={m.nv} nu={m.nu} timestep={m.opt.timestep}")


if __name__ == "__main__":
    main()
