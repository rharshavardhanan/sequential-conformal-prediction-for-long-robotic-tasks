"""Check that recorded panda-gym joint angles reproduce the recorded end-effector in the MuJoCo scene.

Writes data/fk_check.json.  Expected: millimetre-level agreement (median ~1 mm, max ~5 mm).
"""
import json
import os

import mujoco
import numpy as np

from replay_episode import ARM, DATA, ROOT, SCENE

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
adr = [m.joint(n).qposadr[0] for n in ARM]
z = np.load(DATA)
errs = []
for key in (k for k in z.files if k.endswith("/qpos")):
    q, ee = z[key], z[key.replace("/qpos", "/ee")]
    for t in range(len(q)):
        d.qpos[adr] = q[t]
        mujoco.mj_kinematics(m, d)
        errs.append(float(np.linalg.norm(d.site("tcp").xpos - ee[t])))
e = np.array(errs)
out = {"frames": int(e.size), "median_m": float(np.median(e)), "p99_m": float(np.quantile(e, .99)), "max_m": float(e.max()),
       "mujoco_version": mujoco.__version__}
json.dump(out, open(os.path.join(ROOT, "data", "fk_check.json"), "w"), indent=2)
print(out)
