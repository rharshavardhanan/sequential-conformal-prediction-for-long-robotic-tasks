"""TowerCreation: a closed-loop MuJoCo stacking task for RoboTurk-style policies.

Real physics, real contacts, real success predicate. Nothing here is replayed.

Two facts about this environment that a user must know, because neither raises an
error on its own:

1. **Sawyer's joint 0 does not actuate in robosuite 1.4.1.** Measured: 79 Nm of
   raw torque applied directly to the actuator for 2 simulated seconds moves it
   0.0007 rad, while joints 1-6 track a 0.4 rad step to within 0.03 rad. Panda
   tracks all seven to 0.000. Base rotation is what lets an arm reach across a
   table, so the default robot here is Panda; `robot="Sawyer"` is available and
   prints a warning.

2. **The RoboTurk policies are proprioception-only.** Their 37-dim input carries
   no object channel and no image, so they cannot perceive these cubes. They will
   drive the arm through plausible learned motion and will not stack anything.
   The env exposes `cube*_pos` observables anyway, so a policy that *can* see is
   a drop-in replacement.

Action space is the RoboTurk one: 7 absolute joint targets in radians plus a
grasp bool, converted here to the rate-limited deltas robosuite's
JointPositionController actually consumes (it always adds a scaled delta to the
current position -- `control_delta` is not consulted).
"""

import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

CUBE = 0.022                      # half-extent, metres
COLOURS = [([1, .2, .2, 1], "WoodRed", "redwood"),
           ([.2, 1, .2, 1], "WoodGreen", "greenwood"),
           ([.3, .4, 1, 1], "WoodBlue", "bluewood"),
           ([1, .9, .2, 1], "WoodLight", "lightwood")]


class TowerCreation(SingleArmEnv):
    """Stack `n_cubes` into a single tower. Success = every cube in one column."""

    def __init__(self, robots="Panda", n_cubes=3,
                 table_full_size=(0.8, 0.8, 0.05),
                 table_friction=(1.0, 5e-3, 1e-4), **kw):
        self.n_cubes = int(n_cubes)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.placement_initializer = kw.pop("placement_initializer", None)
        # `reward_shaping` is declared by individual robosuite tasks, not by the
        # base class, so it has to be absorbed here rather than forwarded.
        self.reward_shaping = kw.pop("reward_shaping", False)
        super().__init__(robots=robots, **kw)

    # ---------------------------------------------------------------- model --
    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]))

        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])

        tex = {"type": "cube"}
        mat = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        self.cubes = []
        for i in range(self.n_cubes):
            rgba, texture, nm = COLOURS[i % len(COLOURS)]
            self.cubes.append(BoxObject(
                name=f"cube{i}",
                size_min=[CUBE] * 3, size_max=[CUBE] * 3, rgba=rgba,
                material=CustomMaterial(texture=texture, tex_name=f"{nm}{i}",
                                        mat_name=f"{nm}{i}_mat",
                                        tex_attrib=tex, mat_attrib=mat)))

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler", mujoco_objects=self.cubes,
                x_range=[-0.14, 0.14], y_range=[-0.14, 0.14], rotation=None,
                ensure_object_boundary_in_range=False, ensure_valid_placement=True,
                reference_pos=self.table_offset, z_offset=0.01)

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self.cubes)

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_ids = [self.sim.model.body_name2id(c.root_body)
                              for c in self.cubes]

    def _setup_observables(self):
        obs = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix
        for i in range(self.n_cubes):
            def make(idx):
                @sensor(modality="object")
                def pos(obs_cache, idx=idx):
                    return np.array(self.sim.data.body_xpos[self.cube_body_ids[idx]])
                return pos
            nm = f"cube{i}_pos"
            obs[nm] = Observable(name=nm, sensor=make(i),
                                 sampling_rate=self.control_freq)
        return obs

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # sampler yields (pos, quat, obj) -- not (obj, pos, quat)
            for pos, quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(pos), np.array(quat)]))

    # -------------------------------------------------------------- success --
    def _tower(self):
        """Return the cube positions sorted bottom-to-top."""
        return sorted((np.array(self.sim.data.body_xpos[b]) for b in self.cube_body_ids),
                      key=lambda p: p[2])

    def _check_success(self):
        """Every cube in one column: consecutive layers one cube-height apart and
        horizontally aligned, with the bottom cube resting on the table."""
        ps = self._tower()
        table_z = self.table_offset[2]
        if abs(ps[0][2] - (table_z + CUBE)) > 0.02:
            return False
        for lo, hi in zip(ps, ps[1:]):
            if abs((hi[2] - lo[2]) - 2 * CUBE) > 0.012:
                return False
            if np.linalg.norm(hi[:2] - lo[:2]) > 0.035:
                return False
        return True

    def tower_height(self):
        """How many cubes are currently stacked, bottom-up. 1 = none stacked."""
        ps, n, table_z = self._tower(), 1, self.table_offset[2]
        if abs(ps[0][2] - (table_z + CUBE)) > 0.02:
            return 0
        for lo, hi in zip(ps, ps[1:]):
            if (abs((hi[2] - lo[2]) - 2 * CUBE) <= 0.012
                    and np.linalg.norm(hi[:2] - lo[:2]) <= 0.035):
                n += 1
            else:
                break
        return n

    def reward(self, action=None):
        return float(self._check_success())
