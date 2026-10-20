# Handout: Reach → Push policy-arbitration episodes, for MuJoCo replay animation

**What this is:** recorded closed-loop episodes of a working cheap↔expert policy-arbitration setup. They were run in panda-gym (PyBullet), packaged with a ready MuJoCo scene and a replay script so they can be animated in MuJoCo.

> **Important: MuJoCo does not run the policies here.** Joint angles and cube poses recorded in PyBullet are written into MuJoCo frame by frame, and MuJoCo only renders them (a kinematic replay). The policies are PyBullet-trained. Running them with MuJoCo physics would need re-verification and likely re-training. Label videos as "replay of simulated runs".

Read **`CLAIMS.md`** before presenting numbers.

---

## 1. Quick start (about 5 minutes)

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
python mujoco/build_scene.py                              # writes the MuJoCo scene next to Menagerie's panda.xml
python mujoco/verify_fk.py                                # sanity check: expect median ~1 mm, max ~5 mm
python mujoco/replay_episode.py --condition arbitrated_a0.30 --seed 230044 --out arb_230044.mp4
python mujoco/replay_episode.py --condition arbitrated_a0.30 --seed 230044 --viewer   # interactive window
```

Pre-rendered examples are already in `samples/`. The matching PyBullet ground-truth video is in `reference/`.

## 2. Folder layout

| Path | Contents |
|---|---|
| `CLAIMS.md` | results summary: what can and cannot be claimed |
| `data/replay_trajectories_a0.30.npz` | **all recorded episodes**: 4 conditions × 500 seeds (10.7 MB) |
| `data/episode_index.csv` | one row per episode: success, steps, expert steps, escalations, phase-switch step |
| `data/showcase_episodes.csv` | 4 hand-picked seeds with a clear story (3 rescues + 1 honest failure) |
| `data/meta.json` | frames, units, geometry, conventions, arbiter thresholds, disclosures |
| `data/fk_check.json` | measured agreement between recorded end-effector and MuJoCo forward kinematics |
| `mujoco/build_scene.py` | builds `scene_pushdemo.xml`: Menagerie Panda + panda-gym table, cube, goals, policy lamp |
| `mujoco/replay_episode.py` | renders an MP4 with a text overlay, or opens the interactive viewer |
| `mujoco/verify_fk.py` | re-checks the joint mapping on every frame |
| `samples/` | MuJoCo renders: 4 showcase episodes (arbitrated) + cheap/expert/arbitrated for seed 230044 |
| `reference/` | PyBullet render of the same recorded episode, plus the script that made it (needs panda-gym) |
| `results/` | benchmark JSON, efficiency summary, rescue-curve figure, full technical report |
| `third_party/mujoco_menagerie/franka_emika_panda/` | robot model from Google DeepMind MuJoCo Menagerie (Apache-2.0, see its LICENSE) |

## 3. Data format (`replay_trajectories_a0.30.npz`)

**Keys:** `"<condition>/seed<seed>/<field>"`.
- **Conditions:**
  - `cheap`: cheap policy only.
  - `expert`: expert pair only.
  - `arbitrated_a0.30`: live arbitration.
  - `random_a0.30`: random-escalation control.
- **Seeds:** 230000–230499. The same seed means the same initial cube, goal and reach target in all conditions.

| Field | Shape | Meaning |
|---|---|---|
| `qpos` | (T+1, 9) | Panda joints `joint1..joint7` (rad), `finger_joint1/2` (m). Row 0 = after reset |
| `obj_pos` | (T+1, 3) | cube centre (m) |
| `obj_quat` | (T+1, 4) | cube orientation, **PyBullet order (x, y, z, w)** |
| `ee` | (T+1, 3) | end-effector (grasp target) position, for checking |
| `reach_goal` / `task_goal` | (3,) | phase-A target point / Push goal (cube centre) |
| `modes` | (T,) | **0 = cheap policy acted, 1 = expert acted** at step t |
| `labels` | (T,) | 0 = reach phase, 1 = push phase |
| `score`, `flags` | (T,) | arbiter score s(t) and escalation flag (`random_a0.30`: the flag is a random coin) |
| `actions` | (T, 3) | end-effector displacement command in [−1, 1] (× 0.05 m) |
| `success` | () | episode success |

**Timing:** one row per control step = **0.04 s (25 Hz)**. Frame t+1 is the result of action t, so the policy that moved the arm *into* frame t is `modes[t-1]`.

## 4. Conventions that matter when building your own MuJoCo scene
- **World:** z up, metres. Table top at **z = 0**. Robot base at **(−0.6, 0, 0)**. Table 1.1 × 0.7 × 0.4 m centred at (−0.3, 0, −0.2). Floor at z = −0.4. Cube edge 0.04 m.
- **Joint order** in `qpos` equals Menagerie `panda.xml` joint order. **Address joints by name** (`model.joint("joint1").qposadr`): any free joint added earlier in the XML (e.g. the cube) shifts `qpos` indices.
- **Quaternions:** PyBullet is (x, y, z, w); MuJoCo free joints expect **(w, x, y, z)**.
- **Fingers:** the gripper is blocked in these tasks (≈0–0.01 m). Clip to Menagerie's range [0, 0.04].
- **End-effector:** panda-gym's end-effector is the grasp target, **0.105 m along the hand z-axis** (the `tcp` site in our scene). Verified at 1.3 mm median / 5.0 mm max error over 63 019 frames with MuJoCo 3.13.

## 5. Suggested demo storyline (seed 230044)
1. `samples/mujoco_cheap_seed230044.mp4`: cheap policy alone **fails** (cube not pushed to the goal).
2. `samples/mujoco_arbitrated_seed230044.mp4`: same start. The arbiter **escalates once** (overlay turns orange, "EXPERT (escalated)"; the lamp on the table turns orange), the expert acts for 8 steps, control returns to the cheap policy, and the episode **succeeds** using the expert on 8 of 62 steps.
3. `samples/mujoco_expert_seed230044.mp4`: expert alone succeeds, but uses the expert on every step.
4. Optional honesty slide: `samples/mujoco_arbitrated_seed230011.mp4`, where arbitration escalates but still fails.

Pick more episodes with `data/episode_index.csv`, e.g. `condition == arbitrated_a0.30 & success == 1 & escalations >= 1`.

## 6. Provenance
Generated by `panda_gym_demo/composed/part45_calib_bench.py` (`COMPOSED_TASK=push`, deterministic env reset per episode). The full technical report is `results/REPORT_reach_push.md`. Numbers are specific to this task and are not pooled with other project tracks.
