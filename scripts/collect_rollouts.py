"""Collect rollouts and the planned-vs-executed signal the CP monitor is built on.

Runs in .venv-lerobot:

  .venv-lerobot/bin/python scripts/collect_rollouts.py --task-id 0 --n-rollouts 100

Each rollout is written to the HDF5 as soon as it finishes, so a crash at
rollout 87 keeps the first 86. Re-running skips rollouts already present.

The signal, per control step t:

    commanded_delta = action[:3] * 0.05          # metres; controller rescales
    planned_pos     = eef_pos[t] + commanded_delta
    executed_pos    = eef_pos[t+1]
    err_raw         = || planned_pos - executed_pos ||
    err_norm        = err_raw / || commanded_delta ||

`err_norm` exists because OSC is compliant: realised motion is a near-constant
0.251 of the commanded delta (sd 0.006), so err_raw is ~0.75x the commanded
delta and mostly reports how fast the policy asked to move rather than anything
anomalous. Dividing by the commanded magnitude removes that. Both are stored --
which one is the better nonconformity score is an open question for the study,
and re-deriving either from raw arrays later is free.
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

POS_SCALE = 0.05  # OSC_POSE output_max, confirmed in action_space.json
ROT_SCALE = 0.5
EPS = 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument(
        "--repo", default=None, help="default: yzembodied/libero10_task<id>_diffusion"
    )
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--outdir", default="rollouts")
    ap.add_argument(
        "--init-file",
        choices=["pruned", "full"],
        default="full",
        help="'full' = *.init (100 states), 'pruned' = *.pruned_init (50, "
        "what LIBERO's own eval uses). 100 distinct starts needs 'full'.",
    )
    ap.add_argument(
        "--images",
        action="store_true",
        help="also store 128px camera frames (FAIL-Detect baseline only; large)",
    )
    args = ap.parse_args()

    repo = args.repo or f"yzembodied/libero10_task{args.task_id}_diffusion"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from scripts.dp_policy import DiffusionPolicyRunner

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

    runner = DiffusionPolicyRunner(repo, device=args.device)
    fp = runner.fingerprint()
    print(f"[policy] {repo}\n[policy] sha256 {fp}")

    init_name = (
        task.init_states_file
        if args.init_file == "pruned"
        else task.init_states_file.replace(".pruned_init", ".init")
    )
    init_states = torch.load(
        os.path.join(get_libero_path("init_states"), task.problem_folder, init_name),
        weights_only=False,
    )
    print(f"[env] task {args.task_id}: {task.language}")
    print(f"[env] init states: {init_name} {init_states.shape}")
    if args.n_rollouts > init_states.shape[0]:
        print(
            f"[warn] {args.n_rollouts} rollouts > {init_states.shape[0]} init states; "
            f"starts will repeat, weakening independence for calibration"
        )

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"task{args.task_id}.hdf5")
    with h5py.File(path, "a") as f:
        f.attrs.update(
            {
                "policy_repo": repo,
                "policy_sha256": fp,
                "task_id": args.task_id,
                "task_name": task.name,
                "task_language": task.language,
                "bddl_file": bddl,
                "pos_scale": POS_SCALE,
                "rot_scale": ROT_SCALE,
                "control_freq": 20,
                "max_steps": args.max_steps,
                "init_states_file": init_name,
                "osc_gain_reference": 0.251,
            }
        )
        existing = set(f.keys())

    done_n = len(existing)
    print(f"[out] {path} ({done_n} rollouts already present)\n")

    t_start = time.perf_counter()
    for i in range(args.n_rollouts):
        gname = f"rollout_{i:04d}"
        if gname in existing:
            continue

        env.reset()
        obs = env.set_init_state(init_states[i % init_states.shape[0]])
        for _ in range(5):  # settle physics, as LIBERO's own eval does
            obs, _, _, _ = env.step(np.zeros(7))
        runner.reset()

        rec = {
            k: []
            for k in (
                "actions",
                "eef_pos",
                "eef_quat",
                "gripper_qpos",
                "joint_pos",
                "planned_pos",
                "executed_pos",
                "err_raw",
                "err_norm",
                "cmd_norm",
                "object_state",
            )
        }
        imgs = {"agentview": [], "eye_in_hand": []} if args.images else None

        t0, success, steps = time.perf_counter(), False, 0
        while steps < args.max_steps and not success:
            eef_pos = obs["robot0_eef_pos"].copy()
            act = runner.act(obs)

            cmd = act[:3] * POS_SCALE
            planned = eef_pos + cmd

            rec["actions"].append(act.copy())
            rec["eef_pos"].append(eef_pos)
            rec["eef_quat"].append(obs["robot0_eef_quat"].copy())
            rec["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
            rec["joint_pos"].append(obs["robot0_joint_pos"].copy())
            rec["planned_pos"].append(planned)
            rec["object_state"].append(obs["object-state"].copy())
            if imgs is not None:
                imgs["agentview"].append(obs["agentview_image"][::-1][::2, ::2].copy())
                imgs["eye_in_hand"].append(
                    obs["robot0_eye_in_hand_image"][::-1][::2, ::2].copy()
                )

            obs, _, success, _ = env.step(act)

            executed = obs["robot0_eef_pos"].copy()
            cmd_n = float(np.linalg.norm(cmd))
            err = float(np.linalg.norm(planned - executed))
            rec["executed_pos"].append(executed)
            rec["err_raw"].append(err)
            rec["cmd_norm"].append(cmd_n)
            rec["err_norm"].append(err / max(cmd_n, EPS))
            steps += 1

        dt = time.perf_counter() - t0
        with h5py.File(path, "a") as f:
            g = f.create_group(gname)
            for k, v in rec.items():
                g.create_dataset(
                    k, data=np.asarray(v), compression="gzip", compression_opts=4
                )
            if imgs is not None:
                gi = g.create_group("images")
                for k, v in imgs.items():
                    gi.create_dataset(
                        k,
                        data=np.asarray(v, dtype=np.uint8),
                        compression="gzip",
                        compression_opts=4,
                    )
            g.attrs.update(
                {
                    "success": bool(success),
                    "steps": steps,
                    "init_state_index": i % init_states.shape[0],
                    "wall_sec": dt,
                    "policy_sha256": fp,
                }
            )
        done_n += 1
        en = np.asarray(rec["err_norm"])
        print(
            f"  {gname}: {'SUCCESS' if success else 'fail   '} {steps:3d} steps "
            f"{dt:6.1f}s  err_norm mean={en.mean():.3f} p95={np.percentile(en, 95):.3f}",
            flush=True,
        )

    with h5py.File(path, "r") as f:
        rs = [k for k in f if k.startswith("rollout_")]
        succ = sum(int(f[k].attrs["success"]) for k in rs)
        allen = np.concatenate([f[k]["err_norm"][:] for k in rs])
    total = time.perf_counter() - t_start
    print("\n" + "=" * 64)
    print(f"  rollouts    : {len(rs)}  ({succ} success, {len(rs) - succ} fail)")
    print(f"  success rate: {succ / len(rs):.2f}")
    print(
        f"  err_norm    : mean={allen.mean():.3f}  p50={np.median(allen):.3f}  "
        f"p95={np.percentile(allen, 95):.3f}"
    )
    print(f"  (expect ~0.75 given the measured OSC gain of 0.251)")
    print(f"  wall        : {total / 60:.1f} min")
    print(f"  file        : {path}  ({os.path.getsize(path) / 1e6:.0f} MB)")
    print("=" * 64)
    env.close()


if __name__ == "__main__":
    main()
