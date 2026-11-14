"""Collect rollouts and the planned-vs-executed signal the CP monitor is built on.

  .venv-lerobot/bin/python -m scripts.collect_rollouts --task-id 0 --n-rollouts 150

Spec v10 Part D. Everything that defines a rollout is a pure function of its
GLOBAL index, so collection can be split across sessions and days: rollout 87
gets the same init state, seed and split whether it is produced in session 1 or
session 3. Wall-clock time is not part of the data-generating distribution.

The signal, per control step t:

    commanded_delta = action[:3] * 0.05          # metres; controller rescales
    planned_pos     = eef_pos[t] + commanded_delta
    executed_pos    = eef_pos[t+1]
    err_raw         = || planned_pos - executed_pos ||
    err_norm        = err_raw / || commanded_delta ||

`err_norm` exists because OSC is compliant: realised motion is a near-constant
0.251 of the commanded delta, so err_raw is ~0.75x the commanded delta and mostly
reports commanded speed rather than anomaly. Low-motion steps are MASKED, never
dropped -- creeping is the stall signature a dwell-time detector consumes.
"""

import argparse
import json
import os
import time
import traceback
import uuid

import h5py
import numpy as np
import torch

POS_SCALE = 0.05  # OSC_POSE output_max, confirmed in action_space.json
ROT_SCALE = 0.5
EPS = 1e-8
MAX_DELTA = POS_SCALE * np.sqrt(3)  # largest commandable position step
MOTION_FLOOR_FRAC = 0.02  # D.9; calibrated in the Spec v8 audit
BASE_SEED = 20260805  # D.7


def assign(task_id, gidx, n_states, option):
    """D.1/D.7/D.8 -- everything derived from the GLOBAL rollout index."""
    seed = BASE_SEED + task_id * 100000 + gidx
    if option == "A":
        init_idx = gidx  # requires pool >= n_rollouts; enforced by the caller
    else:
        init_idx = int(np.random.default_rng(seed).integers(0, n_states))
    split = "cal" if gidx % 2 == 0 else "eval"
    return seed, init_idx, split


def load_pool(task_id, task, init_file):
    """D.1 -- prefer the extended pool; fall back to the shipped file."""
    from libero.libero import get_libero_path

    ext = f"data/init_states/task{task_id}_init_160.npy"
    if init_file == "extended" and os.path.exists(ext):
        return np.load(ext), ext, "A"
    name = (task.init_states_file if init_file == "pruned"
            else task.init_states_file.replace(".pruned_init", ".init"))
    p = os.path.join(get_libero_path("init_states"), task.problem_folder, name)
    return np.asarray(torch.load(p, weights_only=False)), p, "B"


def scan_existing(path):
    """D.13 -- only groups with complete==True count as done."""
    complete, incomplete = set(), set()
    if not os.path.exists(path):
        return complete, incomplete
    with h5py.File(path, "r") as f:
        for k in f:
            if not k.startswith("rollout_"):
                continue
            (complete if bool(f[k].attrs.get("complete", False)) else incomplete).add(k)
    return complete, incomplete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--n-rollouts", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--outdir", default="data/rollouts")
    ap.add_argument("--suffix", default="_v10")
    ap.add_argument("--init-file", choices=["extended", "full", "pruned"], default="extended")
    ap.add_argument("--no-images", action="store_true",
                    help="disable image capture. Images are NOT recoverable later "
                         "and FAIL-Detect needs them, so they are ON by default.")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="F.1: sigma of zero-mean Gaussian noise added to the action "
                         "before env.step. Part of the fingerprint.")
    args = ap.parse_args()

    images = not args.no_images
    repo = args.repo or f"yzembodied/libero10_task{args.task_id}_diffusion"
    session_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from scripts.dp_policy import DiffusionPolicyRunner
    from scripts import fingerprint as FP

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

    pool, pool_src, option = load_pool(args.task_id, task, args.init_file)
    if option == "A" and args.n_rollouts > pool.shape[0]:
        raise SystemExit(f"[abort] Option A needs pool >= n_rollouts "
                         f"({pool.shape[0]} < {args.n_rollouts}); run gen_init_states.py")

    runner = DiffusionPolicyRunner(repo, device=args.device)
    policy_sha = runner.fingerprint()
    n_act = runner.n_action_steps

    env_cfg = {"camera_heights": 256, "camera_widths": 256, "max_steps": args.max_steps,
               "control_freq": 20, "settle_steps": 5, "bddl_file": os.path.basename(bddl)}
    pert = {"type": "action_noise", "sigma": args.action_noise} if args.action_noise > 0 else {}
    fp_fields, fp_digest = FP.build(repo, policy_sha, FP.array_sha256(pool), env_cfg, pert)

    print(f"[session] {session_id}", flush=True)
    print(f"[task {args.task_id}] {task.language}")
    print(f"[policy] {repo}  sha256 {policy_sha[:16]}...")
    print(f"[init] option {option}, pool {pool.shape} from {pool_src}")
    print(f"[images] {'ON' if images else 'OFF -- FAIL-Detect will be impossible'}")
    print(f"[perturbation] {pert or 'none'}")
    print(f"[fingerprint] {fp_digest}")
    for k, v in fp_fields.items():
        if k not in ("env_config", "perturbation"):
            print(f"    {k:18s} {v[:70]}")

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"task{args.task_id}{args.suffix}.hdf5")

    # ---- D.5 fingerprint enforcement -------------------------------------
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            stored = json.loads(f.attrs["fingerprint_fields"])
        diffs = FP.compare(stored, fp_fields)
        if diffs:
            print("\n[ABORT] fingerprint MISMATCH -- the distribution changed.")
            print("        This task must RESTART, not resume.")
            for k, s, c in diffs:
                print(f"          {k}:\n            stored : {s}\n            current: {c}")
            raise SystemExit(1)
        print("[fingerprint] matches stored -- safe to resume")

    complete, incomplete = scan_existing(path)
    print(f"\n[resume] complete={len(complete)}  incomplete={len(incomplete)} (will overwrite)  "
          f"remaining={args.n_rollouts - len(complete)}")

    with h5py.File(path, "a") as f:
        f.attrs.update({
            "policy_repo": repo, "policy_sha256": policy_sha,
            "task_id": args.task_id, "task_name": task.name,
            "task_language": task.language, "bddl_file": bddl,
            "pos_scale": POS_SCALE, "rot_scale": ROT_SCALE, "control_freq": 20,
            "max_steps": args.max_steps, "init_pool_source": pool_src,
            "init_option": option, "n_init_states": int(pool.shape[0]),
            "osc_gain_reference": 0.251, "motion_floor_frac": MOTION_FLOOR_FRAC,
            "motion_floor_abs": MOTION_FLOOR_FRAC * MAX_DELTA,
            "base_seed": BASE_SEED, "n_action_steps": n_act,
            "images": bool(images), "action_noise_sigma": args.action_noise,
            "fingerprint": fp_digest, "fingerprint_fields": json.dumps(fp_fields),
        })
        for k, v in fp_fields.items():
            f.attrs[f"env_{k}"] = v

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    goal_state = env.env.parsed_problem["goal_state"]
    print(f"[predicates] {len(goal_state)}: {goal_state}\n", flush=True)

    errors, done_this = [], 0
    t_start = time.perf_counter()

    for gidx in range(args.n_rollouts):
        gname = f"rollout_{gidx:04d}"
        if gname in complete:
            continue
        seed, init_idx, split = assign(args.task_id, gidx, pool.shape[0], option)

        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.reset()
            obs = env.set_init_state(pool[init_idx])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))
            runner.reset()

            keys = ("actions", "eef_pos", "eef_quat", "gripper_qpos", "joint_pos",
                    "planned_pos", "executed_pos", "err_raw", "err_norm", "cmd_norm",
                    "object_state", "predicates", "action_chunk", "chunk_len",
                    "chunk_is_new")
            rec = {k: [] for k in keys}
            imgs = {"agentview": [], "eye_in_hand": []} if images else None

            t0, success, steps = time.perf_counter(), False, 0
            while steps < args.max_steps and not success:
                eef_pos = obs["robot0_eef_pos"].copy()
                act, chunk, is_new = runner.act_with_chunk(obs)

                # F.1 perturbation, applied BEFORE the score is derived.
                # The controller tracks the perturbed command, so deriving
                # planned_pos from the clean action would count the injected
                # noise as tracking error and the monitor would be detecting
                # our own noise rather than the failure it caused. cmd_norm
                # uses the same vector, so the score's numerator and
                # denominator agree about what was commanded.
                # dim 6 (gripper) is never perturbed: it is bang-bang, phase
                # segmentation thresholds on its sign, and noise there would
                # fragment the labels. Leaving it clean also makes the executed
                # gripper channel equal the policy's, so segmentation is
                # unambiguous by construction.
                step_act = act.copy()
                if args.action_noise > 0:
                    noise = np.random.normal(0.0, args.action_noise, step_act.shape)
                    noise[6] = 0.0
                    step_act = np.clip(step_act + noise, -1.0, 1.0)

                cmd = step_act[:3] * POS_SCALE
                planned = eef_pos + cmd

                padded = np.zeros((n_act, 7), np.float32)
                padded[: min(len(chunk), n_act)] = chunk[:n_act]
                rec["action_chunk"].append(padded)
                rec["chunk_len"].append(min(len(chunk), n_act))
                rec["chunk_is_new"].append(bool(is_new))

                rec["actions"].append(step_act.copy())  # executed command
                rec["eef_pos"].append(eef_pos)
                rec["eef_quat"].append(obs["robot0_eef_quat"].copy())
                rec["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
                rec["joint_pos"].append(obs["robot0_joint_pos"].copy())
                rec["planned_pos"].append(planned)
                rec["object_state"].append(obs["object-state"].copy())
                if imgs is not None:
                    imgs["agentview"].append(obs["agentview_image"][::-1][::2, ::2].copy())
                    imgs["eye_in_hand"].append(
                        obs["robot0_eye_in_hand_image"][::-1][::2, ::2].copy())

                obs, _, success, _ = env.step(step_act)

                rec["predicates"].append(
                    [bool(env.env._eval_predicate(g)) for g in goal_state])
                executed = obs["robot0_eef_pos"].copy()
                cmd_n = float(np.linalg.norm(cmd))
                err = float(np.linalg.norm(planned - executed))
                rec["executed_pos"].append(executed)
                rec["err_raw"].append(err)
                rec["cmd_norm"].append(cmd_n)
                rec["err_norm"].append(err / max(cmd_n, EPS))
                steps += 1

            dt = time.perf_counter() - t0
            cmd_norm = np.asarray(rec["cmd_norm"])
            motion_mask = cmd_norm >= MOTION_FLOOR_FRAC * MAX_DELTA  # D.9
            grip = np.asarray(rec["actions"])[:, 6]
            regrasps = int(((grip[1:] > 0) & (grip[:-1] <= 0)).sum())  # D.11

            with h5py.File(path, "a") as f:
                if gname in f:
                    del f[gname]  # D.13 overwrite an incomplete group
                g = f.create_group(gname)
                for k, v in rec.items():
                    g.create_dataset(k, data=np.asarray(v), compression="gzip",
                                     compression_opts=4)
                g.create_dataset("motion_mask", data=motion_mask, compression="gzip",
                                 compression_opts=4)
                if imgs is not None:
                    gi = g.create_group("images")
                    for k, v in imgs.items():
                        gi.create_dataset(k, data=np.asarray(v, np.uint8),
                                          compression="gzip", compression_opts=4)
                g.attrs.update({  # D.6 self-describing rollout
                    "success": bool(success), "steps": steps,
                    "global_index": gidx, "init_state_index": init_idx,
                    "split": split, "seed": seed, "regrasp_count": regrasps,
                    "wall_sec": dt, "session_id": session_id,
                    "policy_sha256": policy_sha, "fingerprint": fp_digest,
                    "action_noise_sigma": args.action_noise,
                    "n_masked_steps": int((~motion_mask).sum()),
                    "git_revision": fp_fields["git_revision"],
                    "torch_version": fp_fields["torch"],
                    "mujoco_version": fp_fields["mujoco"],
                    "robosuite_version": fp_fields["robosuite"],
                    "lerobot_version": fp_fields["lerobot"],
                })
                g.attrs["complete"] = True  # D.13 written LAST
            done_this += 1
            en = np.asarray(rec["err_norm"])
            print(f"  {gname}[{split}] {'SUCCESS' if success else 'fail   '} {steps:3d} steps "
                  f"{dt:6.1f}s {dt/steps*1000:5.1f}ms/st  err_norm={en.mean():.3f} "
                  f"regrasp={regrasps} masked={int((~motion_mask).sum())}", flush=True)

        except Exception:
            tb = traceback.format_exc()
            errors.append((gname, tb.strip().split("\n")[-1]))
            print(f"  {gname}: ERROR -- {tb.strip().split(chr(10))[-1]}", flush=True)
            with h5py.File(path, "a") as f:  # D.12 record, do not abort
                if gname in f:
                    del f[gname]
                g = f.create_group(gname)
                g.attrs.update({"complete": False, "error": tb[-2000:],
                                "global_index": gidx, "session_id": session_id})
            continue

    total = time.perf_counter() - t_start
    complete, incomplete = scan_existing(path)

    # ---- D.14 session manifest -------------------------------------------
    man_path = os.path.join(args.outdir, "session_log.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"sessions": []}
    man["sessions"].append({
        "session_id": session_id, "task_id": args.task_id, "repo": repo,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - total)),
        "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": os.uname().nodename,
        "attempted_range": [0, args.n_rollouts - 1], "completed_this_session": done_this,
        "total_complete": len(complete), "fingerprint": fp_digest,
        "errors": [{"rollout": r, "error": e} for r, e in errors],
        "wall_sec": total, "action_noise_sigma": args.action_noise,
    })
    json.dump(man, open(man_path, "w"), indent=2)

    with h5py.File(path, "r") as f:
        ks = sorted(k for k in f if k.startswith("rollout_") and f[k].attrs.get("complete"))
        succ = sum(int(f[k].attrs["success"]) for k in ks)
        cal = sum(1 for k in ks if f[k].attrs["split"] == "cal")
    print("\n" + "=" * 68)
    print(f"  session      : {session_id}   this session: {done_this} rollouts")
    print(f"  task total   : {len(ks)}/{args.n_rollouts} complete  ({succ} success, "
          f"{len(ks)-succ} fail, rate {succ/max(1,len(ks)):.3f})")
    print(f"  split        : {cal} cal / {len(ks)-cal} eval")
    print(f"  errors       : {len(errors)}")
    for r, e in errors:
        print(f"      {r}: {e}")
    print(f"  wall         : {total/60:.1f} min")
    print(f"  file         : {path} ({os.path.getsize(path)/1e6:.0f} MB)")
    print(f"  manifest     : {man_path}")
    print("=" * 68)
    env.close()


if __name__ == "__main__":
    main()
