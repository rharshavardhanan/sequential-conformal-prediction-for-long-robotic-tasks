"""Measure the success rate of a pretrained LeRobot diffusion policy on LIBERO.

Runs in the SEPARATE .venv-lerobot (lerobot needs numpy 2.x; the training env is
pinned <1.24 for robomimic):

  .venv-lerobot/bin/python scripts/eval_pretrained_dp.py --n-eval 20

Why the manual normalization: this checkpoint stores its stats as
`normalize_inputs.*` / `unnormalize_outputs.*` inside model.safetensors, which is
the pre-0.4 LeRobot layout. lerobot 0.4.4 moved to external processor files and
DISCARDS those keys on load ("Unexpected key(s)"), leaving the policy with no
stats at all. Applying them here is not a workaround so much as making the
contract explicit -- which is what we want anyway, since the whole project rests
on knowing exactly what action space the policy emits.

Verified: unnormalize_outputs.buffer_action spans [-0.938, 0.938] with the
gripper at exactly +/-1.0, i.e. LIBERO's native 7-dim delta space, so the
`action[:3] * 0.05` setpoint rule applies unchanged.
"""

import argparse
import json
import time

import numpy as np
import torch


def minmax_norm(x, lo, hi):
    return ((x - lo) / (hi - lo)) * 2.0 - 1.0


def minmax_denorm(x, lo, hi):
    return (x + 1.0) / 2.0 * (hi - lo) + lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="yzembodied/libero10_task0_diffusion")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--no-flip",
        action="store_true",
        help="do NOT vertically flip camera images before feeding the policy",
    )
    ap.add_argument("--out", default="pretrained_dp_result.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from safetensors.torch import load_file

    dev = torch.device(args.device)
    print(f"[load] {args.repo}")
    policy = DiffusionPolicy.from_pretrained(args.repo).to(dev).eval()

    # stats lerobot 0.4.4 threw away
    raw = load_file(hf_hub_download(args.repo, "model.safetensors"))
    S = {
        k: v.float()
        for k, v in raw.items()
        if k.startswith(("normalize_", "unnormalize_"))
    }
    img_mean = S["normalize_inputs.buffer_observation_images_image.mean"].to(dev)
    img_std = S["normalize_inputs.buffer_observation_images_image.std"].to(dev)
    st_lo = S["normalize_inputs.buffer_observation_state.min"].to(dev)
    st_hi = S["normalize_inputs.buffer_observation_state.max"].to(dev)
    jt_lo = S["normalize_inputs.buffer_observation_state_joint.min"].to(dev)
    jt_hi = S["normalize_inputs.buffer_observation_state_joint.max"].to(dev)
    act_lo = S["unnormalize_outputs.buffer_action.min"].to(dev)
    act_hi = S["unnormalize_outputs.buffer_action.max"].to(dev)
    print(f"[load] recovered {len(S)} normalization tensors")
    print(
        f"[load] action range: [{act_lo.min():.3f}, {act_hi.max():.3f}]  "
        f"gripper +/-{act_hi[6]:.1f}"
    )

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)

    # the same 100 fixed init states LIBERO's own eval uses, so the number is comparable
    init_states = torch.load(
        f"{get_libero_path('init_states')}/{task.problem_folder}/{task.init_states_file}",
        weights_only=False,
    )
    print(f"[env ] task: {task.language}")
    print(f"[env ] init states: {init_states.shape}, using first {args.n_eval}")
    print(f"[env ] image flip: {not args.no_flip}")

    def to_img(a):
        if not args.no_flip:
            a = a[::-1]
        t = (
            torch.from_numpy(np.ascontiguousarray(a)).to(dev).permute(2, 0, 1).float()
            / 255.0
        )
        return ((t - img_mean) / img_std).unsqueeze(0)

    successes, times, lengths = 0, [], []
    for ep in range(args.n_eval):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(5):  # settle physics, as LIBERO's own eval does
            obs, _, _, _ = env.step(np.zeros(7))
        policy.reset()

        t0, done, steps = time.perf_counter(), False, 0
        while steps < args.max_steps and not done:
            st = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ).astype(np.float32)
            jt = obs["robot0_joint_pos"].astype(np.float32)
            batch = {
                "observation.images.image": to_img(obs["agentview_image"]),
                "observation.images.wrist_image": to_img(
                    obs["robot0_eye_in_hand_image"]
                ),
                "observation.state": minmax_norm(
                    torch.from_numpy(st).to(dev), st_lo, st_hi
                ).unsqueeze(0),
                "observation.state.joint": minmax_norm(
                    torch.from_numpy(jt).to(dev), jt_lo, jt_hi
                ).unsqueeze(0),
            }
            with torch.no_grad():
                a_norm = policy.select_action(batch)
            act = minmax_denorm(a_norm.squeeze(0), act_lo, act_hi).cpu().numpy()
            obs, _, done, _ = env.step(act)
            steps += 1

        dt = time.perf_counter() - t0
        successes += int(done)
        times.append(dt)
        lengths.append(steps)
        print(
            f"  ep {ep:2d}: {'SUCCESS' if done else 'fail   '}  "
            f"{steps:3d} steps  {dt:6.1f}s  ({dt / steps * 1000:.0f} ms/step)  "
            f"running {successes}/{ep + 1}",
            flush=True,
        )

    rate = successes / args.n_eval
    ci = 1.96 * np.sqrt(rate * (1 - rate) / args.n_eval)
    in_band = 0.40 <= rate <= 0.80
    print("\n" + "=" * 64)
    print(f"  success rate : {rate:.2f}  ({successes}/{args.n_eval})  +/- {ci:.2f}")
    print(
        f"  mean rollout : {np.mean(times):.1f} s  ({np.mean(times) / np.mean(lengths) * 1000:.0f} ms/step)"
    )
    print(f"  40-80% band  : {'PASS' if in_band else 'FAIL'}")
    print(
        f"  projected    : 100 rollouts x 4 tasks = "
        f"{np.mean(times) * 400 / 3600:.1f} h on this machine"
    )
    print("=" * 64)

    json.dump(
        {
            "repo": args.repo,
            "task_id": args.task_id,
            "task": task.language,
            "n_eval": args.n_eval,
            "n_success": successes,
            "success_rate": rate,
            "ci95": float(ci),
            "in_band_40_80": bool(in_band),
            "mean_rollout_sec": float(np.mean(times)),
            "flip_images": not args.no_flip,
            "device": args.device,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
