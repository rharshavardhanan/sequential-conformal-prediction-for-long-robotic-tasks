"""Run the N-rollout acceptance test against a frozen checkpoint.

Reports success rate with a binomial CI, and stamps the checkpoint's sha256 so
the number is traceable to the exact weights that produced it.

  uv run python -m scripts.eval_checkpoint \
      --checkpoint experiments/.../task0_model.pth --task-id 0 --n-eval 20

Acceptance band is 40-80%: below that there is too little nominal behaviour to
calibrate on, above it too few failures to detect.
"""

import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--num-procs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.common import build_cfg, load_one_task, pick_device, sha256_file

    device = pick_device(args.device)
    digest = sha256_file(args.checkpoint)

    cfg = build_cfg(
        [
            "benchmark_name=LIBERO_10",
            "policy=bc_transformer_policy",
            "lifelong=single_task",
            f"seed={args.seed}",
            f"device={device}",
            f"eval.n_eval={args.n_eval}",
            f"eval.num_procs={args.num_procs}",
            f"eval.max_steps={args.max_steps}",
            "eval.use_mp=true",
            "use_wandb=false",
        ]
    )

    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.metric import evaluate_one_task_success
    from libero.lifelong.utils import (
        confidence_interval,
        control_seed,
        safe_device,
        torch_load_model,
    )

    control_seed(cfg.seed)
    benchmark, _, shape_meta = load_one_task(cfg, args.task_id)
    cfg.shape_meta = shape_meta
    cfg.experiment_dir = "/tmp"

    algo = safe_device(
        get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), cfg.device
    )
    state_dict, _, _ = torch_load_model(args.checkpoint, map_location=device)
    algo.policy.load_state_dict(state_dict)
    algo.eval()

    task = benchmark.get_task(args.task_id)
    print(f"[eval] checkpoint : {args.checkpoint}")
    print(f"[eval] sha256     : {digest}")
    print(f"[eval] task[{args.task_id}]   : {task.name}")
    print(
        f"[eval] rollouts   : {args.n_eval} (max {args.max_steps} steps, {args.num_procs} envs)"
    )

    t0 = time.perf_counter()
    success_rate = evaluate_one_task_success(
        cfg=cfg,
        algo=algo,
        task=task,
        task_emb=benchmark.get_task_emb(args.task_id),
        task_id=args.task_id,
    )
    wall = time.perf_counter() - t0

    n_succ = int(round(success_rate * args.n_eval))
    ci = confidence_interval(success_rate, args.n_eval)
    in_band = 0.40 <= success_rate <= 0.80

    print("\n" + "=" * 62)
    print(
        f"  success rate : {success_rate:.2f}  ({n_succ}/{args.n_eval})  +/- {ci:.2f}"
    )
    print(f"  wall clock   : {wall / 60:.2f} min  ({wall / args.n_eval:.1f} s/rollout)")
    print(f"  40-80% band  : {'PASS' if in_band else 'FAIL'}")
    if not in_band:
        print(
            "    "
            + (
                "too high: too few failures to detect"
                if success_rate > 0.80
                else "too low: too little nominal behaviour to calibrate on"
            )
        )
    print("=" * 62)

    result = {
        "checkpoint": args.checkpoint,
        "sha256": digest,
        "task_id": args.task_id,
        "task_name": task.name,
        "n_eval": args.n_eval,
        "n_success": n_succ,
        "success_rate": float(success_rate),
        "ci95": float(ci),
        "in_band_40_80": bool(in_band),
        "wall_sec": wall,
        "sec_per_rollout": wall / args.n_eval,
        "device": device,
        "max_steps": args.max_steps,
    }
    out = args.out or "eval_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
