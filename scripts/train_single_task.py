"""Train the BC policy on ONE libero_10 task, and time it.

Purpose is sizing: measure per-epoch training cost and per-eval cost so the
full 10-task run can be projected before committing to it.

  uv run python scripts/train_single_task.py --task-id 0 --n-epochs 3 --eval-every 3

Note the in-training eval is not optional upstream: `learn_one_task` only ever
writes a checkpoint inside the `epoch % eval_every == 0` block, keeping the
best-success-so-far model. Setting eval_every > n_epochs would train happily and
then crash at the end with no checkpoint to load.
"""

import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--num-procs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument(
        "--summary-out",
        default=None,
        help="also write the summary JSON here (used by train_all_tasks)",
    )
    args = ap.parse_args()

    from scripts.common import build_cfg, load_one_task, pick_device, sha256_file

    device = pick_device(args.device)
    cfg = build_cfg(
        [
            "benchmark_name=LIBERO_10",
            "policy=bc_transformer_policy",
            "lifelong=single_task",
            f"seed={args.seed}",
            f"device={device}",
            f"train.n_epochs={args.n_epochs}",
            f"train.batch_size={args.batch_size}",
            f"train.num_workers={args.num_workers}",
            f"eval.eval_every={args.eval_every}",
            f"eval.n_eval={args.n_eval}",
            f"eval.num_procs={args.num_procs}",
            "eval.use_mp=true",
            "use_wandb=false",
        ]
    )

    import torch
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import control_seed, create_experiment_dir, safe_device

    control_seed(cfg.seed)
    print(
        f"[cfg] device={device}  task_id={args.task_id}  n_epochs={args.n_epochs} "
        f"eval_every={args.eval_every}  n_eval={args.n_eval}  num_procs={args.num_procs}"
    )

    t_load = time.perf_counter()
    benchmark, dataset, shape_meta = load_one_task(cfg, args.task_id)
    t_load = time.perf_counter() - t_load

    task = benchmark.get_task(args.task_id)
    print(f"[data] task: {task.name}")
    print(f"[data] language: {task.language}")
    print(
        f"[data] demos={dataset.n_demos}  sequences={dataset.total_num_sequences}  "
        f"ac_dim={shape_meta['ac_dim']}  load={t_load:.1f}s"
    )

    assert shape_meta["ac_dim"] == 7, (
        f"expected 7-dim action, got {shape_meta['ac_dim']}"
    )

    cfg.shape_meta = shape_meta
    create_experiment_dir(cfg)
    print(f"[exp ] {cfg.experiment_dir}")

    algo = safe_device(
        get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), cfg.device
    )

    n_params = sum(p.numel() for p in algo.policy.parameters())
    print(
        f"[algo] {cfg.lifelong.algo} / {cfg.policy.policy_type}  "
        f"{n_params / 1e6:.1f}M params"
    )

    t0 = time.perf_counter()
    algo.train()
    s_fwd, l_fwd = algo.learn_one_task(dataset, args.task_id, benchmark, {})
    wall = time.perf_counter() - t0

    ckpt = f"{cfg.experiment_dir}/task{args.task_id}_model.pth"
    digest = sha256_file(ckpt)

    n_evals = args.n_epochs // args.eval_every + 1
    print("\n" + "=" * 70)
    print("TIMING")
    print("=" * 70)
    print(f"  wall clock              : {wall / 60:.2f} min")
    print(f"  epochs trained          : {args.n_epochs}  (+1 zero-shot pass)")
    print(f"  evals run               : {n_evals} x {args.n_eval} rollouts")
    print(f"  success AUC (s_fwd)     : {s_fwd:.3f}")
    print(f"  checkpoint              : {ckpt}")
    print(f"  sha256                  : {digest}")

    summary = {
        "task_id": args.task_id,
        "task_name": task.name,
        "device": device,
        "n_epochs": args.n_epochs,
        "eval_every": args.eval_every,
        "n_eval": args.n_eval,
        "num_procs": args.num_procs,
        "batch_size": args.batch_size,
        "n_params_M": n_params / 1e6,
        "n_demos": int(dataset.n_demos),
        "n_sequences": int(dataset.total_num_sequences),
        "wall_sec": wall,
        "s_fwd": float(s_fwd),
        "checkpoint": ckpt,
        "sha256": digest,
    }
    targets = [f"{cfg.experiment_dir}/timing.json"]
    if args.summary_out:
        targets.append(args.summary_out)
    for t in targets:
        with open(t, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {t}")


if __name__ == "__main__":
    main()
