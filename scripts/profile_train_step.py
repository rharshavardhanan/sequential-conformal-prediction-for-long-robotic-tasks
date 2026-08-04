"""Split training cost into data loading vs forward/backward.

If it is I/O bound, more DataLoader workers fixes it cheaply. If it is compute
bound on MPS, the full run belongs on a GPU box and no local tuning will save it.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, RandomSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--batches", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from functools import partial

    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.algos.base import _worker_init_obs_utils
    from libero.lifelong.utils import control_seed, safe_device
    from scripts.common import build_cfg, load_one_task, pick_device

    device = pick_device(args.device)
    cfg = build_cfg(
        [
            "benchmark_name=LIBERO_10",
            "policy=bc_transformer_policy",
            "lifelong=single_task",
            f"device={device}",
            f"train.batch_size={args.batch_size}",
            f"train.num_workers={args.num_workers}",
            "use_wandb=false",
        ]
    )
    control_seed(cfg.seed)

    benchmark, dataset, shape_meta = load_one_task(cfg, args.task_id)
    cfg.shape_meta = shape_meta
    cfg.experiment_dir = "/tmp"
    print(f"device={device} batch={args.batch_size} workers={args.num_workers}")
    print(f"sequences={dataset.total_num_sequences} demos={dataset.n_demos}")
    print(f"batches/epoch = {dataset.total_num_sequences // args.batch_size}")

    obs_modality = {k: list(v) for k, v in cfg.data.obs.modality.items()}
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=RandomSampler(dataset),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=partial(_worker_init_obs_utils, obs_modality)
        if args.num_workers > 0
        else None,
    )

    # --- data only ---
    it = iter(dl)
    next(it)  # warm up workers
    t0 = time.perf_counter()
    for _ in range(args.batches):
        next(it)
    t_data = (time.perf_counter() - t0) / args.batches

    # --- data + fwd/bwd ---
    algo = safe_device(
        get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), cfg.device
    )
    algo.start_task(args.task_id)  # builds optimizer + scheduler
    algo.train()
    it = iter(dl)
    algo.observe(next(it))  # warm up MPS kernels
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.batches):
        algo.observe(next(it))
    if device == "mps":
        torch.mps.synchronize()
    t_full = (time.perf_counter() - t0) / args.batches

    n_batches = dataset.total_num_sequences // args.batch_size
    print("\n" + "=" * 62)
    print(f"  data only        : {t_data * 1000:8.1f} ms/batch")
    print(f"  data + fwd/bwd   : {t_full * 1000:8.1f} ms/batch")
    print(f"  compute (approx) : {(t_full - t_data) * 1000:8.1f} ms/batch")
    print(f"  -> bound by      : {'DATA' if t_data > (t_full - t_data) else 'COMPUTE'}")
    print(
        f"\n  projected epoch  : {t_full * n_batches / 60:6.2f} min ({n_batches} batches)"
    )
    print(f"  50 epochs x 1 task : {t_full * n_batches * 50 / 3600:6.1f} h")
    print(f"  50 epochs x 10 tasks: {t_full * n_batches * 50 * 10 / 3600:6.1f} h")


if __name__ == "__main__":
    main()
