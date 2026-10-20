"""Success rate of BC-RNN checkpoints, run locally where rendering works.

  .venv/bin/python -m scripts.eval_bcrnn --ckpt-dir data/checkpoints --n-eval 20

Selection metric is SUCCESS, not validation loss. Validation loss measures how
well the policy reproduces human actions; success measures whether it recovers
from states the demonstrations never visited. They diverge routinely -- best-val
was epoch 14, but the in-band checkpoint may be elsewhere.

Target band 40-80%: for the arbitration cheap arm, above it there are too few
failures to escalate on, below it the policy is not a sensible default.

Runs in .venv (robomimic + robosuite 1.4.1 + mujoco 3.2.7), NOT .venv-lerobot --
BCRNNPolicy needs robomimic, which the lerobot env does not carry.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import time

import numpy as np
import torch


def sha256_file(p, _c=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(_c), b""):
            h.update(b)
    return h.hexdigest()


def build_cfg(overrides):
    import pathlib

    import libero
    import yaml
    from easydict import EasyDict
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from libero.libero import get_libero_path

    cfg_dir = pathlib.Path(libero.__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(
            compose(config_name="config", overrides=overrides))))
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="data/checkpoints")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--num-procs", type=int, default=1,
                    help="1 => DummyVectorEnv, no fork. Raise only if stable.")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--policy", default="bc_rnn_policy",
                    choices=["bc_rnn_policy", "bc_transformer_policy"])
    ap.add_argument("--glob", default="bcrnn_task*_ep*.pth")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="data/results/bcrnn_eval.json")
    args = ap.parse_args()

    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import SequenceVLDataset, get_dataset
    from libero.lifelong.metric import evaluate_one_task_success
    from libero.lifelong.utils import (confidence_interval, control_seed,
                                       get_task_embs, safe_device)

    cfg = build_cfg([
        "benchmark_name=LIBERO_10", f"policy={args.policy}", "lifelong=single_task",
        f"device={args.device}", f"eval.n_eval={args.n_eval}",
        f"eval.num_procs={args.num_procs}", f"eval.max_steps={args.max_steps}",
        "eval.use_mp=true", "use_wandb=false",
    ])
    control_seed(cfg.seed)

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    ds, shape_meta = get_dataset(
        dataset_path=os.path.join(cfg.folder,
                                  benchmark.get_task_demonstration(args.task_id)),
        obs_modality=cfg.data.obs.modality, initialize_obs_utils=True,
        seq_len=cfg.data.seq_len)
    embs = get_task_embs(cfg, [benchmark.get_task(i).language
                               for i in range(benchmark.n_tasks)])
    benchmark.set_task_embs(embs)
    cfg.shape_meta = shape_meta
    cfg.experiment_dir = "/tmp"
    task = benchmark.get_task(args.task_id)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.glob)))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_dir}")
    print(f"[task {args.task_id}] {task.language}")
    print(f"[eval] {len(ckpts)} checkpoints x {args.n_eval} rollouts "
          f"(max {args.max_steps} steps, {args.num_procs} env)\n")

    rows = []
    for c in ckpts:
        m = re.search(r"_ep(\d+)\.pth$", c)
        ep = int(m.group(1)) if m else -1
        sd = torch.load(c, map_location=args.device, weights_only=False)
        algo = safe_device(
            get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), args.device)
        algo.policy.load_state_dict(sd["state_dict"])
        algo.eval()

        t0 = time.perf_counter()
        sr = evaluate_one_task_success(
            cfg=cfg, algo=algo, task=task,
            task_emb=benchmark.get_task_emb(args.task_id), task_id=args.task_id)
        wall = time.perf_counter() - t0
        n = int(round(sr * args.n_eval))
        ci = confidence_interval(sr, args.n_eval)
        band = 0.40 <= sr <= 0.80
        verdict = "IN BAND" if band else ("TOO HIGH" if sr > 0.80 else "too low")
        rows.append({"epoch": ep, "checkpoint": c, "sha256": sha256_file(c),
                     "val_loss": sd.get("monitor"), "n_eval": args.n_eval,
                     "n_success": n, "success_rate": float(sr), "ci95": float(ci),
                     "in_band_40_80": bool(band), "wall_sec": wall})
        print(f"  ep{ep:03d}  val {sd.get('monitor', float('nan')):8.3f}  "
              f"success {sr:.2f} ({n:2d}/{args.n_eval}) +/-{ci:.2f}  "
              f"{verdict:8s}  {wall/60:4.1f} min", flush=True)
        del algo
        if args.device == "mps":
            torch.mps.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"task_id": args.task_id, "task": task.language,
               "n_eval": args.n_eval, "results": rows}, open(args.out, "w"), indent=2)

    print("\n" + "=" * 62)
    inb = [r for r in rows if r["in_band_40_80"]]
    if inb:
        best = max(inb, key=lambda r: abs(r["success_rate"] - 0.55))
        pick = min(inb, key=lambda r: abs(r["success_rate"] - 0.55))
        print(f"  IN BAND: epochs {[r['epoch'] for r in inb]}")
        print(f"  closest to mid-band (0.55): epoch {pick['epoch']} "
              f"at {pick['success_rate']:.2f}")
        print(f"  sha256: {pick['sha256']}")
    else:
        hi = [r for r in rows if r["success_rate"] > 0.80]
        print("  NO checkpoint in 40-80%.")
        print("  all too high -- take an earlier epoch" if hi else
              "  all too low -- either we stopped training too early, or BC-RNN\n"
              "  cannot reach a useful success rate on LIBERO-Long. Check whether\n"
              "  success rises with epoch: if it does, retrain to 50 epochs with\n"
              "  patience disabled; if flat, the cheap arm needs reconsidering.")
    print(f"  wrote {args.out}")
    print("=" * 62)


if __name__ == "__main__":
    main()
