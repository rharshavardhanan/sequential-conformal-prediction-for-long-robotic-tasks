"""Train LIBERO's BC-RNN on one task, for the policy-arbitration cheap arm.

Designed to run on Colab (T4) unattended:

  python -m scripts.train_bc_rnn --task-id 0 --n-epochs 75 \
      --out /content/drive/MyDrive/bcrnn --save-every 5 --patience 8

Why a custom loop instead of LIBERO's `learn_one_task`: that function only ever
writes a checkpoint INSIDE its `epoch % eval_every == 0` block, keyed on success
rate improving -- and computing success needs MuJoCo rollouts. Training here is
pure behaviour cloning on stored demonstrations, no simulator, so with eval
disabled `learn_one_task` would train for hours and save nothing.

Training data is LIBERO's HUMAN demonstrations (50/task), never a policy's own
rollouts -- training on Diffusion Policy rollouts would be distillation and would
import that policy's failure modes into BC-RNN.

Plateau stopping: patience on validation loss when a demo-level holdout is
available, else on smoothed training loss. The latter measures optimisation
convergence, NOT generalisation -- which is why a checkpoint is kept every
--save-every epochs regardless, so the operating point (target 40-80% success
for arbitration) can be chosen afterwards by evaluating checkpoints rather than
guessed now.
"""

import argparse
import hashlib
import json
import os
import pathlib
import time

import numpy as np
import torch


def sha256_file(p, _c=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(_c), b""):
            h.update(b)
    return h.hexdigest()


def versions():
    def v(m):
        try:
            return str(getattr(__import__(m), "__version__", "unknown"))
        except Exception as e:
            return f"IMPORT_FAIL:{type(e).__name__}"
    return {k: v(k) for k in ("torch", "numpy", "robomimic", "h5py", "torchvision")}


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_cfg(overrides):
    """Compose LIBERO's own hydra config with our overrides, as an EasyDict.

    CONFIG_DIR is derived from the installed `libero` package rather than a repo
    path, so this works identically on Colab and locally.
    """
    import libero
    import yaml
    from easydict import EasyDict
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from libero.libero import get_libero_path

    cfg_dir = pathlib.Path(libero.__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        hydra_cfg = compose(config_name="config", overrides=overrides or [])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    return cfg


def load_one_task(cfg, task_id):
    """Dataset + shape_meta for one task; task embeddings for all.

    Embeddings are computed for every task description because `benchmark`
    indexes them by global task id, but only one hdf5 is ever opened.
    """
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.datasets import SequenceVLDataset, get_dataset
    from libero.lifelong.utils import get_task_embs

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    dataset, shape_meta = get_dataset(
        dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(task_id)),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    return benchmark, SequenceVLDataset(dataset, task_embs[task_id]), shape_meta


def demo_level_split(ds, val_frac, seed):
    """Split sequence indices by DEMO so no demo appears in both halves.

    Sequences from one demonstration overlap heavily; splitting at the sequence
    level leaks and makes validation loss meaninglessly optimistic.
    """
    inner = getattr(ds, "sequence_dataset", ds)
    m = getattr(inner, "_index_to_demo_id", None)
    if m is None:
        return None, None
    idx2demo = np.array([m[i] for i in range(len(inner))])
    demos = np.unique(idx2demo)
    rng = np.random.default_rng(seed)
    rng.shuffle(demos)
    n_val = max(1, int(len(demos) * val_frac))
    val_demos = set(demos[:n_val].tolist())
    val = np.flatnonzero([d in val_demos for d in idx2demo])
    tr = np.flatnonzero([d not in val_demos for d in idx2demo])
    return tr, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=75)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8,
                    help="epochs without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="improvement smaller than this does not reset patience")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    args = ap.parse_args()

    from torch.utils.data import DataLoader, RandomSampler, Subset

    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import control_seed, safe_device

    device = pick_device("auto")
    cfg = build_cfg([
        "benchmark_name=LIBERO_10",
        "policy=bc_rnn_policy",          # <- the cheap arm
        "lifelong=single_task",
        f"seed={args.seed}", f"device={device}",
        f"train.n_epochs={args.n_epochs}",
        f"train.batch_size={args.batch_size}",
        f"train.num_workers={args.num_workers}",
        "use_wandb=false",
    ])
    control_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"[cfg] device={device} policy={cfg.policy.policy_type} "
          f"epochs={args.n_epochs} batch={args.batch_size}")
    benchmark, dataset, shape_meta = load_one_task(cfg, args.task_id)
    task = benchmark.get_task(args.task_id)
    cfg.shape_meta = shape_meta
    cfg.experiment_dir = args.out
    print(f"[data] task {args.task_id}: {task.language}")
    print(f"[data] {dataset.n_demos} demos, {dataset.total_num_sequences} sequences, "
          f"ac_dim={shape_meta['ac_dim']}")
    assert shape_meta["ac_dim"] == 7

    tr_idx, val_idx = demo_level_split(dataset, args.val_frac, args.seed)
    if tr_idx is None:
        print("[val ] demo-level split unavailable -> plateau on SMOOTHED TRAIN LOSS "
              "(measures optimisation convergence, not generalisation)")
        train_ds, val_ds = dataset, None
    else:
        train_ds, val_ds = Subset(dataset, tr_idx), Subset(dataset, val_idx)
        print(f"[val ] demo-level holdout: {len(tr_idx)} train / {len(val_idx)} val seqs")

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          sampler=RandomSampler(ds) if shuffle else None,
                          persistent_workers=args.num_workers > 0)

    train_dl = loader(train_ds, True)
    val_dl = loader(val_ds, False) if val_ds is not None else None

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), device)
    algo.start_task(args.task_id)
    n_par = sum(p.numel() for p in algo.policy.parameters())
    print(f"[algo] {cfg.policy.policy_type}  {n_par/1e6:.1f}M params  "
          f"{len(train_dl)} batches/epoch\n")

    use_amp = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    hist, best, bad, stopped = [], float("inf"), 0, None
    meta = {"task_id": args.task_id, "task": task.language, "device": device,
            "policy": cfg.policy.policy_type, "seed": args.seed,
            "batch_size": args.batch_size, "n_params": int(n_par),
            "versions": versions(), "amp": use_amp,
            "train_data": "LIBERO human demonstrations (50/task)"}

    t_start = time.perf_counter()
    for ep in range(1, args.n_epochs + 1):
        algo.train(); t0 = time.perf_counter(); tot = 0.0
        for data in train_dl:
            if use_amp:
                with torch.cuda.amp.autocast():
                    tot += algo.observe(data)
            else:
                tot += algo.observe(data)
        tr_loss = tot / len(train_dl)

        if val_dl is not None:
            algo.eval(); v = 0.0
            with torch.no_grad():
                for data in val_dl:
                    v += algo.eval_observe(data)
            monitor = v / len(val_dl); tag = "val"
        else:
            monitor = float(np.mean([h["train"] for h in hist[-2:]] + [tr_loss])); tag = "train~"

        dt = time.perf_counter() - t0
        hist.append({"epoch": ep, "train": float(tr_loss), "monitor": float(monitor),
                     "sec": dt})
        improved = monitor < best - args.min_delta
        if improved:
            best, bad = monitor, 0
        else:
            bad += 1
        print(f"  ep {ep:3d}/{args.n_epochs}  train {tr_loss:8.4f}  {tag} {monitor:8.4f}  "
              f"{dt/60:4.2f} min  {'* best' if improved else f'patience {bad}/{args.patience}'}",
              flush=True)

        if ep % args.save_every == 0 or improved or ep == args.n_epochs:
            p = os.path.join(args.out, f"bcrnn_task{args.task_id}_ep{ep:03d}.pth")
            torch.save({"state_dict": algo.policy.state_dict(), "cfg": cfg,
                        "epoch": ep, "train_loss": tr_loss, "monitor": monitor}, p)
            if improved:
                torch.save({"state_dict": algo.policy.state_dict(), "cfg": cfg,
                            "epoch": ep, "train_loss": tr_loss, "monitor": monitor},
                           os.path.join(args.out, f"bcrnn_task{args.task_id}_best.pth"))
            json.dump({**meta, "history": hist, "best_monitor": best,
                       "stopped_early": stopped},
                      open(os.path.join(args.out, f"history_task{args.task_id}.json"), "w"),
                      indent=2)

        if bad >= args.patience:
            stopped = ep
            print(f"\n[stop] no improvement > {args.min_delta} for {args.patience} epochs "
                  f"-- plateaued at epoch {ep} (best {tag} {best:.4f})")
            break

    total = time.perf_counter() - t_start
    ckpts = sorted(f for f in os.listdir(args.out) if f.endswith(".pth"))
    hashes = {c: sha256_file(os.path.join(args.out, c)) for c in ckpts}
    json.dump({**meta, "history": hist, "best_monitor": best, "stopped_early": stopped,
               "epochs_run": len(hist), "wall_sec": total, "sha256": hashes},
              open(os.path.join(args.out, f"history_task{args.task_id}.json"), "w"), indent=2)

    print("\n" + "=" * 66)
    print(f"  epochs run   : {len(hist)}/{args.n_epochs}"
          f"{'  (early stop)' if stopped else ''}")
    print(f"  best {tag:6s}  : {best:.4f}")
    print(f"  wall         : {total/3600:.2f} h   ({np.mean([h['sec'] for h in hist])/60:.2f} min/epoch)")
    print(f"  checkpoints  : {len(ckpts)} in {args.out}")
    print(f"  HANDOFF: send checkpoints + history_task{args.task_id}.json together --")
    print(f"           it carries the sha256 of every file and the library versions.")
    print("=" * 66)


if __name__ == "__main__":
    main()
