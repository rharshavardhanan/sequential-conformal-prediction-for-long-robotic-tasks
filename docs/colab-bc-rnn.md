# BC-RNN on Colab T4 — 75 epochs with plateau stopping

Paste each block as its own cell. **Runtime → Change runtime type → T4 GPU** first.

Budget: ~2.5 min/epoch on T4 → **~3.1 h for 75 epochs**, inside the 5 h cap but
close enough that checkpointing to Drive is mandatory, not optional.

---

### Cell 1 — confirm the GPU

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch; print(torch.__version__, torch.cuda.is_available())
```

If this says anything other than T4/A100/L4, stop and change the runtime type.

---

### Cell 2 — mount Drive (checkpoints must survive a disconnect)

```python
from google.colab import drive; drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/bcrnn_task0'
import os; os.makedirs(OUT, exist_ok=True); print(OUT)
```

---

### Cell 3 — install LIBERO

```python
%cd /content
!git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
# Upstream ships no libero/__init__.py -- without it a PEP 660 editable install
# maps ZERO packages and `import libero` fails with no error at install time.
!touch /content/LIBERO/libero/__init__.py
!pip install -q -e /content/LIBERO
!pip install -q robomimic==0.2.0 --no-deps
!pip install -q easydict hydra-core einops thop transformers six psutil tensorboardX
# robosuite IS required even though training never renders: LIBERO's import chain
# is algos/__init__ -> base.py -> `from libero.lifelong.metric import *` ->
# metric.py -> `from libero.libero.envs import OffScreenRenderEnv` -> robosuite.
# That import is at module level, so it fires regardless of whether any env is
# ever built. robosuite 1.4.1 runs fine on numpy 2.x despite predating it.
# `future` is an undeclared dependency of bddl.
!pip install -q robosuite==1.4.1 bddl==1.0.1 future
```

`robomimic --no-deps` is deliberate: its `egl_probe` dependency needs a C++/CMake
toolchain and is only used to pick a GPU for headless *rendering*. Training never
renders, so it is dead weight and a common install failure.

**Two pip errors here are expected and safe to ignore:**

- `robomimic 0.2.0 requires egl-probe>=1.0.1, which is not installed` — that is
  the `--no-deps` working as intended, see above.
- `diffusers`/`gradio` complaining about `huggingface-hub` — Colab preinstalls
  those and we do not use them. Do **not** pin `transformers<5` to "fix" it; that
  pin is what drags `huggingface-hub` back to 0.x in the first place, and
  transformers 5.x works fine here.

Run the smoke test in Cell 3b before continuing — a broken install surfaces three
cells later as an unrelated-looking error otherwise.

---

### Cell 3b — smoke test the install

```python
import torch, robomimic, transformers, huggingface_hub as hh
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("robomimic", robomimic.__version__, "| transformers", transformers.__version__,
      "| hf_hub", hh.__version__)
from huggingface_hub import snapshot_download          # Cell 6 needs this
import robomimic.utils.obs_utils, robomimic.utils.dataset   # the 2 LIBERO uses
import robosuite; print("robosuite", robosuite.__version__)
print("OK")
```

If `import robomimic.utils.dataset` raises `AttributeError: np.bool`, that is
expected — Cell 4 fixes it. Anything else, stop.

---

### Cell 4 — patch the two things that break on a modern stack

```python
# 1. robomimic 0.2.0 uses np.bool, removed in numpy 1.24. Patch the one line
#    rather than downgrading numpy and fighting Colab's preinstalled torch.
import robomimic, pathlib
p = pathlib.Path(robomimic.__file__).parent / "utils" / "dataset.py"
p.write_text(p.read_text().replace("astype(np.bool)", "astype(bool)"))

# 2. torch.load defaults to weights_only=True since 2.6 and LIBERO's init-state
#    files are pickled numpy. FOUR files need this, not two -- metric.py and
#    evaluate.py only bite during ROLLOUTS, not training, so a run can train for
#    hours and then fail on the first evaluation.
for f, old, new in [
    ("libero/libero/benchmark/__init__.py", "torch.load(init_states_path)",
     "torch.load(init_states_path, weights_only=False)"),
    ("libero/lifelong/metric.py", "torch.load(init_states_path)",
     "torch.load(init_states_path, weights_only=False)"),
    ("libero/lifelong/evaluate.py", "torch.load(init_states_path)",
     "torch.load(init_states_path, weights_only=False)"),
    ("libero/lifelong/utils.py", "torch.load(model_path, map_location=map_location)",
     "torch.load(model_path, map_location=map_location, weights_only=False)"),
]:
    q = pathlib.Path("/content/LIBERO") / f
    s = q.read_text()
    if new not in s and old in s:
        q.write_text(s.replace(old, new))
print("patched")
```

---

### Cell 5 — LIBERO config file (it calls `input()` on first import otherwise)

```python
import os, yaml
root = "/content/LIBERO/libero/libero"
os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
yaml.dump({"benchmark_root": root,
           "bddl_files": f"{root}/bddl_files",
           "init_states": f"{root}/init_files",
           "datasets": f"{root}/../datasets",
           "assets": f"{root}/assets"},
          open(os.path.expanduser("~/.libero/config.yaml"), "w"))
```

---

### Cell 6 — download ONE task's human demonstrations (~1.4 GB, not 13.7)

```python
from huggingface_hub import snapshot_download
TASK = 0
NAME = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5"
snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                  local_dir="/content/LIBERO/libero/datasets",
                  allow_patterns=f"libero_10/{NAME}")
!ls -lh /content/LIBERO/libero/datasets/libero_10/
```

**Train on these human demonstrations, never on Diffusion Policy's rollout
files.** Training a new policy on another policy's rollouts is distillation and
imports that policy's failure modes into BC-RNN.

Do **not** use `benchmark_scripts/download_libero_datasets.py` — `--datasets
libero_10` is not an accepted choice, and `--datasets libero_100
--use-huggingface` globs `libero_100/*`, which matches nothing in that repo and
downloads zero files silently.

---

### Cell 7 — write the trainer

This is the whole script. Nothing to upload — it is standalone.

```python
%%writefile /content/train_bc_rnn.py
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
```

---

### Cell 8 — train

```python
!cd /content && python train_bc_rnn.py \
    --task-id 0 --n-epochs 75 --batch-size 32 --num-workers 2 \
    --out "$OUT" --save-every 5 --patience 8 --min-delta 1e-3 --amp
```

`--amp` enables fp16 autocast, which on a T4's tensor cores is roughly 2x faster
than fp32. Drop it if losses go NaN.

**Plateau stopping:** training halts when the monitored loss fails to improve by
more than `--min-delta` for `--patience` consecutive epochs. It prefers a
**demo-level** validation holdout (10% of demos, never split within a demo —
sequences from one demo overlap heavily, so a sequence-level split leaks and
makes validation loss meaninglessly optimistic). If that split is unavailable it
falls back to smoothed *training* loss, which measures optimisation convergence
rather than generalisation; the log says which is in use.

A checkpoint is written every 5 epochs regardless, so the operating point can be
chosen afterwards rather than guessed now.

---

### Cell 9 — verify before disconnecting

```python
import json, os
h = json.load(open(f"{OUT}/history_task0.json"))
print(f"epochs run   : {h['epochs_run']}  early stop: {h['stopped_early']}")
print(f"best monitor : {h['best_monitor']:.4f}")
print(f"versions     : {h['versions']}")
print(f"checkpoints  : {len(h['sha256'])}")
for k, v in list(h['sha256'].items())[:5]:
    print(f"  {k}  {v[:16]}...")
```

---

## Why plateau ≠ done

Loss plateauing means optimisation converged. It does **not** mean the policy is
at a useful success rate. For arbitration the cheap arm needs to land in roughly
the **40–80%** band — good enough to be the sensible default, weak enough to
sometimes need escalation. That is measured by MuJoCo rollouts, not by loss.

This is exactly why every 5th epoch is kept. Bring the checkpoints back, evaluate
a few in the simulator locally, and pick the one in band. The BC-Transformer run
on this project hit 35% success at epoch 10 and was still climbing — loss curves
alone would not have told you that.

## Handoff (protocol §3.4)

`history_task0.json` already carries the sha256 of every checkpoint plus the
`torch` / `numpy` / `robomimic` / `torchvision` versions used. **Send it with the
weights, never the weights alone** — if a checkpoint is later suspected of having
changed, or two versions need distinguishing, the hash is the only reliable way
to tell. Once chosen, that file is frozen for every rollout collected against it.
