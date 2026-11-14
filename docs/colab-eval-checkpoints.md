# Evaluating BC-RNN checkpoints — which one lands in 40–80%?

Run in the **same Colab session** that trained (everything is already installed).
Locally would mean rebuilding the training stack that was removed from
`pyproject.toml` in September.

Best *validation loss* was epoch 14. That is not the same as best *success rate*
— success depends on recovering from states the demonstrations never visited,
which likelihood does not measure. Evaluate a spread and pick by success.

BC-RNN inference is a single recurrent step, so this is far cheaper than the
diffusion policy's denoising loop: expect **~10 min per checkpoint**, not 35.

---

### Cell E1 — headless rendering

```python
import os
os.environ["MUJOCO_GL"] = "egl"          # required: Colab has no display
os.environ["PYOPENGL_PLATFORM"] = "egl"
!pip install -q imageio
```

Rollouts render; training did not. Without `MUJOCO_GL=egl` the env fails at
construction with an opaque GL error.

---

### Cell E2 — the evaluator

```python
%%writefile /content/eval_ckpt.py
"""Success rate of a BC-RNN checkpoint, via LIBERO's own evaluator.

Selection metric is SUCCESS, not validation loss. For the arbitration cheap arm
the target is 40-80%: above it there are too few failures to escalate on, below
it the policy is not a sensible default.
"""
import argparse, hashlib, json, os, time
import numpy as np, torch


def sha256_file(p, _c=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(_c), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--num-procs", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys; sys.path.insert(0, "/content")
    from train_bc_rnn import build_cfg, load_one_task, pick_device
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.metric import evaluate_one_task_success
    from libero.lifelong.utils import (confidence_interval, control_seed,
                                       safe_device, torch_load_model)

    device = pick_device("auto")
    cfg = build_cfg([
        "benchmark_name=LIBERO_10", "policy=bc_rnn_policy", "lifelong=single_task",
        f"device={device}", f"eval.n_eval={args.n_eval}",
        f"eval.num_procs={args.num_procs}", f"eval.max_steps={args.max_steps}",
        "eval.use_mp=true", "use_wandb=false",
    ])
    control_seed(cfg.seed)
    benchmark, _, shape_meta = load_one_task(cfg, args.task_id)
    cfg.shape_meta = shape_meta
    cfg.experiment_dir = "/tmp"

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    algo.policy.load_state_dict(sd["state_dict"])
    algo.eval()

    task = benchmark.get_task(args.task_id)
    digest = sha256_file(args.checkpoint)
    print(f"[eval] {os.path.basename(args.checkpoint)}  epoch={sd.get('epoch')}  "
          f"val={sd.get('monitor'):.4f}")
    print(f"[eval] sha256 {digest[:16]}...  {args.n_eval} rollouts")

    t0 = time.perf_counter()
    sr = evaluate_one_task_success(cfg=cfg, algo=algo, task=task,
                                   task_emb=benchmark.get_task_emb(args.task_id),
                                   task_id=args.task_id)
    wall = time.perf_counter() - t0
    n = int(round(sr * args.n_eval)); ci = confidence_interval(sr, args.n_eval)
    band = 0.40 <= sr <= 0.80
    print(f"  success {sr:.2f} ({n}/{args.n_eval}) +/-{ci:.2f}   "
          f"{'IN BAND' if band else ('TOO HIGH' if sr > 0.80 else 'TOO LOW')}   "
          f"{wall/60:.1f} min")

    res = {"checkpoint": args.checkpoint, "epoch": sd.get("epoch"),
           "val_loss": sd.get("monitor"), "sha256": digest,
           "n_eval": args.n_eval, "n_success": n, "success_rate": float(sr),
           "ci95": float(ci), "in_band_40_80": bool(band), "wall_sec": wall}
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
```

---

### Cell E3 — sweep the checkpoints

```python
import os, glob, json
CKPTS = sorted(glob.glob(f"{OUT}/bcrnn_task0_ep*.pth"))
print("available:", [os.path.basename(c) for c in CKPTS])

# Spread across the run, not just best-val (ep 14). Success may peak later.
TARGETS = [5, 10, 14, 20, 22]
for ep in TARGETS:
    c = f"{OUT}/bcrnn_task0_ep{ep:03d}.pth"
    if not os.path.exists(c):
        print(f"  ep{ep}: not saved, skipping"); continue
    !cd /content && python eval_ckpt.py --checkpoint "{c}" --task-id 0 \
        --n-eval 20 --num-procs 5 --out "{OUT}/eval_ep{ep:03d}.json"
```

---

### Cell E4 — the summary that decides it

```python
import glob, json
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/eval_ep*.json"))]
print(f"{'epoch':>6} {'val loss':>10} {'success':>18} {'verdict':>10}")
for r in sorted(rows, key=lambda x: x["epoch"]):
    v = "IN BAND" if r["in_band_40_80"] else ("too high" if r["success_rate"] > .8 else "too low")
    print(f"{r['epoch']:>6} {r['val_loss']:>10.3f} "
          f"{r['success_rate']:.2f} ({r['n_success']:2d}/20) +/-{r['ci95']:.2f}  {v:>10}")
```

---

## Reading the result

**One lands in 40–80%.** Done. Freeze that checkpoint, record its sha256, and it
becomes the cheap arm for every arbitration rollout. No retraining.

**All below 40%, rising with epoch.** Success was still climbing when the val-loss
criterion stopped us at 22. Re-run with `--patience 100 --n-epochs 50`
(~2.8 h) to match LIBERO's own protocol, which selects on success rather than
likelihood.

**All below 40%, flat or falling.** BC-RNN cannot do this task at a useful level.
That is a real finding about the cheap arm's viability, not a training failure —
LIBERO-Long is the hardest of the four suites, and LIBERO's published BC-RNN
numbers are for `libero_90`, so there is no reference point saying it should work
here. Report it and reconsider the cheap arm.

**Something is above 80%.** Unlikely given the diffusion policy manages 68.7%,
but if so that checkpoint is useless for arbitration — there would be almost
nothing to escalate on. Take an earlier one.

## After a checkpoint is chosen

It is frozen. Every arbitration rollout uses that exact file — no further
training, no reloading with a different seed — and the sha256 goes into every
rollout, exactly as the diffusion-policy collection did. Calibration rollouts for
arbitration must be **BC-RNN only**; the existing task 0/1/8 files are diffusion
policy trajectories and are a different distribution.
