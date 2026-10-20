"""Batch-1 actor inference latency, CPU vs MPS. Decides the device for every Phase-9 arm.

Phase 9 compares compute across expert-only / cheap-only / arbitrated arms. That
comparison is only meaningful if every arm is timed on the SAME device at the SAME batch
size, because the device ranking INVERTS with network size on this machine: measured
here, CPU beats MPS at batch 256 (8.74 vs 12.57 ms per gradient step) while MPS wins at
2048. The cheap MLP will be ~8k parameters and the push actor is ~1.5M. Benchmarking the
expert on MPS and the cheap policy on CPU would manufacture a compute ratio out of a
device switch.

`torch.mps.synchronize()` brackets the timed region. MPS dispatch is asynchronous: without
it you time enqueue, not compute, and this has produced wrong numbers on this machine
before.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python analysis/inference_latency.py
"""

import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))

from envs import PushOnly
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

CKPT = ROOT / "results" / "mujoco_train" / "push_expert" / "seed0" / "best_model.zip"
OUT = ROOT / "results" / "mujoco_train" / "inference_latency.json"
WARMUP, TIMED = 100, 1000


def bench(device):
    """Time batch-1 deterministic actor calls, each call timed individually."""
    model = TQC.load(CKPT, env=DummyVecEnv([PushOnly]), device=device)
    env = PushOnly()
    obs, _ = env.reset(seed=400_000)
    is_mps = device == "mps"

    for _ in range(WARMUP):
        model.predict(obs, deterministic=True)
    if is_mps:
        torch.mps.synchronize()

    t = np.empty(TIMED)
    for i in range(TIMED):
        t0 = time.perf_counter()
        model.predict(obs, deterministic=True)
        if is_mps:
            # without this we time enqueue, not compute
            torch.mps.synchronize()
        t[i] = time.perf_counter() - t0
    env.close()

    ms = t * 1000.0
    return {
        "device": device,
        "n_warmup": WARMUP,
        "n_timed": TIMED,
        "mean_ms": float(ms.mean()),
        "sd_ms": float(ms.std()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(ms.min()),
        "actor_params": int(sum(p.numel() for p in model.actor.parameters())),
        "total_params": int(sum(p.numel() for p in model.policy.parameters())),
    }


def main():
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint: {CKPT}")
    res = {d: bench(d) for d in ("cpu", "mps") if d == "cpu" or torch.backends.mps.is_available()}
    faster = min(res, key=lambda d: res[d]["p50_ms"])
    ratio = res[max(res, key=lambda d: res[d]["p50_ms"])]["p50_ms"] / res[faster]["p50_ms"]
    out = {
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "batch_size": 1,
        "mps_synchronize_used": True,
        "torch": torch.__version__,
        "results": res,
        "chosen_device": faster,
        "speedup_over_slower": float(ratio),
        "decision": (
            f"All Phase-9 arms MUST be timed batch-1 on '{faster}'. Mixing devices across "
            "arms would turn a device switch into an apparent compute saving."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Batch-1 actor inference, {CKPT.name}  (actor params "
          f"{res[faster]['actor_params']:,}, total saved policy {res[faster]['total_params']:,})")
    print(f"{'device':<8} {'mean':>9} {'sd':>9} {'p50':>9} {'p95':>9} {'min':>9}")
    for d, r in res.items():
        print(f"{d:<8} {r['mean_ms']:>8.3f}m {r['sd_ms']:>8.3f}m {r['p50_ms']:>8.3f}m "
              f"{r['p95_ms']:>8.3f}m {r['min_ms']:>8.3f}m")
    print(f"\nCHOSEN DEVICE: {faster}  ({ratio:.2f}x faster at p50)")
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
