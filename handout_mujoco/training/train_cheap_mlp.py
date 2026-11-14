"""Distil the composed expert into a small MLP by behavioural cloning, then evaluate it
closed-loop.

Architecture: 26 -> 64 -> 64 -> 3, ReLU hidden, tanh output (actions live in [-1,1]^3).
Parameter count is reported from the module, not asserted. The design is deliberately NOT
copied from the unrecovered 5,891-parameter PyBullet cheap policy -- that number is from a
different task in a different simulator and reproducing it would be cargo-culting.

Observations are standardised by the TRAIN-split mean/std, stored in the model's buffers,
so calibration and the live demo apply exactly the transform the policy was distilled
under.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python training/train_cheap_mlp.py
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))

from cheap_policy import CheapMLP, CheapMLPGaussian, save
from composed import PUSH, REACH, ComposedEnv, rollout

DEMOS = ROOT / "results" / "mujoco_train" / "expert_demos.npz"
OUT = ROOT / "results" / "mujoco_train" / "cheap_policy"
EVAL_BASE, N_EVAL = 400_000, 200


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate_cheap(model, n=N_EVAL, seed_base=EVAL_BASE):
    env = ComposedEnv()
    rows, t_inf = [], []
    for i in range(n):
        t = []

        def act(e):
            t0 = time.perf_counter()
            a = model.act(e.obs26())
            t.append(time.perf_counter() - t0)
            return a, "cheap"

        info, steps = rollout(env, act, seed_base + i)
        t_inf += t
        rows.append({"seed": seed_base + i, "success": bool(info["is_success"]),
                     "reached": bool(info["reached"]), "steps": steps,
                     "cube_fell": bool(info["cube_fell"]), "d0": info["d0"]})
    return rows, float(np.median(t_inf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--mask-dims", type=int, nargs="*", default=[],
                    help="obs columns to zero (train AND inference) -- used to build "
                         "a trigger that provably cannot see the regime feature")
    ap.add_argument("--gaussian", action="store_true",
                    help="heteroscedastic head, trained with Gaussian NLL (candidate c)")
    a = ap.parse_args()
    out_dir = ROOT / "results" / "mujoco_train"
    torch.manual_seed(a.torch_seed)

    z = np.load(DEMOS)
    obs, act, phase = z["obs"].copy(), z["act"], z["phase"]
    if a.mask_dims:
        obs[:, a.mask_dims] = 0.0
        print(f"  MASKED obs columns {a.mask_dims} (zeroed at train time and at inference)")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    n_val = int(len(obs) * a.val_frac)
    val_i, tr_i = perm[:n_val], perm[n_val:]

    mean = obs[tr_i].mean(0)
    std = obs[tr_i].std(0)
    std[std < 1e-6] = 1.0

    model = (CheapMLPGaussian if a.gaussian else CheapMLP)(tuple(a.hidden))
    model.obs_mean.copy_(torch.as_tensor(mean))
    model.obs_std.copy_(torch.as_tensor(std))
    n_p = model.n_params()
    print(f"cheap MLP 26 -> {' -> '.join(map(str, a.hidden))} -> 3   "
          f"ACTOR parameters {n_p:,}")
    print(f"  demos {len(obs):,} transitions  (train {len(tr_i):,} / val {len(val_i):,})"
          f"  reach {(phase == REACH).mean():.1%} / push {(phase == PUSH).mean():.1%}")

    Xtr = torch.as_tensor(obs[tr_i]); Ytr = torch.as_tensor(act[tr_i])
    Xva = torch.as_tensor(obs[val_i]); Yva = torch.as_tensor(act[val_i])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    mse = nn.MSELoss()

    def lossf(pred, target):
        """MSE for the deterministic head; Gaussian NLL for the heteroscedastic one."""
        if not a.gaussian:
            return mse(pred, target)
        mu, log_std = pred
        return (log_std + 0.5 * ((target - mu) / log_std.exp()) ** 2).mean()
    hist = []
    for ep in range(a.epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        tot = 0.0
        for s in range(0, len(idx), a.batch):
            b = idx[s : s + a.batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva), Yva).item()
            vmse = mse(model(Xva)[0] if a.gaussian else model(Xva), Yva).item()
        hist.append({"epoch": ep + 1, "train_loss": tot / len(Xtr), "val_loss": vl,
                     "train_mse": tot / len(Xtr), "val_mse": vmse})
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"  epoch {ep + 1:3d}  train {hist[-1]['train_mse']:.5f}  val {vl:.5f}")

    ckpt = out_dir / f"cheap_policy{a.tag}.pt"
    save(model, ckpt, meta={"hidden": a.hidden, "n_params_actor": n_p,
                            "gaussian": bool(a.gaussian), "torch_seed": a.torch_seed,
                            "mask_dims": list(a.mask_dims),
                            "demos": str(DEMOS.name), "val_mse": hist[-1]["val_mse"]})

    if a.mask_dims:
        print(f"\n  masked trigger saved to {ckpt.name}; no closed-loop eval "
              f"(it is a monitor, not a controller)")
        return {"n_params_actor": n_p, "mask_dims": a.mask_dims}
    print(f"\n  closed-loop eval on seeds {EVAL_BASE:,}-{EVAL_BASE + N_EVAL - 1:,} ...")
    rows, inf_s = evaluate_cheap(model)
    ok = np.array([r["success"] for r in rows])
    reached = np.array([r["reached"] for r in rows])
    push_given_reach = (float(ok[reached].mean()) if reached.any() else 0.0)
    res = {
        "architecture": f"26 -> {' -> '.join(map(str, a.hidden))} -> 3, ReLU, tanh output",
        "n_params_actor": n_p,
        "final_train_mse": hist[-1]["train_mse"],
        "final_val_mse": hist[-1]["val_mse"],
        "n_episodes": len(rows),
        "composed_success": float(ok.mean()),
        "composed_ci95_wilson": list(wilson(int(ok.sum()), len(ok))),
        "reach_success": float(reached.mean()),
        "push_success_given_reach": push_given_reach,
        "mean_ep_len": float(np.mean([r["steps"] for r in rows])),
        "frac_cube_fell": float(np.mean([r["cube_fell"] for r in rows])),
        "inference_ms_median_cpu": inf_s * 1000.0,
        "eval_seeds": [EVAL_BASE, EVAL_BASE + N_EVAL - 1],
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "training_history": hist,
    }
    (out_dir / f"cheap_policy{a.tag}.json").write_text(json.dumps(res, indent=2))
    lo, hi = res["composed_ci95_wilson"]
    print("\n  GATE 5 band [0.50, 0.75]")
    print(f"  composed success {res['composed_success']:.3f}  CI95 [{lo:.3f}, {hi:.3f}]")
    print(f"  reach {res['reach_success']:.3f}   push|reach {push_given_reach:.3f}   "
          f"ep_len {res['mean_ep_len']:.1f}   cube fell {res['frac_cube_fell']:.3f}")
    print(f"  inference {res['inference_ms_median_cpu']:.4f} ms (CPU, batch 1)")
    v = res["composed_success"]
    print(f"  -> {'IN BAND' if 0.50 <= v <= 0.75 else ('TOO STRONG' if v > 0.75 else 'TOO WEAK')}")
    return res


if __name__ == "__main__":
    main()
