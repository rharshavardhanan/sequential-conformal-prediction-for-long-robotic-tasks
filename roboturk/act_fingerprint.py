"""Part 2.5 fingerprint + Part 4 sanity checks for the trained ACT.

Run after training. Writes act_fingerprint.json and prints every Part 4 gate as
PASS/FAIL. Part 4 forbids proceeding to the benchmark unless all of them pass.
"""
import argparse, hashlib, json, platform, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_act import GRIP, load_everything  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(HERE / "roboturk_act_best.pt"),
                help="best-val checkpoint, NOT the last-epoch one")
ap.add_argument("--out", default=str(HERE / "act_fingerprint.json"))
ap.add_argument("--partial", action="store_true",
                help="allow a still-training run (skips the epochs-complete gate)")
args = ap.parse_args()
ACT_CKPT = Path(args.ckpt)
BCRNN_CKPT = HERE / "roboturk_bcrnn_state_only.pt"

sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ok = lambda b: "PASS" if b else "FAIL"

act = torch.load(ACT_CKPT, map_location="cpu", weights_only=False)
bc = torch.load(BCRNN_CKPT, map_location="cpu", weights_only=False)
hist, cfg = act["history"], act["cfg"]
best_ep = min(hist, key=lambda r: r["val_l1"])["epoch"]
_, _, _, eps, st = load_everything(HERE / "roboturk_20hz.npz", BCRNN_CKPT)

checks, results = [], {}

# --- Part 4: ACT checkpoint sanity -------------------------------------------------
tl = [h["train_l1"] for h in hist]; vl = [h["val_l1"] for h in hist]
finite = all(np.isfinite(tl)) and all(np.isfinite(vl))
improved = vl[-1] < vl[0]
# "no divergence": final val within noise of the best, not climbing away from it
best = min(vl)
diverged = vl[-1] > best * 1.5
checks.append(("loss finite (no NaN/inf)", finite))
checks.append(("final val L1 < epoch-1 val L1", improved))
checks.append(("no divergence (final val <= 1.5x best)", not diverged))
checks.append(("checkpoint is the best-val epoch, not the last",
               act["epoch"] == best_ep))
if not args.partial:
    checks.append((f"training ran to completion ({cfg['epochs']} epochs)",
                   max(r["epoch"] for r in hist) >= cfg["epochs"]))
results["loss"] = dict(epochs=len(hist), first_val_l1=vl[0], best_val_l1=best,
                       final_val_l1=vl[-1], final_train_l1=tl[-1],
                       final_train_kl=hist[-1]["train_kl"], final_val_kl=hist[-1]["val_kl"])

# --- Part 4: input parity ----------------------------------------------------------
same_stats = all(np.array_equal(np.asarray(act[k], np.float32), np.asarray(bc[k], np.float32))
                 for k in ("s_mu", "s_sd", "a_mu", "a_sd"))
grip_raw = act["a_mu"][GRIP] == 0.0 and act["a_sd"][GRIP] == 1.0
split_ok = (len(eps["train"]), len(eps["val"]), len(eps["cal"]), len(eps["eval"])) == (316, 56, 79, 80)
dims_ok = cfg["sdim"] == 37 and cfg["adim"] == 8
checks.append(("normalisation stats byte-identical to BC-RNN", same_stats))
checks.append(("grasp bool unnormalised (a_mu[7]=0, a_sd[7]=1)", grip_raw))
checks.append(("split still 316/56/79/80 from seed 0", split_ok))
checks.append(("state-only 37-dim in, 8-dim out (no images)", dims_ok))
checks.append(("cal/eval episodes never seen in training",
               not (set(eps["cal"]) | set(eps["eval"])) & (set(eps["train"]) | set(eps["val"]))))

fp = {
    "checkpoint_sha256": sha256(ACT_CKPT),
    "bcrnn_checkpoint_sha256": sha256(BCRNN_CKPT),
    "libs": {"python": platform.python_version(), "torch": torch.__version__,
             "numpy": np.__version__, "platform": platform.platform(),
             "device": "Apple M5 Pro (MPS, 16-core GPU, 24 GB unified)"},
    "train_config": {
        "arch": "state-only ACT: CVAE posterior (4 layers) + transformer encoder (4) "
                "+ decoder (7), d=512, nhead=8, ff=3200, latent=32, chunk=%d" % cfg["chunk"],
        "params": cfg["params"], "bcrnn_params": sum(v.numel() for v in bc["model"].values()),
        **{k: cfg[k] for k in ("chunk", "stride", "batch", "lr", "wd", "kl_weight")},
        "checkpoint_epoch": act["epoch"], "epochs_run": len(hist), **results["loss"],
    },
    "decisions": {**json.load(open(HERE / "fingerprint.json"))["decisions"],
                  "inherited": "D1-D3 unchanged from the BC-RNN run; same npz, same "
                               "seed-0 split, same normalisation statistics."},
    "phase_model": "Part 2 Option A, K=2 (closed/open) via user_control[:,20]",
    "part4_checks": {name: bool(v) for name, v in checks},
}
json.dump(fp, open(args.out, "w"), indent=2)

print(f"ACT      {cfg['params']/1e6:.2f} M params   sha {fp['checkpoint_sha256'][:16]}...")
print(f"BC-RNN   {fp['train_config']['bcrnn_params']/1e6:.2f} M params   "
      f"ratio {cfg['params']/fp['train_config']['bcrnn_params']:.1f}x\n")
print(f"epochs {len(hist)}   val L1 {vl[0]:.4f} -> {vl[-1]:.4f} (best {best:.4f})   "
      f"train L1 {tl[0]:.4f} -> {tl[-1]:.4f}\n")
print("--- Part 4 gates ---")
for name, v in checks:
    print(f"  [{ok(v)}] {name}")
allp = all(v for _, v in checks)
print(f"\n{'ALL CHECKS PASS' if allp else 'BLOCKED'} -- wrote {args.out}")
sys.exit(0 if allp else 1)
