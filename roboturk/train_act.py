"""State-only ACT — Part 2 of RoboTurk_ACT_and_Full_Benchmark.md.

The expensive arm of the arbitration benchmark. Same 37-dim state input, same
8-dim action target and the same seed-0 split as the BC-RNN, so any difference
between the two is architecture and capacity alone. No images, no MuJoCo, no
robosuite, no LIBERO.

Normalisation statistics are reused verbatim from the BC-RNN checkpoint rather
than recomputed -- Part 4's input-parity check requires both policies to see
byte-identical (state, action) pairs, and recomputing would drift in the last
few decimals.

    .venv-lerobot/bin/python roboturk/train_act.py --epochs 50
"""
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
GRIP = 7                                   # action col 7 = grasp bool, kept in {0,1}


# --------------------------------------------------------------------------- model
class ACT(nn.Module):
    """Action Chunking Transformer, state-only: DETR-style CVAE without the backbone.

    A style latent z is inferred from (state, action chunk) during training and
    zeroed at inference; the decoder turns `chunk` learned queries into the
    action chunk in one forward pass.
    """

    def __init__(self, sdim=37, adim=8, chunk=16, d=256, nhead=8, ff=1024,
                 n_enc=4, n_dec=4, n_cvae=3, zdim=32, dropout=0.2, history=20):
        super().__init__()
        self.chunk, self.zdim, self.history = chunk, zdim, history
        enc_layer = lambda: nn.TransformerEncoderLayer(
            d, nhead, ff, dropout=dropout, batch_first=True, norm_first=True)

        # CVAE posterior encoder: [CLS, state, actions] -> mu, logvar
        self.cvae_cls = nn.Parameter(torch.zeros(1, 1, d))
        self.cvae_s, self.cvae_a = nn.Linear(sdim, d), nn.Linear(adim, d)
        self.cvae_pos = nn.Parameter(torch.zeros(1, chunk + 2, d))
        self.cvae = nn.TransformerEncoder(enc_layer(), n_cvae)
        self.to_z = nn.Linear(d, 2 * zdim)

        # main encoder over [z, state]; decoder over learned chunk queries
        self.z_proj, self.s_proj = nn.Linear(zdim, d), nn.Linear(sdim, d)
        self.enc_pos = nn.Parameter(torch.zeros(1, history + 1, d))
        self.enc = nn.TransformerEncoder(enc_layer(), n_enc)
        self.q = nn.Parameter(torch.zeros(1, chunk, d))
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nhead, ff, dropout=dropout,
                                       batch_first=True, norm_first=True), n_dec)
        self.head = nn.Linear(d, adim)

        for p in (self.cvae_cls, self.cvae_pos, self.enc_pos, self.q):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, s, a=None):
        """s (B, history, sdim); a (B, chunk, adim) during training, None at inference.

        The state history is what lets a single pose be disambiguated by the motion
        that led to it -- without it the model must memorise which episode it is in.
        """
        B = s.shape[0]
        cur = s[:, -1]
        if a is None:                                    # inference: z = 0 (mean style)
            mu = logvar = None
            z = s.new_zeros(B, self.zdim)
        else:
            tok = torch.cat([self.cvae_cls.expand(B, -1, -1),
                             self.cvae_s(cur).unsqueeze(1), self.cvae_a(a)], 1)
            mu, logvar = self.to_z(self.cvae(tok + self.cvae_pos)[:, 0]).chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        tok = torch.cat([self.z_proj(z).unsqueeze(1), self.s_proj(s)], 1)
        mem = self.enc(tok + self.enc_pos)
        return self.head(self.dec(self.q.expand(B, -1, -1), mem)), mu, logvar


# ---------------------------------------------------------------------------- data
def load_everything(npz, bcrnn_ckpt):
    """Rebuild the frozen split and reuse BC-RNN's normalisation, or die trying."""
    z = np.load(npz, allow_pickle=True)
    S, A, E, ep_T = z["S"], z["A"], z["E"], z["ep_T"]
    n_ep = len(ep_T)

    # verbatim from RoboTurk_BCRNN_Training.ipynb cell 26 -- do not vary
    rng = np.random.default_rng(0)
    order = rng.permutation(n_ep)
    n_hold = int(round(0.30 * n_ep))
    calib_eval_ep, trainval_ep = np.sort(order[:n_hold]), np.sort(order[n_hold:])
    k = int(round(0.85 * len(trainval_ep)))
    train_ep, val_ep = np.sort(trainval_ep[:k]), np.sort(trainval_ep[k:])
    n_cal = len(calib_eval_ep) // 2
    cal_ep, eval_ep = calib_eval_ep[:n_cal], calib_eval_ep[n_cal:]
    assert (len(train_ep), len(val_ep), len(cal_ep), len(eval_ep)) == (316, 56, 79, 80), \
        "split drifted from the frozen 316/56/79/80"

    starts = np.zeros(n_ep + 1, np.int64)
    np.cumsum(ep_T, out=starts[1:])
    assert starts[-1] == len(S) and np.all(E[starts[:-1]] == np.arange(n_ep)), \
        "ep_T does not agree with E; the cache is not what BC-RNN trained on"

    ck = torch.load(bcrnn_ckpt, map_location="cpu", weights_only=False)
    st = {x: np.asarray(ck[x], np.float32) for x in ("s_mu", "s_sd", "a_mu", "a_sd")}
    assert st["a_mu"][GRIP] == 0.0 and st["a_sd"][GRIP] == 1.0, \
        "grasp bool must stay unnormalised for parity with BC-RNN"
    return S, A, starts, dict(train=train_ep, val=val_ep, cal=cal_ep, eval=eval_ep), st


def chunk_index(eps, starts, ep_T, stride):
    """Global start index of every chunk, plus its episode's inclusive start and
    exclusive end -- both needed to clamp the history and the chunk at the edges."""
    st, bg, en = [], [], []
    for i in eps:
        a, b = starts[i], starts[i + 1]
        s = np.arange(a, b, stride, dtype=np.int64)
        st.append(s); bg.append(np.full(len(s), a, np.int64))
        en.append(np.full(len(s), b, np.int64))
    return np.concatenate(st), np.concatenate(bg), np.concatenate(en)


def make_batch(Sn, An, start, beg, end, chunk, hist, dev):
    """Gather (state history, action chunk, pad mask).

    History runs [t-hist+1 .. t] clamped to the episode start, so early timesteps
    repeat the first frame rather than bleeding in the previous episode. Chunks
    past the episode end are padded and masked out of the loss.
    """
    h = start[:, None] + np.arange(-hist + 1, 1)[None, :]
    np.maximum(h, beg[:, None], out=h)
    idx = start[:, None] + np.arange(chunk)[None, :]
    pad = idx >= end[:, None]
    np.minimum(idx, end[:, None] - 1, out=idx)
    return (torch.from_numpy(Sn[h]).to(dev),
            torch.from_numpy(An[idx]).to(dev),
            torch.from_numpy(pad).to(dev))


# ------------------------------------------------------------------------- training
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(HERE / "roboturk_20hz.npz"))
    p.add_argument("--bcrnn", default=str(HERE / "roboturk_bcrnn_state_only.pt"))
    p.add_argument("--out", default=str(HERE / "roboturk_act.pt"))
    p.add_argument("--history", default=str(HERE / "act_loss_history.json"))
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--obs-history", type=int, default=20, help="state frames fed to the encoder")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=4)
    p.add_argument("--cvae-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--cosine", action="store_true", default=True)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--kl", type=float, default=10.0, help="ACT's kl_weight")
    p.add_argument("--max-batches", type=int, default=0, help="smoke-test escape hatch")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(0)
    S, A, starts, eps, st = load_everything(args.npz, args.bcrnn)
    ep_T = np.diff(starts)

    Sn = ((S - st["s_mu"]) / st["s_sd"]).astype(np.float32)
    An = ((A - st["a_mu"]) / st["a_sd"]).astype(np.float32)

    tr_s, tr_b, tr_e = chunk_index(eps["train"], starts, ep_T, args.stride)
    va_s, va_b, va_e = chunk_index(eps["val"], starts, ep_T, args.stride)
    print(f"device {dev} | chunk {args.chunk} history {args.obs_history} stride {args.stride} batch {args.batch} d={args.dim} drop {args.dropout} wd {args.wd}")
    print(f"train chunks {len(tr_s):,}  val chunks {len(va_s):,}  "
          f"({int(np.ceil(len(tr_s)/args.batch)):,} batches/epoch)")

    model = ACT(S.shape[1], A.shape[1], chunk=args.chunk, d=args.dim, ff=args.ff,
                n_enc=args.enc_layers, n_dec=args.dec_layers, n_cvae=args.cvae_layers,
                dropout=args.dropout, history=args.obs_history).to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01) if args.cosine else None
    start_ep, hist = 1, []
    if args.resume and Path(args.out).exists():
        ck = torch.load(args.out, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched and "sched" in ck: sched.load_state_dict(ck["sched"])
        start_ep, hist = ck["epoch"] + 1, ck.get("history", [])
        print(f"resumed from epoch {ck['epoch']}")
    print(f"ACT params {n_par/1e6:.2f} M")

    def losses(s, a, pad):
        ah, mu, lv = model(s, a)
        l1 = ((ah - a).abs().mean(-1) * ~pad).sum() / (~pad).sum()
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).sum(-1).mean()
        return l1, kl

    rng = np.random.default_rng(0)
    for ep in range(start_ep, args.epochs + 1):
        model.train(); t0 = time.perf_counter()
        perm = rng.permutation(len(tr_s))
        if args.max_batches:
            perm = perm[: args.max_batches * args.batch]
        tl = tk = nb = 0
        for b in range(0, len(perm), args.batch):
            j = perm[b : b + args.batch]
            s, a, pad = make_batch(Sn, An, tr_s[j], tr_b[j], tr_e[j], args.chunk, args.obs_history, dev)
            l1, kl = losses(s, a, pad)
            loss = l1 + args.kl * kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += l1.item(); tk += kl.item(); nb += 1
            if args.max_batches and nb == 1:
                print(f"  first batch: L1 {l1.item():.4f}  KL {kl.item():.4f}")

        model.eval(); vl = vk = vn = 0
        with torch.no_grad():
            vp = np.arange(len(va_s))
            if args.max_batches:
                vp = vp[: args.max_batches * args.batch]
            for b in range(0, len(vp), args.batch):
                j = vp[b : b + args.batch]
                s, a, pad = make_batch(Sn, An, va_s[j], va_b[j], va_e[j], args.chunk, args.obs_history, dev)
                l1, kl = losses(s, a, pad)
                vl += l1.item(); vk += kl.item(); vn += 1

        if sched: sched.step()
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"], train_l1=tl / nb, train_kl=tk / nb,
                   val_l1=vl / vn, val_kl=vk / vn, secs=time.perf_counter() - t0)
        hist.append(row)
        if not math.isfinite(row["train_l1"]) or not math.isfinite(row["val_l1"]):
            print("NaN/inf loss -- stopping"); break
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "sched": sched.state_dict() if sched else None,
                    "history": hist, "s_mu": st["s_mu"], "s_sd": st["s_sd"],
                    "a_mu": st["a_mu"], "a_sd": st["a_sd"],
                    "cfg": dict(chunk=args.chunk, stride=args.stride, batch=args.batch,
                                lr=args.lr, wd=args.wd, kl_weight=args.kl,
                                epochs=args.epochs, params=n_par, sdim=int(S.shape[1]),
                                adim=int(A.shape[1]), history=args.obs_history, dim=args.dim,
                                ff=args.ff, enc_layers=args.enc_layers,
                                dec_layers=args.dec_layers, cvae_layers=args.cvae_layers,
                                dropout=args.dropout, cosine=bool(args.cosine))}, args.out)
        json.dump(hist, open(args.history, "w"), indent=2)
        print(f"epoch {ep:3d}  train_l1 {row['train_l1']:.4f}  val_l1 {row['val_l1']:.4f}  "
              f"train_kl {row['train_kl']:.4f}  val_kl {row['val_kl']:.4f}  "
              f"{row['secs']/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
