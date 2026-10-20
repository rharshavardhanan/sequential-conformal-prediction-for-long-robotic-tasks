"""Layer 3, step 2: train the cheap and expensive arms on Route-B observations.

Both consume the 32-dim observation that INCLUDES the cube positions, so both
can react when a block moves. Both emit the same 8-dim action (7 absolute joint
targets + grasp), which is the RoboTurk action shape -- so the arbitration
machinery is unchanged.

  BC-RNN  (cheap)      MLP -> LSTM -> GMM head.  Same family as the RoboTurk
                       cheap arm, so the comparison stays like-for-like.
  ACT     (expensive)  the CVAE transformer from roboturk/train_act.py, reused
                       verbatim rather than reimplemented.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "roboturk"))
GRIP = 7


class BCRNN(nn.Module):
    """MLP + LSTM + 5-mode GMM. Layer names match the RoboTurk cheap arm."""

    def __init__(self, sdim, adim, hidden=256, layers=2, modes=5):
        super().__init__()
        self.modes, self.adim = modes, adim
        self.enc = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU())
        self.rnn = nn.LSTM(256, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, modes * (2 * adim + 1))

    def forward(self, s, hx=None):
        B, T, _ = s.shape
        z = self.enc(s.reshape(B * T, -1)).reshape(B, T, -1)
        o, hx = self.rnn(z, hx)
        p = self.head(o).reshape(B, T, self.modes, 2 * self.adim + 1)
        return (p[..., :self.adim], p[..., self.adim:2 * self.adim].clamp(-5, 2),
                p[..., -1], hx)


def gmm_nll(mu, logstd, logit, a):
    """Negative log-likelihood of a diagonal-Gaussian mixture."""
    a = a.unsqueeze(-2)
    ll = (-0.5 * (((a - mu) / logstd.exp()) ** 2) - logstd).sum(-1)
    ll = ll + torch.log_softmax(logit, -1)
    return -torch.logsumexp(ll, -1).mean()


def gmm_mean(mu, logit):
    return (torch.softmax(logit, -1).unsqueeze(-1) * mu).sum(-2)


def load(path):
    O, A, ep = [], [], []
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f if k.startswith("demo_") and f[k].attrs["success"])
        for i, k in enumerate(keys):
            O.append(f[k]["obs"][:]); A.append(f[k]["act"][:])
            ep.append(np.full(len(O[-1]), i, np.int32))
        meta = dict(f.attrs)
    return (np.concatenate(O), np.concatenate(A), np.concatenate(ep),
            len(keys), meta)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/tower/demos.hdf5")
    ap.add_argument("--which", choices=["bcrnn", "act", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--outdir", default="data/tower")
    args = ap.parse_args()

    dev = torch.device(args.device)
    O, A, EP, n_ep, meta = load(args.demos)
    sdim, adim = O.shape[1], A.shape[1]
    starts = np.flatnonzero(np.r_[True, EP[1:] != EP[:-1]]).tolist() + [len(EP)]

    n_val = max(4, n_ep // 8)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n_ep)
    val_ep, tr_ep = set(perm[:n_val].tolist()), perm[n_val:].tolist()

    s_mu, s_sd = O.mean(0), O.std(0) + 1e-6
    a_mu, a_sd = A.mean(0), A.std(0) + 1e-6
    On = ((O - s_mu) / s_sd).astype(np.float32)
    An = ((A - a_mu) / a_sd).astype(np.float32)
    print(f"{n_ep} successful demos  {len(O)} steps  obs {sdim}  act {adim}")
    print(f"train {len(tr_ep)} episodes / val {len(val_ep)}")

    def windows(eps, w):
        out = []
        for i in eps:
            lo, hi = starts[i], starts[i + 1]
            for s in range(lo, max(lo + 1, hi - w), max(1, w // 3)):
                if s + w <= hi:
                    out.append(s)
        return np.array(out)

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stats = dict(s_mu=s_mu, s_sd=s_sd, a_mu=a_mu, a_sd=a_sd)
    common = dict(sdim=int(sdim), adim=int(adim), demos=str(args.demos),
                  demos_sha256=sha_file(args.demos), n_demos=int(n_ep),
                  obs_layout=str(meta.get("obs_layout")),
                  act_layout=str(meta.get("act_layout")))

    # ------------------------------------------------------------- BC-RNN --
    if args.which in ("bcrnn", "both"):
        m = BCRNN(sdim, adim).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        tr, va = windows(tr_ep, args.window), windows(sorted(val_ep), args.window)
        best, hist = float("inf"), []
        for ep in range(args.epochs):
            m.train(); rng.shuffle(tr); tot = n = 0
            for b in range(0, len(tr), args.batch):
                idx = tr[b:b + args.batch]
                s = torch.tensor(np.stack([On[i:i + args.window] for i in idx])).to(dev)
                a = torch.tensor(np.stack([An[i:i + args.window] for i in idx])).to(dev)
                mu, ls, lg, _ = m(s)
                loss = gmm_nll(mu, ls, lg, a)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); tot += float(loss); n += 1
            m.eval(); v = vn = 0
            with torch.no_grad():
                for b in range(0, len(va), args.batch):
                    idx = va[b:b + args.batch]
                    s = torch.tensor(np.stack([On[i:i + args.window] for i in idx])).to(dev)
                    a = torch.tensor(np.stack([An[i:i + args.window] for i in idx])).to(dev)
                    mu, ls, lg, _ = m(s)
                    v += float(gmm_nll(mu, ls, lg, a)); vn += 1
            hist.append(dict(epoch=ep, train=tot / max(n, 1), val=v / max(vn, 1)))
            if hist[-1]["val"] < best:
                best = hist[-1]["val"]
                torch.save(dict(model=m.state_dict(), epoch=ep, val=best,
                                **stats, cfg=dict(**common, arch="BCRNN")),
                           outdir / "tower_bcrnn.pt")
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"  bcrnn ep{ep:3d}  train {hist[-1]['train']:8.3f}  "
                      f"val {hist[-1]['val']:8.3f}{'  *' if hist[-1]['val']==best else ''}",
                      flush=True)
        json.dump(hist, open(outdir / "tower_bcrnn_history.json", "w"), indent=1)
        print(f"  -> {outdir/'tower_bcrnn.pt'}  best val {best:.3f}")

    # ---------------------------------------------------------------- ACT --
    if args.which in ("act", "both"):
        from train_act import ACT
        CHUNK, HIST = 16, 10
        m = ACT(sdim, adim, chunk=CHUNK, d=256, ff=1024, n_enc=4, n_dec=4,
                n_cvae=4, dropout=0.1, history=HIST).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
        need = HIST + CHUNK
        tr, va = windows(tr_ep, need), windows(sorted(val_ep), need)
        best, hist = float("inf"), []
        for ep in range(args.epochs):
            m.train(); rng.shuffle(tr); tot = n = 0
            for b in range(0, len(tr), args.batch):
                idx = tr[b:b + args.batch]
                s = torch.tensor(np.stack([On[i:i + HIST] for i in idx])).to(dev)
                a = torch.tensor(np.stack([An[i + HIST:i + HIST + CHUNK]
                                           for i in idx])).to(dev)
                pred, mu, logvar = m(s, a)
                l1 = (pred - a).abs().mean()
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                loss = l1 + 10.0 * kl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); tot += float(l1); n += 1
            m.eval(); v = vn = 0
            with torch.no_grad():
                for b in range(0, len(va), args.batch):
                    idx = va[b:b + args.batch]
                    s = torch.tensor(np.stack([On[i:i + HIST] for i in idx])).to(dev)
                    a = torch.tensor(np.stack([An[i + HIST:i + HIST + CHUNK]
                                               for i in idx])).to(dev)
                    v += float((m(s)[0] - a).abs().mean()); vn += 1
            hist.append(dict(epoch=ep, train=tot / max(n, 1), val=v / max(vn, 1)))
            if hist[-1]["val"] < best:
                best = hist[-1]["val"]
                torch.save(dict(model=m.state_dict(), epoch=ep, val=best, **stats,
                                cfg=dict(**common, arch="ACT", chunk=CHUNK,
                                         history=HIST, dim=256, ff=1024,
                                         enc_layers=4, dec_layers=4, cvae_layers=4,
                                         dropout=0.1,
                                         params=sum(p.numel() for p in m.parameters()))),
                           outdir / "tower_act.pt")
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"  act   ep{ep:3d}  train L1 {hist[-1]['train']:.4f}  "
                      f"val L1 {hist[-1]['val']:.4f}"
                      f"{'  *' if hist[-1]['val']==best else ''}", flush=True)
        json.dump(hist, open(outdir / "tower_act_history.json", "w"), indent=1)
        print(f"  -> {outdir/'tower_act.pt'}  best val L1 {best:.4f}")


if __name__ == "__main__":
    main()
