"""Score any set of policies on the Part 5.1 fidelity metric, side by side.

    ||a_pred(t) - a_demo(t)||_2  in the raw 8-dim action space, over the 80
    held-out evaluation episodes.

Training losses are not comparable across policies (ACT optimises L1 in
normalised space, BC-RNN a GMM NLL), so this is the only honest comparison.
Each ACT checkpoint carries its own cfg, so v1 (single-frame) and v2 (state
history) are both handled without flags.

    .venv-lerobot/bin/python roboturk/compare_policies.py \
        --act roboturk_act_v1_best.pt roboturk_act_best.pt --names v1 v2
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_act import ACT, GRIP, load_everything  # noqa: E402


class BCRNN(nn.Module):
    def __init__(self, sdim, adim, hidden=256, layers=2, modes=5):
        super().__init__()
        self.modes, self.adim = modes, adim
        self.enc = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.rnn = nn.LSTM(256, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, modes * (2 * adim + 1))

    def forward(self, s):
        B, T, _ = s.shape
        o, _ = self.rnn(self.enc(s.reshape(B * T, -1)).reshape(B, T, -1))
        p = self.head(o).reshape(B, T, self.modes, 2 * self.adim + 1)
        mu, logit = p[..., :self.adim], p[..., -1]
        return (torch.softmax(logit, -1).unsqueeze(-1) * mu).sum(-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", nargs="*", default=["roboturk_act_best.pt"])
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--bcrnn", default="roboturk_bcrnn_state_only.pt")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    S, A, starts, eps, st = load_everything(HERE / "roboturk_20hz.npz", HERE / a.bcrnn)
    s_mu, s_sd, a_mu, a_sd = (st[k] for k in ("s_mu", "s_sd", "a_mu", "a_sd"))
    Sn = ((S - s_mu) / s_sd).astype(np.float32)

    bc = BCRNN(37, 8).to(dev)
    bc.load_state_dict(torch.load(HERE / a.bcrnn, map_location="cpu",
                                  weights_only=False)["model"])
    bc.eval()

    models = [("BC-RNN", bc, 0)]
    names = a.names or [Path(p).stem for p in a.act]
    for path, nm in zip(a.act, names):
        ck = torch.load(HERE / path, map_location="cpu", weights_only=False)
        c = ck["cfg"]; h = c.get("history", 1)
        m = ACT(c["sdim"], c["adim"], chunk=c["chunk"], d=c.get("dim", 512),
                ff=c.get("ff", 3200), n_enc=c.get("enc_layers", 4),
                n_dec=c.get("dec_layers", 7), n_cvae=c.get("cvae_layers", 4),
                dropout=c.get("dropout", 0.1), history=h).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        models.append((f"{nm} (ep{ck['epoch']}, {c['params']/1e6:.1f}M, hist {h})", m, h))

    rows = {}
    for label, m, h in models:
        errs, closed = [], []
        with torch.no_grad():
            for i in eps["eval"]:
                sl = slice(starts[i], starts[i + 1])
                s_n, a_true = Sn[sl], A[sl]
                if h == 0:                                   # BC-RNN: sequential LSTM
                    ah = m(torch.tensor(s_n[None]).to(dev))[0].cpu().numpy()
                else:                                        # ACT: stateless, batch it
                    T = len(s_n)
                    idx = np.arange(T)[:, None] + np.arange(-h + 1, 1)[None, :]
                    np.maximum(idx, 0, out=idx)
                    out = []
                    for b in range(0, T, 512):
                        x = torch.tensor(s_n[idx[b:b + 512]]).to(dev)
                        out.append(m(x)[0][:, 0].cpu().numpy())   # first action of the chunk
                    ah = np.concatenate(out)
                d = ah * a_sd + a_mu - a_true
                errs.append((np.linalg.norm(d, axis=1),
                             np.linalg.norm(d[:, :GRIP], axis=1),
                             np.abs(d[:, GRIP])))
                closed.append(a_true[:, GRIP] > 0.5)
        rows[label] = (tuple(np.concatenate([e[k] for e in errs]) for k in range(3)),
                       np.concatenate(closed))

    b8, b7, bg = rows["BC-RNN"][0]
    print(f"\n=== full 8-dim score (the metric partD/Part 5.1 currently use) ===")
    print(f"{'policy':<32}{'mean':>8}{'median':>8}{'p95':>8}{'closed':>8}{'open':>8}{'vs BC':>9}")
    for label, ((e, _, _), c) in rows.items():
        d = "" if label == "BC-RNN" else f"{(b8.mean()-e.mean())/b8.mean()*100:+7.1f}%"
        print(f"{label:<32}{e.mean():>8.4f}{np.median(e):>8.4f}{np.percentile(e,95):>8.4f}"
              f"{e[c].mean():>8.4f}{e[~c].mean():>8.4f}{d:>9}")

    print(f"\n=== 7 continuous joint targets only (grasp bool excluded) ===")
    print(f"{'policy':<32}{'mean':>8}{'median':>8}{'p95':>8}{'closed':>8}{'open':>8}{'vs BC':>9}")
    for label, ((_, e, _), c) in rows.items():
        d = "" if label == "BC-RNN" else f"{(b7.mean()-e.mean())/b7.mean()*100:+7.1f}%"
        print(f"{label:<32}{e.mean():>8.4f}{np.median(e):>8.4f}{np.percentile(e,95):>8.4f}"
              f"{e[c].mean():>8.4f}{e[~c].mean():>8.4f}{d:>9}")

    print(f"\n=== grasp bool alone (|error| on a {{0,1}} target) ===")
    print(f"{'policy':<32}{'mean':>8}{'median':>8}{'p95':>8}{'vs BC':>9}")
    for label, ((_, _, e), c) in rows.items():
        d = "" if label == "BC-RNN" else f"{(bg.mean()-e.mean())/bg.mean()*100:+7.1f}%"
        print(f"{label:<32}{e.mean():>8.4f}{np.median(e):>8.4f}{np.percentile(e,95):>8.4f}{d:>9}")
    print("\n(+ve = closer to the human than BC-RNN.)")


if __name__ == "__main__":
    main()
