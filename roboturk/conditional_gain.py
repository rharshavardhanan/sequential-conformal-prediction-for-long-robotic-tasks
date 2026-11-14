"""Does ACT beat BC-RNN *specifically* where BC-RNN is uncertain?

The overall gain (7.7% on continuous dims) is not the arbitration claim. The claim
is that gain concentrates on the escalated slice. This ranks eval timesteps by
DEPLOYABLE uncertainty signals -- computed from BC-RNN's GMM head alone, needing no
ground-truth action -- and reports ACT's advantage within each top-q% slice.

A trigger that needs a_demo is an oracle and cannot run on a robot; these can.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_act import ACT, GRIP, load_everything  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
        return p[..., :self.adim], p[..., self.adim:2 * self.adim].clamp(-5, 2), p[..., -1]


S, A, starts, eps, st = load_everything(HERE / "roboturk_20hz.npz",
                                        HERE / "roboturk_bcrnn_state_only.pt")
s_mu, s_sd, a_mu, a_sd = (st[k] for k in ("s_mu", "s_sd", "a_mu", "a_sd"))
Sn = ((S - s_mu) / s_sd).astype(np.float32)

bc = BCRNN(37, 8).to(DEV)
bc.load_state_dict(torch.load(HERE / "roboturk_bcrnn_state_only.pt", map_location="cpu",
                              weights_only=False)["model"])
bc.eval()
ck = torch.load(HERE / "roboturk_act_best.pt", map_location="cpu", weights_only=False)
c = ck["cfg"]; H = c.get("history", 1)
act = ACT(c["sdim"], c["adim"], chunk=c["chunk"], d=c.get("dim", 512), ff=c.get("ff", 3200),
          n_enc=c.get("enc_layers", 4), n_dec=c.get("dec_layers", 7),
          n_cvae=c.get("cvae_layers", 4), dropout=c.get("dropout", 0.1), history=H).to(DEV)
act.load_state_dict(ck["model"]); act.eval()
print(f"ACT v2 epoch {ck['epoch']}, {c['params']/1e6:.1f}M, history {H}\n")

e_bc, e_act, U = [], [], {k: [] for k in ("mode_spread", "sigma", "entropy")}
with torch.no_grad():
    for i in eps["eval"]:
        sl = slice(starts[i], starts[i + 1])
        s_n, a_true = Sn[sl], A[sl]
        mu, ls, lg = bc(torch.tensor(s_n[None]).to(DEV))
        w = torch.softmax(lg, -1)                                  # (1,T,K)
        mean = (w.unsqueeze(-1) * mu).sum(-2)                       # (1,T,adim)
        # deployable uncertainty: no ground truth used anywhere below
        spread = ((w.unsqueeze(-1) * (mu - mean.unsqueeze(-2)) ** 2).sum(-2)).sum(-1).sqrt()
        sigma = (w.unsqueeze(-1) * ls.exp()).sum(-2).mean(-1)
        ent = -(w * (w + 1e-9).log()).sum(-1)
        U["mode_spread"].append(spread[0].cpu().numpy())
        U["sigma"].append(sigma[0].cpu().numpy())
        U["entropy"].append(ent[0].cpu().numpy())
        pb = mean[0].cpu().numpy() * a_sd + a_mu

        T = len(s_n)
        idx = np.maximum(np.arange(T)[:, None] + np.arange(-H + 1, 1)[None, :], 0)
        out = [act(torch.tensor(s_n[idx[b:b + 512]]).to(DEV))[0][:, 0].cpu().numpy()
               for b in range(0, T, 512)]
        pa = np.concatenate(out) * a_sd + a_mu

        e_bc.append(np.linalg.norm((pb - a_true)[:, :GRIP], axis=1))
        e_act.append(np.linalg.norm((pa - a_true)[:, :GRIP], axis=1))

e_bc, e_act = np.concatenate(e_bc), np.concatenate(e_act)
U = {k: np.concatenate(v) for k, v in U.items()}
overall = (e_bc.mean() - e_act.mean()) / e_bc.mean() * 100
print(f"ALL {len(e_bc):,} eval timesteps (7 continuous dims):")
print(f"   BC-RNN {e_bc.mean():.4f}   ACT {e_act.mean():.4f}   ACT gain {overall:+.1f}%\n")

print(f"{'signal':<14}{'top q%':>8}{'BC-RNN':>9}{'ACT':>9}{'ACT gain':>10}{'vs overall':>12}")
for name, u in U.items():
    for q in (5, 10, 20, 30):
        thr = np.percentile(u, 100 - q)
        m = u >= thr
        g = (e_bc[m].mean() - e_act[m].mean()) / e_bc[m].mean() * 100
        print(f"{name:<14}{q:>7}%{e_bc[m].mean():>9.4f}{e_act[m].mean():>9.4f}"
              f"{g:>9.1f}%{g - overall:>+11.1f}pp")
    print()
print("A signal is useful for escalation only if 'vs overall' is clearly positive:")
print("that means ACT earns more on the slice you'd pay for than on average.")
