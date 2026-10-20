"""Part D corrected + Part 3 escalation on a deployable trigger + Part 5.1/5.2.

Two changes from the shipped Part D:
  * score = the 7 continuous joint targets, not all 8. The grasp bool enters the
    L2 unnormalised and owns ~98.5% of the threshold-setting error, which is what
    made q_hat_k = 2.28 vacuous (eval coverage 1.000 vs a 0.90 target).
  * escalation triggers on BC-RNN's predicted sigma, not on ||a_pred - a_demo||.
    The latter needs the human's action and cannot run on a robot.

Coverage claims are made only where exchangeability holds: episodes are the
exchangeable unit, so per-episode statistics carry a guarantee and pooled
timesteps do not. Both are reported, labelled.
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_act import ACT, GRIP, load_everything  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ALPHA, K = 0.10, 2
ALPHA_K = ALPHA / K


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


def conformal(scores, alpha):
    s = np.sort(np.asarray([x for x in scores if np.isfinite(x)], float)); n = s.size
    rank = int(math.ceil((1 - alpha) * (n + 1)))
    if rank > n:
        return float("inf"), n, rank, False
    return float(s[rank - 1]), n, rank, True


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


@torch.no_grad()
def episode(i):
    """Per-timestep (BC-RNN err7, ACT err7, sigma, closed-mask) for one episode."""
    sl = slice(starts[i], starts[i + 1])
    s_n, a_true = Sn[sl], A[sl]
    mu, ls, lg = bc(torch.tensor(s_n[None]).to(DEV))
    w = torch.softmax(lg, -1)
    mean = (w.unsqueeze(-1) * mu).sum(-2)
    sigma = (w.unsqueeze(-1) * ls.exp()).sum(-2).mean(-1)[0].cpu().numpy()
    pb = mean[0].cpu().numpy() * a_sd + a_mu
    T = len(s_n)
    idx = np.maximum(np.arange(T)[:, None] + np.arange(-H + 1, 1)[None, :], 0)
    pa = np.concatenate([act(torch.tensor(s_n[idx[b:b + 512]]).to(DEV))[0][:, 0].cpu().numpy()
                         for b in range(0, T, 512)]) * a_sd + a_mu
    return (np.linalg.norm((pb - a_true)[:, :GRIP], axis=1),
            np.linalg.norm((pa - a_true)[:, :GRIP], axis=1),
            sigma, a_true[:, GRIP] > 0.5)


cal = {i: episode(i) for i in eps["cal"]}
ev = {i: episode(i) for i in eps["eval"]}
PH = ("closed", "open")
sel = lambda cl, p: cl if p == "closed" else ~cl
out = {"alpha": ALPHA, "K": K, "alpha_k": ALPHA_K,
       "score": "||a_pred - a_demo||_2 over the 7 continuous joint targets "
                "(grasp bool EXCLUDED -- it owned ~98.5% of the old threshold)",
       "trigger": "BC-RNN predicted sigma (deployable: no ground-truth action needed)"}

# ---------------------------------------------------------------- Task 1: q_hat_k
print("=" * 78)
print("TASK 1 -- safety threshold q_hat_k on the CORRECTED 7-dim score")
print("=" * 78)
AGGS = {"max": np.max, "p95": lambda x: np.percentile(x, 95),
        "p90": lambda x: np.percentile(x, 90), "p75": lambda x: np.percentile(x, 75)}
print(f"{'agg':>5}{'phase':>8}{'q_hat':>9}{'eval cov':>10}{'target':>8}{'eval stat max':>15}"
      f"{'informative?':>14}")
qk = {}
out["per_category"] = {}
for agg_name, agg in AGGS.items():
    out["per_category"][agg_name] = {}
    for p in PH:
        cm = [float(agg(cal[i][0][sel(cal[i][3], p)])) for i in eps["cal"]
              if sel(cal[i][3], p).any()]
        em = [float(agg(ev[i][0][sel(ev[i][3], p)])) for i in eps["eval"]
              if sel(ev[i][3], p).any()]
        q, n, rank, feas = conformal(cm, ALPHA_K)
        cov = float(np.mean(np.asarray(em) <= q))
        if agg_name == "max":
            qk[p] = q
        info = "yes" if cov < 0.999 else "NO (vacuous)"
        out["per_category"][agg_name][p] = dict(q=q, n_cal=n, rank=rank, feasible=feas,
                                                n_eval=len(em), coverage=cov, eval_stat_max=max(em))
        print(f"{agg_name:>5}{p:>8}{q:>9.4f}{cov:>10.3f}{1-ALPHA:>8.2f}{max(em):>15.4f}{info:>14}")
print("\n  (shipped 8-dim version: q_closed 2.2836 / q_open 2.2111, coverage 1.000,")
print("   eval max 1.213 -- a threshold that could never be violated.)")

# --------------------------------------------- Task 2: escalation threshold on sigma
print("\n" + "=" * 78)
print("TASK 2 -- escalation threshold on sigma (deployable trigger)")
print("=" * 78)
out["escalation"] = {}
lam = {}
for rate in (0.05, 0.10, 0.20):
    lam[rate] = {}
    row = {}
    for p in PH:
        pool = np.concatenate([cal[i][2][sel(cal[i][3], p)] for i in eps["cal"]
                               if sel(cal[i][3], p).any()])
        L = float(np.quantile(pool, 1 - rate))
        lam[rate][p] = L
        hit = tot = 0
        per_ep = []
        for i in eps["eval"]:
            m = sel(ev[i][3], p)
            if not m.any():
                continue
            x = ev[i][2][m]
            hit += int((x > L).sum()); tot += len(x)
            per_ep.append(float((x > L).mean()))
        # episodes ARE exchangeable -> a conformal bound on the per-episode rate is valid
        cal_ep_rates = []
        for i in eps["cal"]:
            m = sel(cal[i][3], p)
            if m.any():
                cal_ep_rates.append(float((cal[i][2][m] > L).mean()))
        bound, _, _, feas = conformal(cal_ep_rates, ALPHA)
        held = float(np.mean(np.asarray(per_ep) <= bound))
        row[p] = dict(lam=L, eval_rate=hit / tot, ep_rate_bound=bound,
                      bound_holds_on_eval=held, feasible=feas)
    out["escalation"][f"{rate:.2f}"] = row
    print(f"\ntarget {rate:.0%} of timesteps:")
    for p in PH:
        r = row[p]
        print(f"   {p:<7} lambda {r['lam']:.4f}   eval firing {r['eval_rate']:.2%}   "
              f"per-episode rate <= {r['ep_rate_bound']:.2%} "
              f"(holds on {r['bound_holds_on_eval']:.0%} of eval episodes, target {1-ALPHA:.0%})")
print("\n  The per-episode bound is a VALID conformal statement (episodes are")
print("  exchangeable). The pooled lambda itself is a budget heuristic, not a")
print("  coverage guarantee -- the safety guarantee lives in q_hat_k alone.")

# ------------------------------------------------- Task 3: arbitrated fidelity H1/H2
print("\n" + "=" * 78)
print("TASK 3 -- arbitrated fidelity (Part 5.1 / 5.2)")
print("=" * 78)
eb = np.concatenate([ev[i][0] for i in eps["eval"]])
ea = np.concatenate([ev[i][1] for i in eps["eval"]])
sg = np.concatenate([ev[i][2] for i in eps["eval"]])
cl = np.concatenate([ev[i][3] for i in eps["eval"]])
print(f"BC-RNN alone {eb.mean():.4f}      ACT alone {ea.mean():.4f}      "
      f"(ACT full-time gain {(eb.mean()-ea.mean())/eb.mean()*100:+.1f}%)\n")
print(f"{'target':>7}{'ACT used':>10}{'arbitrated':>12}{'H1 vs BC':>10}{'H2 vs ACT':>11}"
      f"{'gain recovered':>16}")
out["arbitration"] = {}
for rate in (0.05, 0.10, 0.20):
    esc = np.zeros(len(eb), bool)
    for p in PH:
        m = sel(cl, p)
        esc |= m & (sg > lam[rate][p])
    arb = np.where(esc, ea, eb)
    h1 = (eb.mean() - arb.mean()) / eb.mean() * 100
    h2 = (arb.mean() - ea.mean()) / ea.mean() * 100
    rec = (eb.mean() - arb.mean()) / (eb.mean() - ea.mean()) * 100
    out["arbitration"][f"{rate:.2f}"] = dict(act_fraction=float(esc.mean()),
                                             arbitrated=float(arb.mean()),
                                             h1_vs_bcrnn_pct=h1, h2_vs_act_pct=h2,
                                             gain_recovered_pct=rec)
    print(f"{rate:>6.0%}{esc.mean():>10.2%}{arb.mean():>12.4f}{h1:>+9.1f}%{h2:>+10.1f}%"
          f"{rec:>15.1f}%")
print("\n  'gain recovered' = how much of ACT's full-time advantage arbitration keeps.")
print("  H1 positive = better than BC-RNN alone. H2 near 0 = close to ACT alone.")

# leading-indicator check (replaces the meaningless q_esc < q_hat_k comparison)
viol = np.zeros(len(eb), bool)
for p in PH:
    m = sel(cl, p)
    viol |= m & (eb > qk[p])
esc10 = np.zeros(len(eb), bool)
for p in PH:
    m = sel(cl, p)
    esc10 |= m & (sg > lam[0.10][p])
print(f"\nLeading-indicator check: safety threshold exceeded on {viol.mean():.2%} of "
      f"timesteps,\n  escalation fires on {esc10.mean():.2%} (10% target). "
      f"Escalation must be the larger: {'PASS' if esc10.mean() > viol.mean() else 'FAIL'}")
out["leading_indicator"] = dict(safety_violation_rate=float(viol.mean()),
                                escalation_rate=float(esc10.mean()),
                                passes=bool(esc10.mean() > viol.mean()))

json.dump(out, open(HERE / "partD_corrected.json", "w"), indent=2, default=float)
print("\nwrote partD_corrected.json")
