"""Cheap policy for the composed task (needed by the Part 3 diagnostics; methodology = round 1 Part 3).

One MLP 23-64-64-3 (ReLU, tanh): input = slide observation (18) + current phase goal (3) +
phase one-hot (2).  MSE behaviour cloning on composed-expert (obs, action) pairs, whole
episodes in seed order.  Same pre-registered budget rule as round 1: smallest of
{1000, 2000, 5000, 10000} steps with selection success >= 0.30 AND >= 0.20 below the
composed expert (0.865); fallbacks as in round 1.
"""
import json
import os

import numpy as np
import torch

import compose as K  # adds panda_gym_demo/ to sys.path
import common as C

BUDGETS = (1000, 2000, 5000, 10000)
N_SELECT = 100
SEEDS = {"distill": 140_000, "select": 150_000}
HIDDEN, EPOCHS, BATCH, LR = 64, 400, 256, 1e-3

P2 = json.load(open(os.path.join(K.RESULTS, "part2_compose.json")))
EXP_SR = P2["composed_success_rate"]
env = K.Composed()
ex = K.Experts()

X, Y, i, n_succ = [], [], 0, 0
while len(X) < max(BUDGETS):
    env.reset(SEEDS["distill"] + i)
    ev = "continue"
    while ev in ("continue", "phase_switch"):
        a = ex(env)
        X.append(env.flat())
        Y.append(a)
        ev, _ = env.step(a)
    n_succ += ev == "success"
    i += 1
X, Y = np.asarray(X, np.float32), np.clip(np.asarray(Y, np.float32), -1, 1)
print("collected %d steps from %d composed-expert episodes (%d successful)" % (len(X), i, n_succ))


def train(n, seed=0):
    torch.manual_seed(seed)
    x, y = X[:n], Y[:n]
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.from_numpy((x - mu) / sd), torch.from_numpy(y)
    net = C.build_mlp(x.shape[1], 3, HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt), generator=g)
        for b in range(0, len(xt), BATCH):
            idx = perm[b:b + BATCH]
            loss = torch.mean((net(xt[idx]) - yt[idx]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net, mu, sd, float(torch.mean((net(xt) - yt) ** 2))


rows = {}
for B in BUDGETS:
    net, mu, sd, mse = train(B)
    path = os.path.join(K.MODELS, ("cheap_composed_%d.pt" if K.TASK == "slide" else "cheap_" + K.TASK + "_%d.pt") % B)
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "hidden": HIDDEN, "budget_steps": B}, path)
    cheap = K.Cheap(path)
    recs = [K.run_episode(env, SEEDS["select"] + j, cheap, None, "cheap") for j in range(N_SELECT)]
    k = sum(r["success"] for r in recs)
    rows[B] = {"train_mse": mse, "success": k, "n": N_SELECT, "success_rate": k / N_SELECT,
               "wilson95": C.wilson(k, N_SELECT), "phaseA_success_rate": float(np.mean([r["phaseA_success"] for r in recs])),
               "path": os.path.basename(path)}
    print(B, rows[B])

ok = [B for B in BUDGETS if rows[B]["success_rate"] >= 0.30 and rows[B]["success_rate"] <= EXP_SR - 0.20]
if ok:
    chosen, why = min(ok), "smallest budget with success>=0.30 and gap>=0.20"
else:
    gap = [B for B in BUDGETS if rows[B]["success_rate"] <= EXP_SR - 0.20]
    chosen, why = ((max(gap, key=lambda B: rows[B]["success_rate"]), "no budget met both; best with gap>=0.20")
                   if gap else (min(BUDGETS), "no budget kept a 0.20 gap; smallest budget"))
out = {"budgets": rows, "chosen_budget": chosen, "rule_outcome": why, "composed_expert_rate": EXP_SR,
       "expert_steps": int(len(X)), "expert_episodes": i, "cheap_params": K.Cheap(os.path.join(K.MODELS, rows[chosen]["path"])).params(),
       "expert_actor_params": {"reach": ex.a.params(), K.PHASES[1]: ex.b.params()}, "seeds": SEEDS}
json.dump(out, open(os.path.join(K.RESULTS, "part4a_distill.json"), "w"), indent=2)
print("CHOSEN", chosen, why)

