"""Rescue curves: round-1 PickAndPlace, composed Reach->Slide, composed Reach->Push."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
I2 = json.load(open(os.path.join(HERE, "..", "results", "iter2", "iter2_rescue.json")))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for path, label, col, mk in [("results", "Reach -> Slide (gate failed)", "#1f77b4", "o"),
                             ("results_push", "Reach -> Push, fine-tuned Push expert", "#2ca02c", "D")]:
    fp = os.path.join(HERE, path, "part3_horizon_rescue.json")
    if not os.path.exists(fp):
        continue
    P = json.load(open(fp))
    t = [int(k) for k in P["rescue_curve"]]
    ax.fill_between(t, [v["wilson95"][0] for v in P["rescue_curve"].values()],
                    [v["wilson95"][1] for v in P["rescue_curve"].values()], color=col, alpha=.18)
    ax.plot(t, [v["rate"] for v in P["rescue_curve"].values()], mk + "-", color=col, ms=5, label=label)
t2 = [int(k) for k in I2["rescue_curve"]]
ax.plot(t2, [v["rate"] for v in I2["rescue_curve"].values()], "s--", color="#d62728", label="round-1 PickAndPlace")
ax.axvline(15, color="k", lw=.8, ls=":", label="G3.1 threshold (15% of 100-step horizon)")
ax.set_xlabel("expert pair takes permanent control at step t")
ax.set_ylabel("success on 100 cheap-failure seeds (Wilson 95%)")
ax.set_ylim(-.02, 1.02)
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
for d in ("results", "results_push"):
    os.makedirs(os.path.join(HERE, d, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, d, "figures", "rescue_curve.png"), dpi=150)
