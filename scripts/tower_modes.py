"""Does BC-RNN's GMM already know where the cube is, and does the adapter throw it away?

The k-NN analysis says the supervision is one-to-many (37% of total action variance
survives among indistinguishable states; 44% of neighbours want a different cube).
A 5-mode GMM head CAN represent that. But tower_policies.BCRNNPolicy.act() collapses
it with gmm_mean() -- the mixture mean -- which is exactly the average of the branches.

If the modes are far apart and picking the most likely one targets better than the
mean, the averaging is an inference bug, not a learning failure, and the fix is nearly
free. If the modes are collapsed on top of each other, the network never learned the
branches and the fix has to be in the data.

    .venv-lerobot/bin/python -m scripts.tower_modes --n 20
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from scripts.tower_collect import make_env, observation
from scripts.tower_control import joint_action, eef_pos
from scripts.tower_policies import BCRNNPolicy
from scripts.tower_diagnose import (
    CLOSE_OK,
    LIFTED,
    NEAR,
    cubes_of,
    intended_cube,
    classify,
    summarise,
)

GRIP = 7


class ModeBCRNN(BCRNNPolicy):
    """Same weights, three ways of turning the mixture into one action."""

    def __init__(self, ckpt, device="mps", mode="mean", seed=0):
        super().__init__(ckpt, device)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.last = None

    @torch.no_grad()
    def act(self, obs):
        s = torch.tensor(self._norm(obs)[None, None]).to(self.dev)
        mu, logstd, logit, self.hx = self.m(s, self.hx)
        w = torch.softmax(logit, -1)[0, 0].cpu().numpy()  # (K,)
        m = mu[0, 0].cpu().numpy()  # (K, adim)
        mean = (w[:, None] * m).sum(0)
        if self.mode == "mean":
            a = mean
        elif self.mode == "argmax":
            a = m[int(np.argmax(w))]
        elif self.mode == "sample":
            a = m[self.rng.choice(len(w), p=w / w.sum())]
        else:
            raise ValueError(self.mode)
        # how far apart are the modes, in the units the action is measured in?
        self.last = {
            "spread": float(
                np.sqrt((w[:, None] * (m - mean) ** 2).sum(0)[:GRIP].sum())
            ),
            "top_w": float(w.max()),
            "entropy": float(-(w * np.log(w + 1e-9)).sum()),
        }
        a = self._denorm(a)
        return a[:GRIP], float(a[GRIP] > 0.5)


def run(env, pol, max_steps, ep):
    env.reset()
    pol.reset()
    table_z = env.table_offset[2]
    grasps, stats = [], []
    pending, prev_grip, prev_z = None, False, None
    lifted = carried = False
    for t in range(max_steps):
        obs = observation(env)[: pol.cfg["sdim"]]
        q, g = pol.act(obs)
        stats.append(pol.last)
        gc = g >= 0.5
        cubes, e = cubes_of(env), eef_pos(env)
        tgt_i, _ = intended_cube(cubes, table_z)
        dists = np.linalg.norm(cubes - e, axis=1)
        d_int = float(dists[tgt_i]) if tgt_i is not None else np.nan
        holding = bool(
            dists.min() < NEAR and cubes[int(np.argmin(dists))][2] - table_z > LIFTED
        )
        if holding:
            lifted = True
            if prev_z is not None and abs(e[2] - prev_z) <= 0.004:
                carried = True
        if gc and not prev_grip:
            pending = {
                "t": t,
                "intended": None if tgt_i is None else int(tgt_i),
                "nearest": int(np.argmin(dists)),
                "dist_intended": d_int,
                "dist_nearest": float(dists.min()),
                "min_dist_intended_before": d_int,
                "height_above_cube": float(e[2] - cubes[tgt_i][2])
                if tgt_i is not None
                else np.nan,
                "grasped": False,
                "resolved": False,
            }
        if pending is not None and not pending["resolved"]:
            if t - pending["t"] >= 60 or (t - pending["t"] > 3 and not gc):
                pending["grasped"] = bool(holding)
                pending["resolved"] = True
                grasps.append(pending)
                pending = None
        prev_grip, prev_z = gc, float(e[2])
        env.step(joint_action(env, q, 1.0 if gc else -1.0))
        if env._check_success():
            break
    if pending is not None:
        pending["grasped"] = False
        grasps.append(pending)
    succ = bool(env._check_success())
    return {
        "policy": pol.mode,
        "episode": ep,
        "success": succ,
        "height": int(env.tower_height()),
        "steps": t + 1,
        "grasps": grasps,
        "n_grasp_attempts": len(grasps),
        "failure": classify(succ, grasps, lifted, carried, int(env.tower_height()), {}),
    }, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/tower/rel/tower_bcrnn.pt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="data/tower/mode_collapse.json")
    a = ap.parse_args()

    out = {"checkpoint": a.ckpt, "n_episodes": a.n, "modes": {}}
    for mode in ("mean", "argmax", "sample"):
        env = make_env(seed=a.seed)
        pol = ModeBCRNN(a.ckpt, device=a.device, mode=mode, seed=a.seed)
        eps, st = [], []
        for i in range(a.n):
            r, s = run(env, pol, a.max_steps, i)
            eps.append(r)
            st += s
        env.close()
        s = summarise(eps, mode)
        spread = np.array([x["spread"] for x in st])
        topw = np.array([x["top_w"] for x in st])
        s["mode_spread_median"] = float(np.median(spread))
        s["mode_spread_p90"] = float(np.percentile(spread, 90))
        s["top_weight_median"] = float(np.median(topw))
        out["modes"][mode] = s
        md = s["close_dist_median"]
        print(
            f"  {mode:<7} success {s['success_rate']:>4.0%}  height {s['height_mean']:.2f}  "
            f"close {(md * 100 if md else float('nan')):>5.1f} cm  "
            f"<3cm {s['within_3cm'] or 0:>4.0%}  grasp {s['grasp_success_rate']:>4.0%}  "
            f"failures {s['failure_codes']}",
            flush=True,
        )

    sp = out["modes"]["mean"]
    print(
        f"\n  GMM mode spread (normalised action units): median "
        f"{sp['mode_spread_median']:.4f}  p90 {sp['mode_spread_p90']:.4f}"
    )
    print(f"  top mixture weight: median {sp['top_weight_median']:.3f}")
    print("\n  A large spread with a top weight near 1/K means the branches exist and")
    print("  the mean discards them. A spread near zero means the modes collapsed and")
    print("  the fix must be in the data, not the inference.")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
