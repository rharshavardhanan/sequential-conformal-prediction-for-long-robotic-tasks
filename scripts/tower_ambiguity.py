"""Is the Route-B observation enough to determine the expert's next action?

The expert carries hidden state the observation does not: `stage`, `timer`, and
`idx` (which cube is next). If two states look nearly identical to the network
but carry different action labels, behaviour cloning is being asked to fit a
one-to-many function -- and least-squares regression answers such a function
with the MEAN of the branches, which lands the arm between the cubes rather
than on one. That is exactly the measured failure (code B).

This quantifies the ambiguity without running the simulator.

    .venv-lerobot/bin/python -m scripts.tower_ambiguity
"""

import argparse
import json
import pathlib

import h5py
import numpy as np

from scripts.tower_target import target_cube

from scripts.tower_target import STACKED  # noqa: F401
N_CUBES = 3
CUBE0 = 23  # absolute cube block starts here in the observation
GRIP_Q = 21  # the 2 gripper joint positions


def load(path):
    O, A, ep = [], [], []
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f if k.startswith("demo_") and f[k].attrs["success"])
        for i, k in enumerate(keys):
            O.append(f[k]["obs"][:])
            A.append(f[k]["act"][:])
            ep.append(np.full(len(O[-1]), i, np.int32))
    return np.concatenate(O), np.concatenate(A), np.concatenate(ep), len(keys)


def intended(cubes, table_z):
    """Delegates to the one shared implementation (scripts/tower_target.py)."""
    i, _ = target_cube(cubes, table_z)
    return -1 if i is None else i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/tower/demos_rel.hdf5")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--queries", type=int, default=4000)
    ap.add_argument("--out", default="data/tower/ambiguity.json")
    a = ap.parse_args()

    O, A, EP, n_ep = load(a.demos)
    print(
        f"{n_ep} successful demos   {len(O):,} states   obs {O.shape[1]}  act {A.shape[1]}"
    )

    mu, sd = O.mean(0), O.std(0) + 1e-6
    On = ((O - mu) / sd).astype(np.float32)
    amu, asd = A.mean(0), A.std(0) + 1e-6
    An = ((A - amu) / asd).astype(np.float32)

    rng = np.random.default_rng(0)
    q = rng.choice(len(O), size=min(a.queries, len(O)), replace=False)

    # --- 1 & 2, one pass ---
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b  keeps the working set at B x N, not B x N x D.
    table_z = O[:, CUBE0 + 2].min()  # a cube resting on the table
    tgt_all = np.array([intended(O[i, CUBE0:CUBE0 + 9].reshape(3, 3), table_z)
                        for i in range(len(O))])
    sq = (On ** 2).sum(1)
    nb_spread, nb_dist, disagree = [], [], []
    B = 128
    for s0 in range(0, len(q), B):
        idx = q[s0:s0 + B]
        d = sq[None, :] + (On[idx] ** 2).sum(1)[:, None] - 2.0 * (On[idx] @ On.T)
        d[EP[None, :] == EP[idx][:, None]] = np.inf   # neighbours from OTHER episodes only
        nn = np.argpartition(d, a.k, axis=1)[:, :a.k]
        for r, row in enumerate(nn):
            nb_spread.append(An[row, :7].std(0).mean())
            nb_dist.append(float(np.sqrt(np.maximum(d[r, row], 0)).mean()))
            disagree.append(float(np.mean(tgt_all[row] != tgt_all[idx[r]])))
    nb_spread = np.asarray(nb_spread)
    nb_dist = np.asarray(nb_dist)
    disagree = np.asarray(disagree)
    global_spread = An[:, :7].std(0).mean()

    # --- 3. how far apart are the candidate cubes? the scale of a wrong pick ---
    seps = []
    for i in q:
        c = O[i, CUBE0 : CUBE0 + 9].reshape(3, 3)
        live = [x for x in c if x[2] - table_z < STACKED]
        if len(live) >= 2:
            seps.append(float(np.linalg.norm(live[0][:2] - live[1][:2])))
    seps = np.asarray(seps)

    ratio = float(nb_spread.mean() / global_spread)
    out = {
        "demos": a.demos,
        "n_demos": n_ep,
        "n_states": int(len(O)),
        "k": a.k,
        "queries": int(len(q)),
        "neighbour_action_spread": float(nb_spread.mean()),
        "global_action_spread": float(global_spread),
        "spread_ratio": ratio,
        "neighbour_obs_distance_mean": float(nb_dist.mean()),
        "target_disagreement_mean": float(disagree.mean()),
        "target_disagreement_over_20pct": float(np.mean(disagree > 0.2)),
        "cube_separation_median_m": float(np.median(seps)) if seps.size else None,
        "cube_separation_mean_m": float(seps.mean()) if seps.size else None,
    }

    print(f"\n{'=' * 70}\nOBSERVATION -> ACTION AMBIGUITY\n{'=' * 70}")
    print(
        f"  action spread among the {a.k} nearest states (other episodes) "
        f"{nb_spread.mean():.4f}"
    )
    print(
        f"  action spread across the whole dataset                  {global_spread:.4f}"
    )
    print(f"  ratio                                                   {ratio:.3f}")
    print(
        f"     -> {ratio:.0%} of the dataset's total action variation survives"
        f" even among\n        states the network sees as nearly identical."
    )
    print(
        f"\n  neighbours whose expert-rule target cube DIFFERS        "
        f"{disagree.mean():.1%}"
    )
    print(
        f"  queries where >20% of neighbours disagree on the target "
        f"{out['target_disagreement_over_20pct']:.1%}"
    )
    print(
        f"\n  median separation between the two candidate cubes       "
        f"{out['cube_separation_median_m']:.3f} m"
    )
    print(
        f"     -> averaging two candidates lands the gripper ~"
        f"{out['cube_separation_median_m'] / 2 * 100:.0f} cm from either."
    )

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
