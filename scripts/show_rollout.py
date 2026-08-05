"""Read the collected rollout values.

# list every rollout in the file
.venv-lerobot/bin/python -m scripts.show_rollout rollouts/task0.hdf5

# per-step values for one rollout
.venv-lerobot/bin/python -m scripts.show_rollout rollouts/task0.hdf5 -r 0

# dump one rollout to CSV (opens in Excel/pandas)
.venv-lerobot/bin/python -m scripts.show_rollout rollouts/task0.hdf5 -r 0 --csv out.csv
"""

import argparse

import h5py
import numpy as np

from scripts.subphases import PHASES, segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-r", "--rollout", type=int, default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--every", type=int, default=20, help="print every Nth step")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        keys = sorted(k for k in f if k.startswith("rollout_"))

        if args.rollout is None:
            print(f"file    : {args.path}")
            print(f"policy  : {f.attrs.get('policy_repo')}")
            print(f"sha256  : {str(f.attrs.get('policy_sha256'))[:32]}...")
            print(f"task    : {f.attrs.get('task_language')}")
            print(
                f"scaling : pos x {f.attrs.get('pos_scale')}  rot x {f.attrs.get('rot_scale')}"
            )
            print(f"\n{len(keys)} rollouts\n")
            print(
                f"{'rollout':>9} {'outcome':>8} {'steps':>6} "
                f"{'err_norm mean':>13} {'p95':>7} {'max':>7} {'err_raw mean':>13}"
            )
            for k in keys:
                g = f[k]
                en, er = g["err_norm"][:], g["err_raw"][:]
                print(
                    f"{k.split('_')[1]:>9} "
                    f"{'SUCCESS' if g.attrs['success'] else 'fail':>8} "
                    f"{g.attrs['steps']:>6} {en.mean():>13.3f} "
                    f"{np.percentile(en, 95):>7.3f} {en.max():>7.3f} {er.mean():>13.5f}"
                )
            if keys:
                allen = np.concatenate([f[k]["err_norm"][:] for k in keys])
                print(
                    f"\npooled err_norm: mean={allen.mean():.3f} "
                    f"median={np.median(allen):.3f} p95={np.percentile(allen, 95):.3f}"
                )
                print("(expect ~0.75: OSC realises 0.251 of each commanded delta)")
            return

        k = f"rollout_{args.rollout:04d}"
        if k not in f:
            print(f"{k} not in file; have {keys[0]}..{keys[-1]}")
            return
        g = f[k]
        labels, segs = segment(g["actions"][:, 6])
        print(
            f"{k}  {'SUCCESS' if g.attrs['success'] else 'FAIL'}  "
            f"{g.attrs['steps']} steps  init_state={g.attrs['init_state_index']}"
        )
        print(f"\navailable arrays:")
        for name in g:
            print(f"  {name:16s} {str(g[name].shape):14s} {g[name].dtype}")

        print(f"\nsub-phases ({len(segs)} segments):")
        for s, e, p in segs:
            print(f"  [{s:4d}:{e:4d}] {p}")

        print(f"\nper-step values (every {args.every}th):")
        print(
            f"{'t':>5} {'phase':>10} {'err_raw':>9} {'err_norm':>9} {'cmd_norm':>9} "
            f"{'planned_pos (x,y,z)':>28} {'executed_pos':>28}"
        )
        for t in range(0, len(labels), args.every):
            pp, ep = g["planned_pos"][t], g["executed_pos"][t]
            print(
                f"{t:>5} {PHASES[labels[t]]:>10} {g['err_raw'][t]:>9.5f} "
                f"{g['err_norm'][t]:>9.3f} {g['cmd_norm'][t]:>9.5f} "
                f"  ({pp[0]:+.4f},{pp[1]:+.4f},{pp[2]:+.4f})"
                f"  ({ep[0]:+.4f},{ep[1]:+.4f},{ep[2]:+.4f})"
            )

        print("\nerr_norm by phase:")
        for i, p in enumerate(PHASES):
            m = labels == i
            if m.any():
                v = g["err_norm"][:][m]
                print(
                    f"  {p:10s} n={m.sum():4d}  mean={v.mean():.3f}  "
                    f"p95={np.percentile(v, 95):.3f}  max={v.max():.3f}"
                )

        if args.csv:
            cols = ["err_raw", "err_norm", "cmd_norm"]
            arr = np.column_stack(
                [np.arange(len(labels)), [PHASES[i] for i in labels]]
                + [g[c][:] for c in cols]
                + [g["planned_pos"][:], g["executed_pos"][:], g["actions"][:]]
            )
            hdr = ",".join(
                ["t", "phase"]
                + cols
                + [f"planned_{a}" for a in "xyz"]
                + [f"executed_{a}" for a in "xyz"]
                + [f"action_{i}" for i in range(7)]
            )
            np.savetxt(args.csv, arr, delimiter=",", fmt="%s", header=hdr, comments="")
            print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
