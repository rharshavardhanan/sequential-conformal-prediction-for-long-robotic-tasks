"""Inspect a LIBERO demo file: HDF5 structure, and export frames as PNG.

uv run python -m scripts.inspect_dataset                     # structure of task 0
uv run python -m scripts.inspect_dataset --demo 0 --frames 6 # + export images
"""

import argparse
import glob
import os

import h5py
import numpy as np


def show_tree(g, prefix="", max_items=None):
    keys = list(g.keys())
    shown = keys if max_items is None else keys[:max_items]
    for i, k in enumerate(shown):
        item = g[k]
        last = i == len(shown) - 1 and len(shown) == len(keys)
        branch = "`-- " if last else "|-- "
        if isinstance(item, h5py.Group):
            print(f"{prefix}{branch}{k}/")
            show_tree(item, prefix + ("    " if last else "|   "), max_items)
        else:
            print(f"{prefix}{branch}{k}  {str(item.shape):18s} {item.dtype}")
    if max_items is not None and len(keys) > max_items:
        print(f"{prefix}... {len(keys) - max_items} more ({len(keys)} total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="path to a *_demo.hdf5")
    ap.add_argument(
        "--task", default="STUDY_SCENE1", help="substring match if --file omitted"
    )
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="how many PNGs to export")
    ap.add_argument("--outdir", default="dataset_preview")
    args = ap.parse_args()

    from libero.libero import get_libero_path

    path = args.file
    if path is None:
        cands = sorted(
            glob.glob(os.path.join(get_libero_path("datasets"), "libero_10", "*.hdf5"))
        )
        match = [c for c in cands if args.task.lower() in os.path.basename(c).lower()]
        path = (match or cands)[0]

    print(f"file: {path}")
    print(f"size: {os.path.getsize(path) / 1e9:.2f} GB\n")

    with h5py.File(path, "r") as f:
        print("=" * 72)
        print("TOP-LEVEL")
        print("=" * 72)
        show_tree(f, max_items=4)

        data = f["data"]
        demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
        print(f"\ndemos: {len(demos)}  ({demos[0]} .. {demos[-1]})")
        for k, v in data.attrs.items():
            s = str(v)
            print(f"  attr {k}: {s[:110]}{'...' if len(s) > 110 else ''}")

        d = data[demos[args.demo]]
        print("\n" + "=" * 72)
        print(f"ONE DEMO: {demos[args.demo]}")
        print("=" * 72)
        show_tree(d)
        for k, v in d.attrs.items():
            if k != "model_file":
                print(f"  attr {k}: {v}")

        acts = d["actions"][:]
        print(f"\nactions: shape={acts.shape} dtype={acts.dtype}")
        print(f"  per-dim min: {np.round(acts.min(0), 3)}")
        print(f"  per-dim max: {np.round(acts.max(0), 3)}")
        print(f"  within [-1,1]: {bool((acts >= -1).all() and (acts <= 1).all())}")
        grip = acts[:, 6]
        print(f"  gripper dim unique values: {np.unique(np.round(grip, 3))[:8]}")
        print("\nfirst 3 action rows (dx dy dz drx dry drz grip):")
        for r in acts[:3]:
            print("   " + "  ".join(f"{x:+.3f}" for x in r))

        # episode lengths across demos -- sizes the collection budget
        lens = [data[k]["actions"].shape[0] for k in demos]
        print(
            f"\nepisode lengths: min={min(lens)} max={max(lens)} "
            f"mean={np.mean(lens):.0f} total_steps={sum(lens)}"
        )

        if args.frames > 0:
            import imageio

            os.makedirs(args.outdir, exist_ok=True)
            obs = d["obs"]
            cams = [k for k in obs.keys() if k.endswith("_rgb")]
            n = obs[cams[0]].shape[0]
            idxs = np.linspace(0, n - 1, args.frames).astype(int)
            print(
                f"\nexporting {args.frames} frames x {len(cams)} cameras -> {args.outdir}/"
            )
            for cam in cams:
                for j in idxs:
                    img = obs[cam][j]
                    # LIBERO stores agentview flipped vertically
                    out = os.path.join(
                        args.outdir, f"{demos[args.demo]}_{cam}_t{j:04d}.png"
                    )
                    imageio.imwrite(out, img[::-1])
            # contact sheet: one row per camera, frames left->right
            rows = [
                np.concatenate([obs[c][j][::-1] for j in idxs], axis=1) for c in cams
            ]
            sheet = np.concatenate(rows, axis=0)
            sheet_path = os.path.join(
                args.outdir, f"contact_sheet_{demos[args.demo]}.png"
            )
            imageio.imwrite(sheet_path, sheet)
            print(f"contact sheet: {sheet_path}  {sheet.shape}")
            print(f"cameras: {cams}")


if __name__ == "__main__":
    main()
