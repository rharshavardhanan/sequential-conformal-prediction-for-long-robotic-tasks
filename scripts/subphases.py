"""Split a trajectory into sub-phases for stage-aware conformal thresholds.

BDDL goals give only two milestones per libero_10 task (every goal is a two-term
`And`), and each milestone still contains reach + grasp + carry + place.
Calibrating on those buckets is barely better than one global threshold, so the
stage labels come from gripper events instead.

The gripper command (`action[:, 6]`) is bang-bang -- exactly -1 (open) or +1
(closed) -- so its transitions are clean segmentation points:

    reach     gripper open, approaching     -> fast, loose tolerance
    grasp     window around an open->close  -> slow, tight tolerance
    transport gripper closed, carrying      -> fast, loose
    release   window around a close->open   -> slow, tight

`grasp` and `release` are split out as their own short windows because that is
where a small tracking deviation actually means something; lumping them into the
neighbouring reach/transport segment is what washes the signal out.

  .venv-lerobot/bin/python -m scripts.subphases --demo    # validate on human demos
  .venv-lerobot/bin/python -m scripts.subphases --rollouts rollouts/task0.hdf5
"""

import argparse

import numpy as np

PHASES = ["reach", "grasp", "transport", "release"]
PHASE_ID = {p: i for i, p in enumerate(PHASES)}


def segment(gripper_cmd, window=5):
    """Per-timestep phase labels from the bang-bang gripper command.

    Args:
        gripper_cmd: (T,) array, negative = open, positive = closed.
        window: steps either side of a transition counted as grasp/release.

    Returns:
        labels (T,) int array indexing PHASES, and a list of
        (start, end, phase_name) segments.
    """
    g = np.asarray(gripper_cmd)
    T = len(g)
    closed = g > 0

    # baseline: closed -> transport, open -> reach
    labels = np.where(closed, PHASE_ID["transport"], PHASE_ID["reach"]).astype(np.int8)

    # transitions: +1 where it closes, -1 where it opens
    trans = np.diff(closed.astype(np.int8))
    for t in np.nonzero(trans)[0]:
        lo, hi = max(0, t - window + 1), min(T, t + window + 1)
        labels[lo:hi] = PHASE_ID["grasp"] if trans[t] > 0 else PHASE_ID["release"]

    segs, start = [], 0
    for i in range(1, T + 1):
        if i == T or labels[i] != labels[start]:
            segs.append((start, i, PHASES[labels[start]]))
            start = i
    return labels, segs


def summarize(labels_list, name):
    all_lab = np.concatenate(labels_list)
    print(f"\n=== {name}: {len(labels_list)} trajectories, {len(all_lab)} steps ===")
    print(f"{'phase':11s} {'steps':>8s} {'share':>7s}  {'mean len':>9s}")
    for p in PHASES:
        i = PHASE_ID[p]
        n = int((all_lab == i).sum())
        lens = []
        for lab in labels_list:
            runs = np.diff(
                np.flatnonzero(np.concatenate(([True], lab[1:] != lab[:-1], [True])))
            )
            vals = lab[np.flatnonzero(np.concatenate(([True], lab[1:] != lab[:-1])))]
            lens += list(runs[vals == i])
        ml = np.mean(lens) if lens else 0
        print(f"{p:11s} {n:8d} {n / len(all_lab) * 100:6.1f}% {ml:9.1f}")
    nseg = np.mean(
        [
            len(segment(np.where(l == PHASE_ID["transport"], 1, -1))[1])
            for l in labels_list
        ]
    )
    return all_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--demo", action="store_true", help="validate on LIBERO human demos"
    )
    ap.add_argument("--rollouts", default=None, help="path to a collected task*.hdf5")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    import h5py

    if args.demo:
        import os

        from libero.libero import benchmark, get_libero_path

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        path = os.path.join(
            get_libero_path("datasets"), suite.get_task_demonstration(args.task_id)
        )
        print(f"demos: {path}")
        labs, segcounts = [], []
        with h5py.File(path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[
                : args.limit
            ]
            for k in keys:
                a = f["data"][k]["actions"][:]
                lab, segs = segment(a[:, 6], args.window)
                labs.append(lab)
                segcounts.append(len(segs))
        summarize(labs, f"human demos task {args.task_id}")
        print(
            f"\nsegments per trajectory: mean={np.mean(segcounts):.1f} "
            f"min={min(segcounts)} max={max(segcounts)}"
        )
        print("(BDDL would give 2 milestones per task -- this is the granularity gain)")

    if args.rollouts:
        labs, succ_labs, fail_labs = [], [], []
        with h5py.File(args.rollouts, "r") as f:
            for k in sorted(x for x in f if x.startswith("rollout_")):
                a = f[k]["actions"][:]
                lab, _ = segment(a[:, 6], args.window)
                labs.append(lab)
                (succ_labs if f[k].attrs["success"] else fail_labs).append(lab)
        summarize(labs, f"collected {args.rollouts}")
        if succ_labs and fail_labs:
            print(
                f"\nsuccess trajectories: {len(succ_labs)}   "
                f"failure trajectories: {len(fail_labs)}"
            )
            print("failures per phase (the number that sets CP evaluation precision):")
            for p in PHASES:
                i = PHASE_ID[p]
                n = sum(1 for l in fail_labs if (l == i).any())
                ci = 1.96 * np.sqrt(0.25 / n) if n else float("nan")
                print(
                    f"  {p:11s} {n:4d} failing trajectories touch this phase"
                    f"   +/-{ci:.2f} CI on a 50% detection rate"
                )


if __name__ == "__main__":
    main()
