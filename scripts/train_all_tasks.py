"""Train N libero_10 tasks in sequence and record a frozen-checkpoint manifest.

Each task runs in its own subprocess. That is deliberate: it reuses the exact
code path already validated on the Mac, and it hands the CUDA/MPS caching
allocator back to the OS between tasks instead of letting it accumulate across a
multi-hour run.

  uv run python -m scripts.train_all_tasks --tasks 0,1,2,3 --device cuda \
      --num-workers 12 --n-epochs 50

The manifest is the thing that matters downstream: CP's guarantee assumes one
data-generating process per calibration set, so every rollout must be traceable
to the exact weights that produced it. `checkpoints.json` pins task -> path ->
sha256, and the collection loop stamps that hash into each rollout file.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks", default="0,1,2,3", help="comma-separated task ids, or 'all' for 0-9"
    )
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--num-procs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--manifest", default="checkpoints.json")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="train this many tasks concurrently on one GPU. Each "
        "process needs ~8 GB VRAM and its own DataLoader "
        "workers, so scale --num-workers down accordingly.",
    )
    args = ap.parse_args()

    tasks = (
        list(range(10))
        if args.tasks == "all"
        else [int(x) for x in args.tasks.split(",")]
    )
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs("summaries", exist_ok=True)

    print(f"training tasks {tasks} on device={args.device}")
    print(f"manifest -> {args.manifest}\n")

    if args.parallel > 1:
        print(f"running {args.parallel} tasks concurrently on one GPU")
        print(
            f"  ~{args.parallel * 8} GB VRAM and "
            f"{args.parallel * args.num_workers} DataLoader workers in total"
        )
        print("  (checkpoints are named task<id>_model.pth, so concurrent tasks")
        print("   sharing an experiment dir do not overwrite each other)\n")

    entries, t_start = [], time.perf_counter()
    lock = threading.Lock()
    done = [0]

    def run_task(tid):
        summary_path = f"summaries/task{tid}.json"
        log_path = f"{args.logdir}/train_task{tid}.log"
        cmd = [
            sys.executable,
            "-m",
            "scripts.train_single_task",
            "--task-id",
            str(tid),
            "--n-epochs",
            str(args.n_epochs),
            "--eval-every",
            str(args.eval_every),
            "--n-eval",
            str(args.n_eval),
            "--num-procs",
            str(args.num_procs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--summary-out",
            summary_path,
        ]
        print(f"[start] task {tid} -> {log_path}", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0

        if rc != 0 or not os.path.exists(summary_path):
            entry = {
                "task_id": tid,
                "status": "failed",
                "returncode": rc,
                "log": log_path,
            }
            msg = f"FAILED (rc={rc}) after {dt / 60:.1f} min -- see {log_path}"
        else:
            entry = json.load(open(summary_path))
            entry["status"] = "ok"
            entry["log"] = log_path
            msg = f"ok in {dt / 60:.1f} min  sha256={entry['sha256'][:16]}..."

        # rewrite after every task so a crash still leaves a usable manifest
        with lock:
            entries.append(entry)
            done[0] += 1
            print(f"[{done[0]}/{len(tasks)}] task {tid}: {msg}", flush=True)
            with open(args.manifest, "w") as f:
                json.dump(
                    {
                        "tasks": sorted(entries, key=lambda e: e["task_id"]),
                        "seed": args.seed,
                        "n_epochs": args.n_epochs,
                        "device": args.device,
                    },
                    f,
                    indent=2,
                )
        return entry

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        list(pool.map(run_task, tasks))
    entries.sort(key=lambda e: e["task_id"])

    total = time.perf_counter() - t_start
    ok = [e for e in entries if e.get("status") == "ok"]
    print("\n" + "=" * 66)
    print(f"  {len(ok)}/{len(tasks)} tasks trained in {total / 3600:.2f} h")
    print("=" * 66)
    for e in entries:
        if e.get("status") != "ok":
            print(f"  task {e['task_id']}: FAILED  ({e['log']})")
        else:
            print(
                f"  task {e['task_id']}: {e['sha256'][:16]}...  {e['wall_sec'] / 60:6.1f} min  "
                f"{os.path.basename(os.path.dirname(e['checkpoint']))}"
            )
    print(f"\nmanifest: {args.manifest}")
    print("Next: verify success rates land in 40-80% before collecting at scale:")
    for e in ok:
        print(
            f"  uv run python -m scripts.eval_checkpoint --checkpoint {e['checkpoint']} "
            f"--task-id {e['task_id']} --n-eval 20"
        )


if __name__ == "__main__":
    main()
