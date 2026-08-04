"""Progress bar + ETA for a running train_single_task job.

LIBERO prints one line per epoch and nothing in between, so there is no built-in
bar. This parses the log instead -- it never touches the training process, so it
is safe to run (or ctrl-C) at any time.

  uv run python scripts/watch_training.py                 # one shot
  uv run python scripts/watch_training.py -f              # refresh every 30 s
"""

import argparse
import re
import subprocess
import time

TRAIN = re.compile(r"Epoch:\s+(\d+) \| train loss:\s*(-?[\d.]+) \| time:\s*([\d.]+)")
EVAL = re.compile(r"Epoch:\s+(\d+) \| succ:\s*([\d.]+).*?best succ:\s*([\d.]+)")
EVAL_T = re.compile(r"evaluate task \d+ takes ([\d.]+) seconds")


def render(log, total_epochs):
    text = open(log).read()
    trains = TRAIN.findall(text)
    evals = EVAL.findall(text)
    eval_ts = [float(x) for x in EVAL_T.findall(text)]

    running = (
        subprocess.run(
            ["pgrep", "-f", "scripts.train_single_task"], capture_output=True
        ).returncode
        == 0
    )

    # epoch 0 is the zero-shot pass, not a real training epoch
    done = max((int(e) for e, _, _ in trains), default=0)
    times = [float(t) for e, _, t in trains if int(e) > 0]
    # fall back to the profiled numbers until real samples exist, so the ETA
    # is meaningful before the first epoch finishes
    mean_ep = sum(times) / len(times) if times else 5.65
    mean_ev = sum(eval_ts) / len(eval_ts) / 60 if eval_ts else 1.6

    frac = done / total_epochs
    width = 44
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)

    remaining_epochs = total_epochs - done
    remaining_evals = max(0, remaining_epochs // 5)
    eta = remaining_epochs * mean_ep + remaining_evals * mean_ev

    print(f"\n  [{bar}] {frac * 100:5.1f}%   epoch {done}/{total_epochs}")
    print(f"  status      : {'RUNNING' if running else 'STOPPED'}")
    print(f"  epoch time  : {mean_ep:.2f} min  (n={len(times)})")
    print(f"  eval time   : {mean_ev:.2f} min  (n={len(eval_ts)})")
    print(f"  elapsed     : {(sum(times) + sum(eval_ts) / 60) / 60:.2f} h")
    print(f"  ETA         : {eta / 60:.2f} h remaining")

    if trains:
        print(f"  train loss  : {trains[-1][1]}  (epoch {trains[-1][0]})")
    if evals:
        print("\n  success rate @ 20 rollouts")
        for ep, succ, best in evals:
            s = float(succ)
            mark = "  <- in 40-80% band" if 0.40 <= s <= 0.80 else ""
            print(
                f"    epoch {int(ep):3d}: {s:.2f}  ({int(round(s * 20)):2d}/20)"
                f"  best={best}{mark}"
            )
    else:
        print("\n  no eval yet (first at epoch 0, then every 5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/train_task0.log")
    ap.add_argument("--total-epochs", type=int, default=50)
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    while True:
        render(args.log, args.total_epochs)
        if not args.follow:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
