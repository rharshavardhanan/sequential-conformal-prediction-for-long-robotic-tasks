#!/usr/bin/env bash
# Sequential CP safety monitor — live MuJoCo policy arbitration.
#
# Opens the monitor GUI: offscreen MuJoCo render on the left, decision panel on the
# right (banner, thresholds, tracking, motion, arbitration, policy stack), live s(t)
# and cube-to-goal plots below.
#
# Closed loop: every observation comes from MuJoCo physics at the current step and every
# action from a network or the P-servo. Nothing is replayed.
#
# Plain python on purpose. The GUI is matplotlib (macosx backend), NOT the MuJoCo passive
# viewer, so `mjpython` is not required — and the passive viewer could not render the
# panel anyway, because mjvScene primitives cannot draw text, axes or numbers.
#
#   ./run_policy.sh                 all three rescue seeds (400016, 400167, 400169)
#   ./run_policy.sh 400169          a single seed
#   ./run_policy.sh 400169 0.20     seed + alpha_esc
#   ./run_policy.sh --save out.mp4  headless, write a shareable video
#
# Keys:  SPACE pause/resume   RIGHT step once   R reset seed   N next seed   Q quit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(cd "$HERE/.." && pwd)/.venv-lerobot/bin/python"

[[ -x "$PY" ]] || { echo "error: $PY not found"; exit 1; }
for f in results/mujoco_train/push_expert/seed0/best_model.zip \
         results/mujoco_train/cheap_policy_16x16.pt \
         results/mujoco_train/calibration.json \
         results/mujoco_train/arbitration_eval.json; do
  [[ -f "$HERE/$f" ]] || { echo "error: missing $f"; exit 1; }
done

ARGS=()
if [[ "${1:-}" == "--save" ]]; then
  ARGS+=(--save "${2:?usage: ./run_policy.sh --save out.mp4}")
else
  [[ $# -ge 1 ]] && ARGS+=(--seed "$1")
  [[ $# -ge 2 ]] && ARGS+=(--alpha-esc "$2")
fi

cd "$HERE"
PYTHONPATH="$HERE/live" exec "$PY" live/arena_reach_push.py "${ARGS[@]}"
