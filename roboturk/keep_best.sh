#!/bin/bash
# Snapshot roboturk_act.pt whenever the epoch just written is the best val_l1 so far.
# train_act.py rewrites one path each epoch, so without this the run keeps the LAST
# weights rather than the BEST. json.dump of the history happens after torch.save,
# so a history entry for epoch N means N's .pt is fully on disk.
#
# Every snapshot is logged to keep_best.log; only a >=1.5% improvement over the last
# ANNOUNCED best is echoed, so a steadily-descending curve doesn't emit ~40 events.
cd /Users/harshav/PROJECTS/PROJECT_1/roboturk || exit 1
last=""; announced=""
while pgrep -f train_act.py >/dev/null; do
  if [ -f act_loss_history.json ]; then
    read -r ep best_ep best_val <<<"$(python3 - <<'PY'
import json
h = json.load(open("act_loss_history.json"))
b = min(h, key=lambda r: r["val_l1"])
print(h[-1]["epoch"], b["epoch"], f'{b["val_l1"]:.4f}')
PY
)"
    if [ "$ep" = "$best_ep" ] && [ "$ep" != "$last" ]; then
      cp roboturk_act.pt roboturk_act_best.pt && last="$ep"
      echo "$(date +%H:%M:%S) epoch $ep val_l1 $best_val" >> keep_best.log
      if [ -z "$announced" ] || awk "BEGIN{exit !($best_val < $announced * 0.985)}"; then
        announced="$best_val"; echo "new best: epoch $ep val_l1 $best_val"
      fi
    fi
  fi
  sleep 20
done
b=$(tail -1 keep_best.log 2>/dev/null)
echo "training ended. best checkpoint: ${b:-none}"
