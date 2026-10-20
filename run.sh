#!/usr/bin/env bash
# Everything you can run, in one place.  ./run.sh  with no arguments lists it.
#
# Why one dispatcher instead of a dozen .sh files: every command needs the same
# three lines of setup (the right venv, the MPS fallback, a check that the
# calibration artefact exists). Duplicating that into separate scripts means
# fixing it in a dozen places when it changes.

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv-lerobot/bin/python
ROLLOUTS=data/rollouts/task0_v10.hdf5
ART=data/results/demo_task0.json
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Every checkpoint this project uses is already in ~/.cache/huggingface/hub.
# Without these, huggingface_hub sends a HEAD request per load to check for a
# newer revision, and offline that costs 1+2+4+8 = 15s of retry backoff before
# it falls back to the cache it already had. Offline mode skips the round trip
# entirely -- and it also pins you to the exact cached weights, which is what
# the conformal guarantee assumes anyway.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

c()  { printf '\033[1;36m%s\033[0m\n' "$*"; }          # heading
dim(){ printf '\033[2m%s\033[0m\n' "$*"; }             # the command being run
die(){ printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

[ -x "$PY" ] || die "[abort] $PY not found. Run scripts/setup/bootstrap.sh first."

run() { dim "\$ $*"; echo; "$@"; }

need_artefact() {
  [ -f "$ART" ] || die "[abort] $ART missing. Recreate it with:
  $PY -m scripts.cp_monitor --rollouts $ROLLOUTS \\
      --alpha 0.1 --cal-frac 0.8 --alpha-esc 0.3 --beta 0.1 --out $ART"
}

usage() {
cat <<'EOF'
  ./run.sh <command> [args]

  WATCH A RECORDED ROLLOUT   instant, deterministic, always works
    fail                     the failure case  - VIOLATION at step 40
    safe                     the success case  - stays SAFE throughout
    stall                    the stall case    - STUCK_TIMEOUT at 128
    replay <N>               any rollout by index, e.g. ./run.sh replay 94
    cases                    list every rollout with the step it would alarm
    versus                   flat CP vs per-phase CP on the same failure

  RUN THE SIMULATOR LIVE     real MuJoCo + real policy
    live [INIT]              Diffusion Policy, full 600 steps   (~3 min)
    fast [INIT]              BC-Transformer, ~4x faster, UNCALIBRATED
    arbitrate [INIT]         live with the arbiter panel enabled
    runs <N>                 N episodes back to back, then a summary
    showcase                 4 curated episodes: 3 clean, 1 caught. Stops on alarm
    versus-live              LIVE flat CP vs per-phase CP, 4 episodes, with a tally
    live-fast [INIT]         same demo ~5x faster - UNCALIBRATED (see below)

  ROBOTURK SAWYER          BC-RNN vs ACT arbitration, real recorded human motion
    sawyer                   episode 2 - BC-RNN escalates to ACT
    sawyer-quiet             episode 0 - stays on BC-RNN the whole way
    sawyer-ep <N>            any episode 0-530
    sawyer-scan              which episodes escalate

  TOWER CREATION           closed-loop MuJoCo stacking, Panda
    tower                    the scripted expert - builds the tower
    tower-bcrnn              the learned cheap arm  (does NOT complete the task)
    tower-act                the learned expensive arm (does NOT complete it)
    tower-eval               closed-loop success rate for both learned arms

  OTHER TASKS
    task1 [INIT]             live on task 1  (cream cheese + butter)
    task8 [INIT]             live on task 8  (both moka pots)

  INFO AND MAINTENANCE
    policies                 what policies exist and their MEASURED success
    thresholds               print the calibration artefact
    record                   re-render the three demo mp4s
    test                     run the test suite
    calibrate <TASK> <FRAC>  regenerate an artefact, e.g. ./run.sh calibrate 0 0.8

  In the live window: SPACE pause/resume, -> step once, UP/DOWN speed, Q quit.
EOF
}

cmd="${1:-}"; shift || true
case "$cmd" in

  fail)   need_artefact; run $PY -m scripts.demo_monitor --replay "$ROLLOUTS" -r 83 --artefact "$ART" ;;
  safe)   need_artefact; run $PY -m scripts.demo_monitor --replay "$ROLLOUTS" -r 41 --artefact "$ART" ;;
  stall)  need_artefact; run $PY -m scripts.demo_monitor --replay "$ROLLOUTS" -r 76 --artefact "$ART" ;;
  replay) need_artefact
          [ $# -ge 1 ] || die "usage: ./run.sh replay <rollout index 0-149>"
          run $PY -m scripts.demo_monitor --replay "$ROLLOUTS" -r "$1" --artefact "$ART" ;;
  versus) need_artefact
          c "rollout_0083: per-phase CP catches it at step 40."
          c "Flat CP -- one threshold, 1.415 -- never fires at all."
          run $PY -m scripts.demo_monitor --replay "$ROLLOUTS" -r 83 \
              --compare-flat --artefact "$ART" ;;

  cases)  need_artefact; run $PY -m scripts.demo_monitor --replay "$ROLLOUTS" --list --artefact "$ART" ;;

  live)   need_artefact
          c "Diffusion Policy, 600 steps, ~3 minutes. SPACE pauses."
          run $PY -m scripts.demo_monitor --live --task-id 0 --init "${1:-3}" \
              --policy diffusion_task0 --max-steps 600 --camera frontview --res 512 \
              --artefact "$ART" ;;

  showcase) need_artefact
          c "4 episodes chosen to be representative: 3 the robot completes"
          c "(monitor stays green) and 1 it botches (caught at step 40)."
          c "Each run ENDS the moment the monitor fires."
          run $PY -m scripts.demo_monitor --live --task-id 0 \
              --inits 3,7,83,27 --halt-on any --max-steps 600 \
              --camera frontview --res 512 --artefact "$ART" ;;

  versus-live) need_artefact
          c "Three LIVE episodes with both thresholds drawn. Inits verified"
          c "ON THIS MACHINE: 83 and 20 are caught by per-phase CP and MISSED"
          c "by flat CP; 3 is a clean success both stay silent on."
          c "~3 min per episode."
          run $PY -m scripts.demo_monitor --live --task-id 0 --compare-flat \
              --inits 83,20,3 --max-steps 600 --camera frontview --res 512 \
              --artefact "$ART" ;;

  live-fast) need_artefact
          c "DDPM denoising 100 -> 20 steps: ~5x faster, but a DIFFERENT policy"
          c "than the thresholds were calibrated on, so the run is UNCALIBRATED."
          run $PY -m scripts.demo_monitor --live --task-id 0 --init "${1:-83}" \
              --policy diffusion_task0 --infer-steps 20 --max-steps 600 \
              --camera frontview --res 512 --artefact "$ART" ;;

  fast)   need_artefact
          c "BC-Transformer: ~64 ms/step instead of ~300, so it plays about 4x faster."
          c "UNCALIBRATED - no artefact exists for this policy, so no thresholds are"
          c "shown and no safety claim is made. Use 'live' for the CP demo."
          run $PY -m scripts.demo_monitor --live --task-id 0 --init "${1:-83}" \
              --policy bct_task0 --uncalibrated --max-steps 600 \
              --camera frontview --res 512 --artefact "$ART" ;;

  arbitrate) need_artefact
          c "No expensive arm is trained, so escalation is shown as a counterfactual."
          run $PY -m scripts.demo_monitor --live --task-id 0 --init "${1:-83}" \
              --arbitrate --calm-steps 20 --max-steps 600 \
              --camera frontview --res 512 --artefact "$ART" ;;

  runs)   need_artefact
          [ $# -ge 1 ] || die "usage: ./run.sh runs <N>"
          c "$1 episodes at 600 steps each - about 3 minutes per run."
          c "A sanity check, not a benchmark: the reported 0.687 comes from 150 rollouts."
          run $PY -m scripts.demo_monitor --live --task-id 0 --init 83 \
              --runs "$1" --max-steps 600 --camera frontview --res 512 \
              --artefact "$ART" ;;

  task1)  A=data/results/demo_task1.json; [ -f "$A" ] || die "[abort] $A missing; ./run.sh calibrate 1 0.75"
          run $PY -m scripts.demo_monitor --live --task-id 1 --init "${1:-0}" \
              --policy diffusion_task1 --max-steps 600 --camera frontview --res 512 --artefact "$A" ;;
  task8)  A=data/results/demo_task8.json; [ -f "$A" ] || die "[abort] $A missing; ./run.sh calibrate 8 0.65"
          run $PY -m scripts.demo_monitor --live --task-id 8 --init "${1:-0}" \
              --policy diffusion_task8 --max-steps 600 --camera frontview --res 512 --artefact "$A" ;;

  sawyer)       run $PY -m scripts.roboturk_arena --episode 2 --stride 4 --max-steps 300 ;;
  sawyer-quiet) run $PY -m scripts.roboturk_arena --episode 0 --stride 4 --max-steps 300 ;;
  sawyer-ep)    [ $# -ge 1 ] || die "usage: ./run.sh sawyer-ep <0-530>"
                run $PY -m scripts.roboturk_arena --episode "$1" --stride 4 --max-steps 300 ;;
  sawyer-scan)  c "scanning for episodes where BC-RNN crosses the escalation threshold"
                $PY - <<'EOF'
from scripts.roboturk_arena import load_policies, load_thresholds, episode_stream, RT
th = load_thresholds(); bc, act, st, _ = load_policies("mps")
print(f"  q_esc  closed {th['q_esc']['closed']:.3f}   open {th['q_esc']['open']:.3f}")
for ep in range(14):
    mx = {"closed": 0.0, "open": 0.0}
    for d in episode_stream(RT/"roboturk_20hz.npz", ep, bc, act, st, "mps",
                            stride=8, max_steps=260):
        mx[d["phase"]] = max(mx[d["phase"]], d["s_bc"])
    f = [p for p in ("closed","open") if mx[p] > th["q_esc"][p]]
    print(f"  ep {ep:3d}  closed {mx['closed']:.3f}  open {mx['open']:.3f}"
          + ("   <== ESCALATES" if f else ""))
EOF
                ;;

  tower)       run $PY -m scripts.tower_run --policy expert ;;
  tower-bcrnn) run $PY -m scripts.tower_run --policy bcrnn ;;
  tower-act)   run $PY -m scripts.tower_run --policy act ;;
  tower-eval)  $PY -c "
from scripts.tower_policies import BCRNNPolicy, ACTPolicy, evaluate
for cls, nm, ck in ((BCRNNPolicy,'BC-RNN','data/tower/rel/tower_bcrnn.pt'),
                    (ACTPolicy,'ACT','data/tower/rel/tower_act.pt')):
    r = evaluate(cls(ck), n=10, verbose=False)
    print(f'  {nm:8s} success {r[\"success\"]}/10   mean height {r[\"mean_height\"]:.2f}/3')
" ;;

  policies)   run $PY -m scripts.demo_monitor --list-policies ;;
  thresholds) need_artefact
              $PY -c "
import json; a=json.load(open('$ART'))
print('calibration artefact: $ART')
print(f\"  alpha {a['alpha']}   alpha_esc {a['alpha_esc']}   beta {a['beta']}   cal_frac {a['cal_frac']}\")
print(f\"  policy sha256 {a['policy_sha256'][:16]}...\")
print(f\"  {'phase':11s} {'q_esc':>9s} {'q_safe':>9s} {'d_hat':>8s}\")
for p in ('reach','grasp','transport','release'):
    print(f\"  {p:11s} {a['escalation_thresholds'][p]:9.3f} {a['thresholds'][p]:9.3f} {a['dwell_limits'][p]:8.0f}\")
print()
print('produced by:'); print(' ', a['command'])" ;;

  record) need_artefact
          for R in 41 83 76; do
            c "rendering rollout_00$R"
            MPLBACKEND=Agg $PY -m scripts.demo_monitor --replay "$ROLLOUTS" -r $R \
              --artefact "$ART" --save "data/results/demo_r$R.mp4"
          done
          ls -la data/results/demo_r*.mp4 ;;

  test)   run $PY -m pytest tests/ -q ;;

  calibrate)
          [ $# -ge 2 ] || die "usage: ./run.sh calibrate <task-id> <cal-frac>
  pre-registered: task 0 -> 0.80, task 1 -> 0.75, task 8 -> 0.65"
          run $PY -m scripts.cp_monitor --rollouts "data/rollouts/task$1_v10.hdf5" \
              --alpha 0.1 --cal-frac "$2" --alpha-esc 0.3 --beta 0.1 \
              --out "data/results/demo_task$1.json" ;;

  ""|-h|--help|help) usage ;;
  *) printf '\033[1;31munknown command: %s\033[0m\n\n' "$cmd"; usage; exit 1 ;;
esac
