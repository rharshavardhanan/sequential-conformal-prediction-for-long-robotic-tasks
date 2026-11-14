"""Single-screen experiment viewer for the sequential-CP safety monitor.

  # replay a recorded rollout (always works, deterministic, labelled [REPLAY])
  .venv-lerobot/bin/python -m scripts.demo_monitor \
      --replay data/rollouts/task0_v10.hdf5 -r 83 \
      --artefact data/results/demo_task0.json

  # name the demo cases by index
  .venv-lerobot/bin/python -m scripts.demo_monitor \
      --replay data/rollouts/task0_v10.hdf5 --list \
      --artefact data/results/demo_task0.json

  # drive it from the live simulator (~4.4x slower than real time, stochastic)
  .venv-lerobot/bin/python -m scripts.demo_monitor --live --task-id 0 --init 83 \
      --artefact data/results/demo_task0.json

Both sources yield the identical tuple, so the viewer cannot tell them apart and
the two paths cannot drift into two different demos.
"""

import argparse
import json

import h5py
import numpy as np
from matplotlib.patches import Patch

from scripts.monitor import SafetyMonitor
from scripts.subphases import PHASES, segment


def replay_source(path, rollout_index):
    """Yield (frame, action, eef_before, eef_after) from a recorded rollout."""
    name = f"rollout_{rollout_index:04d}"
    with h5py.File(path, "r") as f:
        if name not in f:
            raise SystemExit(f"[abort] {name} not in {path}")
        g = f[name]
        frames = g["images"]["agentview"][:]
        actions, eef, ex = g["actions"][:], g["eef_pos"][:], g["executed_pos"][:]
    for t in range(len(actions)):
        yield frames[t], actions[t], eef[t], ex[t]


# Upstream names tasks 3 and 8 with a typo ("libeo"), so the obvious pattern
# f"libero10_task{id}_diffusion" 404s on exactly those two. Same map as
# collect_rollouts.py's --repo default, with the typo encoded rather than
# rediscovered.
def task_repo(task_id):
    """HF repo holding the frozen DiffusionPolicy checkpoint for a task."""
    stem = "libeo10" if task_id in (3, 8) else "libero10"
    return f"yzembodied/{stem}_task{task_id}_diffusion"


def live_source(task_id, init_index, max_steps=600, device="mps", monitor=None,
                camera="frontview", res=384, repo=None, policy="diffusion_task0",
                adapter=None, on_step=None, infer_steps=None):
    """Yield the same tuple from a live MuJoCo rollout under the frozen policy.

    ~300 ms/step on MPS -- about 6x slower than the 20 Hz control rate -- and
    stochastic: a rollout that failed during collection may not fail again.

    `camera` defaults to frontview, not the agentview used during collection:
    agentview is framed tight on the workspace and cuts the arm off at the top.
    frontview shows the whole arm, the table, every object and the basket. The
    policy is unaffected -- it always consumes agentview and the wrist camera,
    which are rendered regardless; this choice only changes what YOU see.
    """
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv


    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

    pool = np.load(f"data/init_states/task{task_id}_init_160.npy")
    from scripts.policies import registry as R

    runner = adapter if adapter is not None else R.build(policy, device=device)
    if infer_steps:
        # Fewer denoising steps = a different sampler = a different policy. The
        # thresholds were calibrated on rollouts generated with the default, so
        # this trades the guarantee for speed. Caller marks the run uncalibrated.
        inner = getattr(runner, "_runner", runner)
        if hasattr(inner, "policy"):
            inner.policy.config.num_inference_steps = int(infer_steps)
            inner.policy.diffusion.num_inference_steps = int(infer_steps)
            print(f"[speed] DDPM denoising steps -> {infer_steps} "
                  f"(default 100). This is NOT the calibrated policy.")
    if monitor is not None and getattr(runner, "_check_policy", True):
        # Same guard the replay path applies. A mismatched checkpoint raises
        # nothing on its own -- the thresholds just stop meaning what they claim.
        monitor.check_policy(runner.fingerprint())
    # POLICY CAMERAS MUST STAY AT 256. dp_policy._img() does no resizing -- it
    # feeds the env's raw frames straight into the ResNet18 spatial-softmax head,
    # which was trained at the 256 px collection used. Rendering agentview at any
    # other size both crashes the head and changes the data-generating
    # distribution the thresholds were calibrated against. Only the extra display
    # camera is free to be larger, via robosuite's per-camera size lists.
    POLICY_RES = 256
    cams, heights = ["agentview", "robot0_eye_in_hand"], [POLICY_RES, POLICY_RES]
    if camera in cams:
        if res != POLICY_RES:
            print(f"[note] --camera {camera} is a policy input; forcing "
                  f"--res {POLICY_RES} so the policy sees what it was calibrated on")
        res = POLICY_RES
    else:
        cams.append(camera)
        heights.append(res)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=list(heights),
                             camera_widths=list(heights), camera_names=cams)
    try:
        # EXACTLY collect_rollouts.py's sequence, in the same order. The policy
        # is a stochastic sampler, so seeding after the reset and the settle
        # steps -- as this used to -- leaves the RNG in a different state at the
        # first inference and produces a DIFFERENT rollout from the same init
        # state. Measured: init 79 failed in the recording and succeeded live,
        # so the demo silently lost the case it was chosen to show.
        seed = 20260805 + task_id * 100000 + init_index   # BASE_SEED, D.7
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset()
        obs = env.set_init_state(pool[init_index])
        for _ in range(5):  # settle, as in collection
            obs, _, _, _ = env.step(np.zeros(7))
        runner.reset()

        done, steps = False, 0
        while steps < max_steps and not done:
            before = obs["robot0_eef_pos"].copy()
            frame = obs[f"{camera}_image"][::-1].copy()
            out = runner.act(obs)
            action = out.action if hasattr(out, "action") else out
            obs, _, done, _ = env.step(action)
            yield frame, action, before, obs["robot0_eef_pos"].copy()
            steps += 1
        if on_step is not None:
            on_step({"success": bool(done), "steps": steps})
    finally:
        env.close()


def list_cases(path, artefact_path):
    """Name candidate demo rollouts with the step each would first alarm."""
    mon_a = json.load(open(artefact_path))
    th = np.array([mon_a["thresholds"][p] for p in PHASES])
    out = []
    with h5py.File(path, "r") as f:
        for k in sorted(x for x in f if x.startswith("rollout_")
                        and f[x].attrs.get("complete")):
            g = f[k]
            labels, segs = segment(g["actions"][:, 6])
            score, mask = g["err_norm"][:], g["motion_mask"][:]
            hit = (score > th[labels]) & mask
            out.append({
                "name": k,
                "success": bool(g.attrs["success"]),
                "steps": int(g.attrs["steps"]),
                "alarm_step": int(np.argmax(hit)) if hit.any() else None,
                "phases": sorted({p for _, _, p in segs}, key=PHASES.index),
            })
    return out


# ----------------------------------------------------------------------------
# viewer -- renders StatePackets and nothing else. It never computes a score.
# ----------------------------------------------------------------------------

PHASE_COLOR = {"reach": "#e8f0fe", "grasp": "#fde8e8",
               "transport": "#e8f6ec", "release": "#fdf3e0"}
HOT, OKG = "#c62828", "#2e7d32"
VERDICT_COLOR = {"SAFE": "#2e7d32", "ESCALATE": "#ef6c00",
                 "IN ESCALATION BAND": "#ef6c00",
                 "VIOLATION": "#c62828", "STUCK_TIMEOUT": "#6a1b9a"}
MAX_DELTA = 0.05 * np.sqrt(3)  # largest commandable step, metres (86.6 mm)
# Bar full-scale for the tracking panel. MAX_DELTA is the theoretical maximum and
# is almost never approached: measured over all 150 task-0 rollouts, commanded
# motion is p50 18.9 mm, p90 40.5 mm. Scaling bars to 86.6 mm makes a typical
# command a 22% sliver. 40 mm keeps the common case legible; 10.8% of steps clip,
# and the numeric value is always printed beside the bar.
BAR_FULL_SCALE = 0.040


def _build_figure(artefact, title):
    """Four regions: robot, decision panel, live signal, event log."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title("Sequential CP safety monitor")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3.6, 1.9, 0.7],
                  width_ratios=[1.0, 1.25], hspace=0.34, wspace=0.06,
                  left=0.055, right=0.98, top=0.92, bottom=0.06)

    ax_robot = fig.add_subplot(gs[0, 0]); ax_robot.axis("off")
    ax_panel = fig.add_subplot(gs[0, 1]); ax_panel.axis("off")
    ax_sig = fig.add_subplot(gs[1, :])
    ax_log = fig.add_subplot(gs[2, :]); ax_log.axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold", ha="left", x=0.02)
    ax_sig.set_xlabel("timestep"); ax_sig.set_ylabel("err_norm")
    ax_sig.grid(alpha=0.25, linewidth=0.5)
    return fig, ax_robot, ax_panel, ax_sig, ax_log


def _redraw_signal(ax, packets, artefact, xmax, cache):
    """Score, stepped thresholds, shaded escalation band, phase bands, mask gaps.

    PERFORMANCE: this runs once per control step, so it must not rebuild the
    axes. The naive version cleared the axis and re-created every artist plus the
    legend each frame; cost grew with timestep and the window visibly stuttered
    past ~200 steps.

    What actually changes per frame is small:
      * the score line grows by one point            -> set_data
      * q_safe / q_esc are PIECEWISE CONSTANT per phase, so they only move at a
        phase boundary                               -> set_data (cheap) 
      * closed phase bands never change again        -> added once, incrementally
      * only the CURRENT phase band grows            -> one patch recreated
      * the legend never changes                     -> built once

    The band matters visually: q_esc sits only 3-5% below q_safe for
    reach/grasp/transport, so two dashed lines would coincide. Shading turns it
    into three legible regions -- below / inside / above.
    """
    if not packets:
        return

    calibrated = cache.get("calibrated", True)
    t = np.array([p.t for p in packets])
    score = np.array([p.score for p in packets])
    scorable = np.array([p.scorable for p in packets])
    safe = np.array([p.q_safe for p in packets])
    esc = np.array([p.q_esc for p in packets])

    if "score" not in cache:
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlabel("timestep")
        ax.set_ylabel("err_norm")
        cache["bands"] = []        # closed phase spans, drawn once each
        cache["cur"] = None        # the growing span for the current phase
        cache["fill"] = None
        cache["vline"] = None
        cache["dot"] = None
        cache["nseg"] = 0
        cache["safe"], = ax.step([], [], where="post", color="#c62828",
                                 linestyle="--", linewidth=1.8, zorder=3,
                                 label=r"$\hat{q}_{safe}$ (per phase)")
        cache["esc"], = ax.step([], [], where="post", color="#ef6c00",
                                linestyle=":", linewidth=1.5, zorder=3,
                                label=r"$\hat{q}_{esc}$")
        cache["score"], = ax.plot([], [], color="#1a237e", linewidth=1.8,
                                  zorder=4, label="s(t)")
        if cache.get("flat") is not None:
            ax.axhline(cache["flat"], color="#6a1b9a", linestyle="-.", linewidth=1.8,
                       zorder=3, label="Flat CP (one threshold)")
        cache["masked"] = ax.scatter([], [], s=10, color="#9e9e9e", zorder=4,
                                     label="below motion floor (not scored)")
        h, lab = ax.get_legend_handles_labels()
        if cache.get("arb_on"):
            h = [Patch(facecolor="#f0a202", alpha=0.30)] + h
            lab = ["escalation band"] + lab
        else:
            # q_esc is not plotted without an arbiter, so it must not appear in
            # the legend either -- a key for an invisible line is just a puzzle.
            keep = [i for i, x in enumerate(lab) if "esc" not in x]
            h, lab = [h[i] for i in keep], [lab[i] for i in keep]
        ax.legend(h, lab,
                  bbox_to_anchor=(0, 1.01, 1, 0.12), loc="lower left", ncol=5,
                  mode="expand", fontsize=8, frameon=False, borderaxespad=0)

    # --- phase bands: append closed segments, redraw only the open one --------
    bounds, st = [], 0
    for i in range(1, len(packets) + 1):
        if i == len(packets) or packets[i].phase != packets[st].phase:
            bounds.append((packets[st].t, packets[i - 1].t + 1, packets[st].phase))
            st = i
    for lo, hi, ph in bounds[cache["nseg"]:len(bounds) - 1]:
        cache["bands"].append(
            ax.axvspan(lo, hi, color=PHASE_COLOR[ph], zorder=0))
    cache["nseg"] = max(cache["nseg"], len(bounds) - 1)
    if cache["cur"] is not None:
        cache["cur"].remove()
    lo, hi, ph = bounds[-1]
    cache["cur"] = ax.axvspan(lo, hi, color=PHASE_COLOR[ph], zorder=0)

    # --- lines: pure data updates -------------------------------------------
    show_esc = cache.get("arb_on", False)
    if calibrated:
        cache["safe"].set_data(t, safe)
        cache["esc"].set_data(t, esc if show_esc else np.full_like(esc, np.nan))
    cache["score"].set_data(t, np.where(scorable, score, np.nan))
    if (~scorable).any():
        cache["masked"].set_offsets(np.c_[t[~scorable], score[~scorable]])

    # --- escalation band: one PolyCollection, recreated (cannot set_data) ----
    if cache["fill"] is not None:
        cache["fill"].remove()
        cache["fill"] = None
    if calibrated and show_esc:
        cache["fill"] = ax.fill_between(t, esc, safe, step="post",
                                        color="#f0a202", alpha=0.30, zorder=1)

    fired = [p for p in packets if p.violated]
    if fired and cache["vline"] is None:
        cache["vline"] = ax.axvline(fired[0].t, color="#c62828", linewidth=2,
                                    zorder=5)
        cache["dot"] = ax.scatter([fired[0].t], [fired[0].score], s=70,
                                  color="#c62828", zorder=6)

    ax.set_xlim(0, max(120, min(xmax, packets[-1].t + 60)))
    lo_y = (min(score.min(), esc.min()) if calibrated else score.min()) - 0.05
    hi_y = (max(score.max(), safe.max()) if calibrated else score.max()) + 0.05
    ax.set_ylim(lo_y, hi_y)


def _redraw_panel(ax, p, artefact, meta):
    """Only quantities the algorithm consumes. No confidence, no probabilities."""
    from matplotlib.patches import Rectangle

    ax.clear(); ax.axis("off")
    calibrated = meta.get("calibrated", True)
    verdict = p.verdict if calibrated else "UNCALIBRATED"
    # "ESCALATE" names an ACTION. With a single policy arm and no arbiter there
    # is nowhere to escalate to, so nothing escalated -- the score merely entered
    # the band between q_esc and q_safe. Headlining a verb for an event that did
    # not occur is exactly the kind of overclaim the rest of this screen avoids,
    # and a small "counterfactual" note in the log does not undo a banner.
    # q_esc exists to TRIGGER A SWITCH. With one arm there is no switch, so
    # crossing it changes nothing that happens and nothing a viewer can act on --
    # it is only a second alarming-looking state on runs that are perfectly fine.
    # The escalation threshold belongs in `--arbitrate`, where it does something.
    arb_on = meta.get("arbiter") is not None
    if verdict == "ESCALATE" and not arb_on:
        verdict = "SAFE"
    ax.add_patch(Rectangle((0.0, 0.88), 1.0, 0.11, transform=ax.transAxes,
                           color=VERDICT_COLOR[verdict] if calibrated else "#9e9e9e",
                           zorder=1))
    ax.text(0.5, 0.945, verdict, transform=ax.transAxes, ha="center",
            va="center", fontsize=20, fontweight="bold", color="white", zorder=2)

    if not calibrated:
        # Every threshold on file belongs to a different policy. Print the raw
        # score and nothing derived from a comparison against it.
        ax.text(0.02, 0.855, "thresholds belong to another policy - none shown",
                transform=ax.transAxes, fontsize=9, color="#555", style="italic")
        rows = [("phase", p.phase),
                ("policy", f"{meta.get('policy_display','?')[:14]} "
                           f"{meta['policy_sha256'][:8]}"),
                ("step", f"{p.t} / {meta['max_steps']}"),
                ("s(t)", f"{p.score:.3f}" +
                         ("" if p.scorable else "  (below motion floor)")),
                ("q_esc", "-- not calibrated"),
                ("q_safe", "-- not calibrated"),
                ("dwell", f"{p.dwell} steps in {p.phase}")]
        y = 0.75
        for k, v in rows:
            ax.text(0.02, y, k, transform=ax.transAxes, fontsize=11, color="#555")
            ax.text(0.24, y, str(v)[:30], transform=ax.transAxes, fontsize=10,
                    family="monospace",
                    color="#9e9e9e" if "not calibrated" in str(v) else "#000")
            y -= 0.052
        ax.text(0.02, y - 0.02, "To get thresholds for this policy:",
                transform=ax.transAxes, fontsize=9, color="#555")
        ax.text(0.02, y - 0.06, "collect_rollouts --suffix _bct  then  cp_monitor",
                transform=ax.transAxes, fontsize=9, family="monospace", color="#777")
        return

    # The banner is the LATCHED, episode-level claim; the rows underneath are
    # the INSTANTANEOUS state. Without this line the two look contradictory:
    # a red VIOLATION banner sitting above a score comfortably inside the band,
    # with nothing saying the violation happened 25 steps and one phase ago.
    if verdict == "SAFE":
        sub_txt = "no threshold crossed so far this episode"
    else:
        when = dict(meta.get("latched_at", {}))
        # Same reason as the banner: with no arbiter nothing escalated, so the
        # latch is named for the fact (a threshold was crossed), not the action.
        if meta.get("arbiter") is None:
            when.pop("ESCALATE", None)
        bits = [f"{k} @ step {v}" for k, v in when.items() if v is not None]
        sub_txt = "LATCHED  " + ",  ".join(bits) if bits else "latched earlier"
    ax.text(0.02, 0.855, sub_txt[:52], transform=ax.transAxes, ha="left",
            fontsize=9, color="#555", style="italic")

    # "IN BAND" here meant "within normal limits", while the banner uses "band"
    # for the ESCALATION zone between q_esc and q_safe. Two meanings of one word
    # on adjacent lines read as a contradiction, so this says which side of which
    # threshold the current step sits on, in the same vocabulary as the banner.
    if not p.scorable:
        live = "not scored (below motion floor)"
    elif not arb_on:
        live = "NORMAL" if p.score <= p.q_safe else "ABOVE q_safe"
    else:
        live = ("NORMAL  (below q_esc)" if p.score <= p.q_esc
                else "IN THE BAND  (q_esc to q_safe)" if p.score <= p.q_safe
                else "ABOVE q_safe")
    ax.text(0.02, 0.812, f"RIGHT NOW: {live}", transform=ax.transAxes, ha="left",
            fontsize=10, fontweight="bold", family="monospace",
            color="#2e7d32" if live.startswith("NORMAL")
            else "#9e9e9e" if live.startswith("not scored") else "#c62828")

    flat_q = meta.get("flat_threshold")
    if flat_q is not None:
        # The whole argument in one row: the SAME score against a per-phase
        # ceiling and against a single global one. Where they disagree is
        # exactly what phase conditioning buys.
        meta.setdefault("_flat_hit", [False])
        if p.scorable and p.score > flat_q:
            meta["_flat_hit"][0] = True
        sub_v = "VIOLATION" if p.violated else "SAFE"
        flat_v = "VIOLATION" if meta["_flat_hit"][0] else "SAFE"
        # Stacked in the LEFT column: x >= 0.5 is TRACKING/MOTION territory.
        for row, (lab, val) in enumerate(((" SUBTASK CP  (per phase)", sub_v),
                                          (" FLAT CP     (one threshold)", flat_v))):
            yy = 0.785 - row * 0.042
            ax.text(0.02, yy, lab, transform=ax.transAxes, fontsize=10,
                    color="#555", family="monospace", fontweight="bold")
            ax.text(0.40, yy, val, transform=ax.transAxes, fontsize=11,
                    family="monospace", fontweight="bold",
                    color=HOT if val == "VIOLATION" else OKG)

    rows = [
        ("phase", p.phase),
        ("policy", f"{meta.get('policy_display','?')[:14]} "
                    f"{meta['policy_sha256'][:8]}"),
        ("step", f"{p.t} / {meta['max_steps']}"),
        ("s(t)", f"{p.score:.3f}" + ("" if p.scorable else "   (below motion floor)")),
        *((("q_esc", f"{p.q_esc:.3f}"),) if arb_on else ()),
        ("q_safe", f"{p.q_safe:.3f}"),
        *((("flat q", f"{flat_q:.3f}"),) if flat_q is not None else ()),
        ("", (f"{p.score:.3f} {'<' if p.score < p.q_esc else '>'} {p.q_esc:.3f} "
              f"{'<' if p.score < p.q_safe else '>'} {p.q_safe:.3f}") if arb_on
             else (f"{p.score:.3f}  {'OVER' if p.score > p.q_safe else 'under'}  "
                   f"{p.q_safe:.3f}")),
        ("dwell", f"{p.dwell} / {p.dwell_limit:.0f} steps"
                  + ("   OVER LIMIT" if p.dwell > p.dwell_limit else "")),
    ]
    y = 0.75 if flat_q is None else 0.68
    for k, v in rows:
        over = k == "dwell" and p.dwell > p.dwell_limit
        ax.text(0.02, y, k, transform=ax.transAxes, fontsize=11, color="#555")
        ax.text(0.24, y, str(v)[:24], transform=ax.transAxes, fontsize=10,
                fontweight="bold" if k in ("phase", "s(t)") or over else "normal",
                color="#6a1b9a" if over else "#000",
                family="monospace")
        y -= 0.052

    y -= 0.012
    ax.text(0.02, y, "ALL THRESHOLDS", transform=ax.transAxes, fontsize=9,
            color="#555", fontweight="bold"); y -= 0.045
    for ph in PHASES:
        marker = "> " if ph == p.phase else "  "
        ax.text(0.02, y, f"{marker}{ph:10s} {artefact['thresholds'][ph]:.3f}",
                transform=ax.transAxes, fontsize=9, family="monospace",
                fontweight="bold" if ph == p.phase else "normal",
                color="#000" if ph == p.phase else "#777")
        y -= 0.042


def _draw_tracking(ax, p, y0):
    """Commanded vs realised displacement as two bars on a shared scale.

    This is what makes the score legible without the formula: the policy asked
    for this much motion, the arm delivered this much, and the residual is
    normalised by the request. Both bars are metres; neither is invented.
    """
    ax.text(0.56, y0, "TRACKING", transform=ax.transAxes, fontsize=10,
            color="#555", fontweight="bold")
    ax.text(0.74, y0, "bars 0-40mm", transform=ax.transAxes, fontsize=8,
            color="#999")
    for j, (label, val, color) in enumerate(
            [("commanded", p.cmd_norm, "#1a237e"), ("actual", p.realised, "#00897b")]):
        y = y0 - 0.05 * (j + 1)
        ax.text(0.56, y, label, transform=ax.transAxes, fontsize=9, color="#555")
        ax.barh(y + 0.012, 0.22 * min(val / BAR_FULL_SCALE, 1.0), height=0.026,
                left=0.72, transform=ax.transAxes, color=color, zorder=3)
        ax.text(0.94, y, f"{val*1000:5.1f}mm", transform=ax.transAxes,
                fontsize=9, family="monospace")


def _draw_motion(ax, p, y0):
    """Motion floor as a filled bar. A stationary arm has a LOW deviation score,
    which is exactly why stall detection is a separate mechanism.

    MOVING describes the CURRENT step; the stall flag is a LATCH from an earlier
    one. Rendering them side by side reads as a contradiction, so the latch gets
    its own line with the step it fired on.
    """
    frac = min(p.cmd_norm / BAR_FULL_SCALE, 1.0)
    filled = int(round(frac * 14))
    ax.text(0.56, y0, "MOTION", transform=ax.transAxes, fontsize=10,
            color="#555", fontweight="bold")
    ax.text(0.56, y0 - 0.05, "|" + "#" * filled + "." * (14 - filled) + "|",
            transform=ax.transAxes, fontsize=10, family="monospace")
    state = "MOVING" if p.scorable else "BELOW FLOOR"
    ax.text(0.88, y0 - 0.05, state, transform=ax.transAxes, fontsize=10,
            family="monospace", fontweight="bold",
            color="#2e7d32" if p.scorable else "#9e9e9e")
    if p.stalled:
        ax.text(0.56, y0 - 0.11, f"STALL LATCHED  dwell {p.dwell} > "
                                 f"d_hat {p.dwell_limit:.0f}",
                transform=ax.transAxes, fontsize=10, family="monospace",
                fontweight="bold", color="#6a1b9a")


def _draw_policy_stack(ax, y0, entries, active_name, decision):
    """The policy stack, drawn from REGISTRY metadata only.

    `success` is whatever was actually measured; an entry with success=None
    prints "not measured" rather than a placeholder. This is what stops the GUI
    inventing a number for a policy that has never been trained.
    """
    ax.text(0.56, y0, "POLICY STACK", transform=ax.transAxes, fontsize=10,
            color="#555", fontweight="bold")
    y = y0 - 0.05
    for e in entries:
        active = e.name == active_name
        if not e.available:
            dot, colour, state = "o", "#9e9e9e", "UNAVAILABLE"
        elif active:
            dot, colour, state = "*", "#2e7d32", "ACTIVE"
        else:
            dot, colour, state = "o", "#777", "STANDBY"
        succ = (f"{e.success:.0%} n={e.success_n}" if e.success is not None
                else "not measured")
        ax.text(0.56, y, f"{dot} {e.display_name:15s} {e.compute_class:9s} "
                         f"{succ:13s} {state}",
                transform=ax.transAxes, fontsize=8, family="monospace",
                color=colour, fontweight="bold" if active else "normal")
        y -= 0.038
    if decision is not None and decision.event:
        ax.text(0.56, y, f"  ARBITER  {decision.event}"[:78], transform=ax.transAxes,
                fontsize=9, family="monospace", color="#ef6c00", fontweight="bold")


def _draw_arbitration(ax, y0, decision):
    """Current arbitration state. Absent an expensive arm this says so."""
    ax.text(0.56, y0, "ARBITRATION", transform=ax.transAxes, fontsize=10,
            color="#555", fontweight="bold")
    if decision is None:
        ax.text(0.56, y0 - 0.045, "not enabled (--arbitrate)", transform=ax.transAxes,
                fontsize=9, family="monospace", color="#9e9e9e")
        return
    rows = [("policy", decision.selected_policy),
            ("compute", decision.compute_class.upper()),
            ("reason", decision.reason)]
    y = y0 - 0.05
    for k, v in rows:
        ax.text(0.56, y, k, transform=ax.transAxes, fontsize=9, color="#555")
        ax.text(0.72, y, str(v)[:26], transform=ax.transAxes, fontsize=9,
                family="monospace",
                color="#ef6c00" if k == "compute" and
                      decision.compute_class == "expensive" else "#000")
        y -= 0.038


def _should_halt(p, mode):
    if mode == "violation":
        return p.violated
    if mode == "stuck":
        return p.stalled
    if mode == "any":
        return p.violated or p.stalled
    return False


def run_viewer(source, monitor, title, meta, save=None, speed=1.0,
               hold=True, halt_on="none"):
    """Render packets as they are finalised. Returns the run summary."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    artefact = monitor.a
    fig, ax_robot, ax_panel, ax_sig, ax_log = _build_figure(artefact, title)
    # created on the first frame: replay is 128px agentview, live is 384px
    # frontview, and imshow needs the real shape to set its extent correctly
    state = {"im": None}
    phase_txt = ax_robot.set_title("", fontsize=18, fontweight="bold")
    packets, log = [], []
    sig_cache = {"calibrated": meta.get("calibrated", True),
                 "arb_on": meta.get("arbiter") is not None,
                 "flat": meta.get("flat_threshold")}

    def update(item):
        frame, action, before, after = item
        if state["im"] is None:
            state["im"] = ax_robot.imshow(frame)
        else:
            state["im"].set_data(frame)
        new = monitor.step(action, before, after)
        if not new:
            return
        packets.extend(new)
        p = packets[-1]
        phase_txt.set_text(p.phase.upper())
        for q in new:
            for flag, key in (("violated", "VIOLATION"), ("stalled", "STUCK"),
                              ("escalated", "ESCALATE")):
                if getattr(q, flag) and key not in meta.setdefault("latched_at", {}):
                    meta["latched_at"][key] = q.t
        decision = None
        if meta.get("arbiter") is not None:
            decision = meta["arbiter"].step(p)
            if decision.event:
                log.append(f"{p.t:4d}  ARBITER     {decision.event}")
        _redraw_panel(ax_panel, p, artefact, meta)
        _draw_tracking(ax_panel, p, 0.79)
        _draw_motion(ax_panel, p, 0.58)
        # Arbitration and the policy stack are only drawn when arbitration is
        # actually running. Showing "ARBITRATION: not enabled" above a list of
        # policies that are all on STANDBY invites the viewer to ask which arm
        # is doing what, when the answer is "one arm, no switching" -- screen
        # space spent raising a question the demo does not answer.
        if meta.get("arbiter") is not None:
            _draw_arbitration(ax_panel, 0.42, decision)
            if meta.get("stack"):
                _draw_policy_stack(ax_panel, 0.22, meta["stack"],
                                   decision.selected_policy if decision
                                   else meta.get("policy_name"), decision)
        _redraw_signal(ax_sig, packets, artefact, meta["max_steps"], sig_cache)
        for q in new:
            if q.event and not (meta.get("arbiter") is None
                                and "q_esc CROSSED" in q.event):
                log.append(q.event)
        ax_log.clear(); ax_log.axis("off")
        for i, line in enumerate(log[-4:]):
            ax_log.text(0.01, 0.82 - i * 0.24, line, transform=ax_log.transAxes,
                        fontsize=10, family="monospace")

    if save and halt_on != "none":
        # FuncAnimation pulls frames on its own schedule, so the interactive
        # loop's break has no equivalent here. Say so rather than silently
        # ignoring the flag.
        print("[note] --halt-on has no effect with --save; the recording runs "
              "to completion. It works in the live window.")
    if save:
        anim = FuncAnimation(fig, update, frames=source,
                             interval=max(1, int(50 / speed)), repeat=False,
                             cache_frame_data=False, save_count=10000)
        anim.save(save, writer="ffmpeg", fps=max(1, int(20 * speed)), dpi=100)
        print(f"wrote {save}")
        plt.close(fig)
    else:
        # Interactive: drive the loop directly rather than through FuncAnimation.
        # A live source is a generator backed by a simulator stepping at ~300 ms;
        # FuncAnimation's timer assumes it can pull frames on demand and stutters
        # against that. A plain loop advances exactly as fast as the sim does.
        plt.ion()
        fig.show()
        ctl = {"paused": False, "once": False, "quit": False, "delay": 0.001}

        def on_key(ev):
            if ev.key == " ":
                ctl["paused"] = not ctl["paused"]
            elif ev.key in ("right", "n"):
                ctl["once"] = True           # advance one step while paused
            elif ev.key == "up":
                ctl["delay"] = max(0.001, ctl["delay"] / 2)
            elif ev.key == "down":
                ctl["delay"] = min(1.0, max(0.002, ctl["delay"] * 2))
            elif ev.key in ("q", "escape"):
                ctl["quit"] = True

        fig.canvas.mpl_connect("key_press_event", on_key)
        help_txt = fig.text(0.5, 0.012,
                            "SPACE pause/resume     -> step once     "
                            "UP/DOWN speed     Q quit",
                            ha="center", fontsize=9, family="monospace",
                            color="#777")

        for item in source:
            # Pausing stops pulling from the source, so MuJoCo pauses too --
            # the simulation itself halts rather than running on invisibly.
            while ctl["paused"] and not ctl["once"] and not ctl["quit"]:
                if not plt.fignum_exists(fig.number):
                    break
                help_txt.set_text("PAUSED    SPACE resume     -> step once     "
                                  "UP/DOWN speed     Q quit")
                plt.pause(0.05)
            if ctl["quit"] or not plt.fignum_exists(fig.number):
                print("[stopped early]")
                break
            ctl["once"] = False
            help_txt.set_text("SPACE pause/resume     -> step once     "
                              "UP/DOWN speed     Q quit")
            update(item)
            fig.canvas.draw_idle()
            plt.pause(ctl["delay"])
            if packets and _should_halt(packets[-1], halt_on):
                print(f"[halt] {packets[-1].verdict} at step {packets[-1].t} "
                      f"-- ending this episode")
                plt.pause(1.2)      # hold the frame so the moment is visible
                break
        plt.ioff()

    packets.extend(monitor.flush())
    summary = _summary(packets, meta)
    if not save and plt.fignum_exists(fig.number):
        if hold:
            print("\nepisode finished -- close the window to exit")
            plt.show()
        else:
            # In --runs N mode a blocking show() would stall the whole
            # experiment after run 1 until the window was closed by hand, and
            # every run would leak its figure.
            plt.pause(0.8)
    plt.close(fig)
    return summary


def _summary(packets, meta):
    """Reports what exists. Expensive-policy fraction is OMITTED, not zeroed --
    there is no second arm to measure."""
    if not packets:
        return {}
    score = np.array([p.score for p in packets])
    scorable = np.array([p.scorable for p in packets])
    viol = [p for p in packets if p.violated]
    esc = [p for p in packets if p.escalated]
    stall = [p for p in packets if p.stalled]
    calibrated = meta.get("calibrated", True)
    flat_q = meta.get("flat_threshold")
    s = {
        "steps": len(packets),
        "steps_masked": int((~scorable).sum()),
        "max_err_norm": float(score[scorable].max()) if scorable.any() else None,
        "policy": f"{meta.get('policy_display', '?')} "
                  f"{meta['policy_sha256'][:8]}",
    }
    if flat_q is not None:
        # The same score against a single global threshold -- the baseline the
        # per-phase thresholds are meant to beat. Recorded per run so a
        # multi-run demo can COUNT the disagreements rather than assert them.
        hits = [q.t for q in packets if q.scorable and q.score > flat_q]
        s["flat_threshold"] = float(flat_q)
        s["flat_violation_step"] = hits[0] if hits else None
        s["flat_verdict"] = "VIOLATION" if hits else "SAFE"
    if calibrated:
        # Every field below is a comparison against a threshold. With an
        # artefact calibrated on a different policy they would be arithmetic
        # without meaning, so they are omitted rather than printed.
        s.update({
            "verdict": packets[-1].verdict,
            "violation_step": viol[0].t if viol else None,
            "escalation_step": esc[0].t if esc else None,
            "stall_step": stall[0].t if stall else None,
            "steps_above_q_safe": float(np.mean(
                [p.score > p.q_safe for p in packets if p.scorable])),
            "steps_above_q_esc": float(np.mean(
                [p.score > p.q_esc for p in packets if p.scorable])),
        })
    else:
        s["verdict"] = "UNCALIBRATED"
        s["note"] = ("no artefact for this policy -- score reported, "
                     "no threshold comparison made")
    print("\n" + "=" * 62)
    print("  RUN SUMMARY")
    print("=" * 62)
    for k, v in s.items():
        print(f"  {k:22s} {v}")
    if s["escalation_step"] is not None:
        print("  escalation is COUNTERFACTUAL: no second policy arm exists yet")
    print("=" * 62)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artefact", default="data/results/demo_task0.json")
    ap.add_argument("--replay", default=None, help="path to a collected task*.hdf5")
    ap.add_argument("-r", "--rollout", type=int, default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="name candidate demo cases")
    ap.add_argument("--save", default=None, help="write an mp4 instead of showing")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--camera", default="frontview",
                    choices=["frontview", "agentview", "sideview", "birdview"],
                    help="live mode only: which view to display. frontview shows "
                         "the whole arm; agentview (used during collection) crops it")
    ap.add_argument("--compare-flat", action="store_true",
                    help="also draw the FLAT CP baseline -- one threshold for the "
                         "whole episode -- and report both verdicts. Shows what "
                         "per-phase conditioning buys.")
    ap.add_argument("--halt-on", default="none",
                    choices=["none", "violation", "stuck", "any"],
                    help="end the episode the moment the monitor fires. The "
                         "default runs to completion, which is what you want for "
                         "measuring detection rate (you need the ground truth); "
                         "halting is what you want for a demo.")
    ap.add_argument("--infer-steps", type=int, default=None,
                    help="DDPM denoising steps. Default (100) is what the "
                         "calibration rollouts used. Lowering it is 4-8x faster "
                         "but produces a DIFFERENT policy, so the thresholds no "
                         "longer describe it -- the run is marked UNCALIBRATED.")
    ap.add_argument("--res", type=int, default=384,
                    help="live mode only: display render size in pixels")
    ap.add_argument("--max-steps", type=int, default=600,
                    help="live mode only: stop the episode early, for a short demo")
    ap.add_argument("--policy", default="diffusion_task0",
                    help="registry name of the policy to run in live mode. "
                         "See --list-policies.")
    ap.add_argument("--list-policies", action="store_true",
                    help="print the registry and exit")
    ap.add_argument("--arbitrate", action="store_true",
                    help="enable cheap/expensive arbitration. Needs --expensive.")
    ap.add_argument("--expensive", default=None,
                    help="registry name of the expensive arm. Without it the "
                         "arbiter reports escalation as a counterfactual.")
    ap.add_argument("--calm-steps", type=int, default=20,
                    help="consecutive in-band steps before de-escalating")
    ap.add_argument("--uncalibrated", action="store_true",
                    help="run a policy that has NO calibration artefact. The CP "
                         "thresholds are hidden and no safety claim is made -- "
                         "you see the robot and the raw score only. Required for "
                         "any policy the artefact was not calibrated on.")
    ap.add_argument("--inits", default=None,
                    help="comma-separated init states to run in order, instead "
                         "of consecutive ones from --init. Lets a demo show a "
                         "representative mix rather than whatever comes next.")
    ap.add_argument("--runs", type=int, default=1,
                    help="run the episode N times and print an aggregate. "
                         "A sanity check, NOT a statistically strong benchmark.")
    ap.add_argument("--repo", default=None,
                    help="live mode only: HF repo of the policy to run. Defaults to "
                         "this task's frozen checkpoint. A different policy will be "
                         "REFUSED unless --artefact was calibrated on it -- see "
                         "check_policy(). Must be a lerobot DiffusionPolicy; DiT and "
                         "smolvla checkpoints need their own runner class.")
    args = ap.parse_args()

    if args.list_policies:
        from scripts.policies import registry as R
        print(f"{'name':20s} {'display':17s} {'class':10s} {'avail':6s} "
              f"{'measured success':18s} note")
        for e in R.list_policies():
            succ = (f"{e.success:.3f} (n={e.success_n})"
                    if e.success is not None else "not measured")
            print(f"{e.name:20s} {e.display_name:17s} {e.compute_class:10s} "
                  f"{str(e.available):6s} {succ:18s} {e.note}")
        return

    monitor = SafetyMonitor.from_file(args.artefact)

    if args.list:
        if not args.replay:
            raise SystemExit("[abort] --list needs --replay <hdf5>")
        rows = list_cases(args.replay, args.artefact)
        print(f"{'rollout':>14} {'outcome':>8} {'steps':>6} {'alarm':>6}  phases")
        for c in rows:
            print(f"{c['name']:>14} {'SUCCESS' if c['success'] else 'fail':>8} "
                  f"{c['steps']:>6} {str(c['alarm_step']):>6}  {','.join(c['phases'])}")
        return

    if args.live:
        from scripts.arbiter import Arbiter
        from scripts.policies import registry as R

        entry = R.get(args.policy)
        adapter = entry.build(device="mps")
        sha = adapter.fingerprint()

        # The guarantee is tied to the policy that generated the calibration
        # set. A different checkpoint is a different data-generating process, so
        # the thresholds simply stop meaning what they claim -- and nothing in
        # the stack would raise. Either the hashes match, or the run is
        # explicitly marked uncalibrated and shows no thresholds at all.
        hash_match = sha == monitor.a["policy_sha256"]
        calibrated = hash_match
        if args.infer_steps:
            # Two different failures, and only one of them is a hash mismatch.
            # The WEIGHTS are unchanged here, so the hash guard passes -- but
            # fewer denoising steps is a different sampler, hence a different
            # data-generating process. A hash cannot see that, so it is marked
            # explicitly and the run proceeds WITHOUT threshold claims. Aborting
            # would be wrong: nothing is inconsistent, the thresholds simply do
            # not describe this sampler.
            calibrated = False
            print("[note] --infer-steps changes the sampler, so the calibrated "
                  "thresholds no longer describe this policy. Running with no "
                  "safety claim; the weights themselves are unchanged.")
        if not hash_match and not args.uncalibrated:
            raise SystemExit(
                f"[abort] {args.policy} (sha {sha[:8]}) was not the policy this "
                f"artefact was calibrated on (sha {monitor.a['policy_sha256'][:8]}).\n"
                f"  Conformal thresholds do not transfer across policies.\n"
                f"  Either:\n"
                f"    1. calibrate for this policy --\n"
                f"         .venv-lerobot/bin/python -m scripts.collect_rollouts \\\n"
                f"             --task-id {args.task_id} --n-rollouts 150 "
                f"--suffix _{args.policy}\n"
                f"         .venv-lerobot/bin/python -m scripts.cp_monitor \\\n"
                f"             --rollouts data/rollouts/task{args.task_id}_{args.policy}.hdf5 \\\n"
                f"             --alpha 0.1 --cal-frac 0.8 --alpha-esc 0.3 --beta 0.1 \\\n"
                f"             --out data/results/demo_{args.policy}.json\n"
                f"    2. or pass --uncalibrated to watch it drive with NO safety claim.")

        arbiter = None
        if args.arbitrate:
            arbiter = Arbiter(cheap_policy=args.policy,
                              expensive_policy=args.expensive,
                              calm_steps=args.calm_steps)
            if args.expensive is None:
                print("[note] --arbitrate without --expensive: escalation is "
                      "reported as a counterfactual; no policy switch occurs.")
            elif not R.get(args.expensive).available:
                raise SystemExit(
                    f"[abort] --expensive {args.expensive} has no checkpoint "
                    f"({R.get(args.expensive).checkpoint}). Naming an untrained "
                    f"policy as the expensive arm would show a switch that cannot "
                    f"happen. Omit --expensive to see the counterfactual instead.")

        if args.compare_flat and monitor.a.get("flat_threshold") is None:
            raise SystemExit("[abort] this artefact has no flat_threshold; "
                             "regenerate it with scripts.cp_monitor")
        meta = {"flat_threshold": (monitor.a.get("flat_threshold")
                                   if args.compare_flat else None),
                "policy_sha256": sha, "max_steps": args.max_steps,
                "policy_display": adapter.display_name,
                "policy_name": args.policy,
                "calibrated": calibrated,
                "arbiter": arbiter,
                "stack": [e for e in R.list_policies(task_id=args.task_id)],
                "adapter": adapter}
        src = live_source(args.task_id, args.init, max_steps=args.max_steps,
                          monitor=(monitor if calibrated else None),
                          camera=args.camera, res=args.res, adapter=adapter,
                          infer_steps=args.infer_steps)
        title = (f"[LIVE] task {args.task_id} - init {args.init} - "
                 f"{adapter.display_name}"
                 + ("" if calibrated else "  [UNCALIBRATED - no safety claim]"))
    else:
        if args.replay is None or args.rollout is None:
            raise SystemExit("[abort] need --replay <hdf5> -r <index>, or --live")
        with h5py.File(args.replay, "r") as f:
            sha = str(f.attrs["policy_sha256"])
            steps = int(f[f"rollout_{args.rollout:04d}"].attrs["steps"])
        monitor.check_policy(sha)
        if args.compare_flat and monitor.a.get("flat_threshold") is None:
            raise SystemExit("[abort] this artefact has no flat_threshold; "
                             "regenerate it with scripts.cp_monitor")
        meta = {"flat_threshold": (monitor.a.get("flat_threshold")
                                   if args.compare_flat else None),
                "policy_sha256": sha, "max_steps": steps,
                "policy_display": "Diffusion Policy",
                "policy_name": "diffusion_task0", "calibrated": True,
                "arbiter": None, "stack": None}
        src = replay_source(args.replay, args.rollout)
        title = f"[REPLAY] task 0 - rollout_{args.rollout:04d}"

    print(f"{title}\nartefact: {args.artefact}")
    print(f"produced by: {monitor.a['command']}")

    if args.runs <= 1 and not args.inits:
        run_viewer(src, monitor, title, meta, save=args.save, speed=args.speed,
               halt_on=args.halt_on)
        return

    if not args.live:
        raise SystemExit("[abort] --runs > 1 needs --live (replay is one fixed rollout)")

    from scripts.experiment import summarize_runs
    from scripts.arbiter import Arbiter

    # An explicit list lets a demo show a representative mix -- consecutive
    # init states are whatever the pool happens to hold next, which can be four
    # clean successes in a row or four failures, neither of which is honest.
    inits = ([int(x) for x in args.inits.split(",")] if args.inits
             else [args.init + i for i in range(args.runs)])
    rows = []
    for i, init_i in enumerate(inits):
        rmon = SafetyMonitor.from_file(args.artefact)
        rmeta = dict(meta)
        if args.arbitrate:
            rmeta["arbiter"] = Arbiter(cheap_policy=args.policy,
                                       expensive_policy=args.expensive,
                                       calm_steps=args.calm_steps)
        adapter = meta["adapter"]
        adapter.reset()
        rsrc = live_source(args.task_id, init_i, max_steps=args.max_steps,
                           monitor=None, camera=args.camera, res=args.res,
                           adapter=adapter)
        rtitle = f"{title}  |  RUN {i + 1}/{len(inits)}  init {init_i}"
        print(f"\n--- run {i + 1}/{len(inits)} (init {init_i}) ---")
        out = run_viewer(rsrc, rmon, rtitle, rmeta, hold=False,
                         halt_on=args.halt_on,
                         save=(args.save.replace(".mp4", f"_run{i+1}.mp4")
                               if args.save else None),
                         speed=args.speed)
        if rmeta.get("arbiter") is not None:
            out.update(rmeta["arbiter"].summary())
        rows.append(out)
    summarize_runs(rows)


if __name__ == "__main__":
    main()
