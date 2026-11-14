"""RoboTurk Sawyer arbitration viewer -- BC-RNN (cheap) vs ACT (expensive).

WHAT IS REAL HERE, AND WHAT IS NOT
----------------------------------
Real: the Sawyer, its motion, both policies, both thresholds, every number shown.
      The arm is a robosuite Sawyer posed frame-by-frame from the joint angles a
      human actually commanded during a RoboTurk teleoperation session.
Not simulated: physics. MuJoCo is used as a RENDERER (`sim.forward()`), not as a
      dynamics engine.

Why not closed-loop: RoboTurk Tower Creation is a REAL-ROBOT dataset with no
MuJoCo environment, and the policies are state-only -- all 37 input dims are
proprioception, with zero object channels and no images. A policy that cannot
perceive a block cannot stack one. Driving a simulated scene with it would show
an arm replaying memorised motion past blocks it cannot see, which would look
convincing and mean nothing. So the arm shows what the human did, and the
policies are scored on how well they predict the human's next command -- which
is exactly what partD_v2.json calibrated.

  score s(t) = || a_pred(t) - a_demo(t) ||_2   over the 7 continuous joint targets
  phase      = gripper closed / open           (K=2, from the grasp bool)
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent.parent
RT = HERE / "roboturk"
sys.path.insert(0, str(RT))

GRIP = 7  # action column 7 is the grasp bool, kept out of the score
SDIM, ADIM = 37, 8
JOINT_SLICE = slice(8, 15)  # measured arm joint positions inside the 37-dim state


# ----------------------------------------------------------------- policies --
class BCRNN(torch.nn.Module):
    """Layer names are load-bearing: they are the checkpoint's state_dict keys."""

    def __init__(self, sdim=SDIM, adim=ADIM, hidden=256, layers=2, modes=5):
        super().__init__()
        self.modes, self.adim = modes, adim
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(sdim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.rnn = torch.nn.LSTM(256, hidden, layers, batch_first=True)
        self.head = torch.nn.Linear(hidden, modes * (2 * adim + 1))

    def forward(self, s, hx=None):
        B, T, _ = s.shape
        z = self.enc(s.reshape(B * T, -1)).reshape(B, T, -1)
        o, hx = self.rnn(z, hx)
        p = self.head(o).reshape(B, T, self.modes, 2 * self.adim + 1)
        return (
            p[..., : self.adim],
            p[..., self.adim : 2 * self.adim].clamp(-5, 2),
            p[..., -1],
            hx,
        )


def gmm_mean(mu, logit):
    return (torch.softmax(logit, -1).unsqueeze(-1) * mu).sum(-2)


def load_policies(
    device, act_ckpt="roboturk_act_best.pt", bc_ckpt="roboturk_bcrnn_state_only.pt"
):
    """Both arms plus the normalisation stats they were trained with."""
    from train_act import ACT

    dev = torch.device(device)
    bc_raw = torch.load(RT / bc_ckpt, map_location="cpu", weights_only=False)
    bc = BCRNN().to(dev)
    bc.load_state_dict(bc_raw["model"])
    bc.eval()

    ck = torch.load(RT / act_ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    act = ACT(
        c["sdim"],
        c["adim"],
        chunk=c["chunk"],
        d=c.get("dim", 512),
        ff=c.get("ff", 3200),
        n_enc=c.get("enc_layers", 4),
        n_dec=c.get("dec_layers", 7),
        n_cvae=c.get("cvae_layers", 4),
        dropout=c.get("dropout", 0.1),
        history=c.get("history", 1),
    ).to(dev)
    act.load_state_dict(ck["model"])
    act.eval()

    st = {
        k: np.asarray(bc_raw[k])
        for k in ("s_mu", "s_sd", "a_mu", "a_sd")
        if k in bc_raw
    }
    if not st:
        st = {k: np.asarray(ck[k]) for k in ("s_mu", "s_sd", "a_mu", "a_sd")}
    meta = {
        "act_epoch": ck.get("epoch"),
        "act_params": c.get("params"),
        "act_history": c.get("history", 1),
        "act_chunk": c.get("chunk"),
    }
    return bc, act, st, meta


def load_compute_costs(device, path=None):
    """Measured per-step costs from benchmark/compute_benchmark.json, or None.

    ATTRIBUTED cost, not this process's wall clock. The arena deliberately runs
    BOTH arms every step so it can draw the comparison, so its own wall clock is
    not what an arbitrated system would spend. What is displayed instead is the
    cost the arbitrated system WOULD have incurred given the arm the arbiter
    selected, priced from the benchmark's measured per-step numbers.

    ACT is priced chunked (cost/chunk), which is how it would actually be
    deployed -- one forward emits `chunk` actions. The arena itself re-runs ACT
    every step and keeps 1 of 16, an upper bound that is not the deployed cost.

    Returns None if the benchmark has not been run; the panel then hides itself
    rather than inventing numbers, so the arena still runs standalone.
    """
    p = path or RT / "benchmark" / "compute_benchmark.json"
    try:
        d = json.load(open(p))["devices"][str(device)]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    chunk = d["act"]["chunk"]
    return {
        "bc_ms": d["bcrnn"]["throughput_ms"],
        "act_ms": d["act"]["throughput_ms"] / chunk,
        "chunk": chunk,
        "device": str(device),
    }


def load_thresholds(path=None):
    """Frozen, from partD_v2.json. Never recomputed here."""
    d = json.load(open(path or RT / "partD_v2.json"))
    esc = d["escalation"]["max"]["0.30"]["per_phase"]
    return {
        "q_safe": {k: d["safety"]["max"][k]["q"] for k in ("closed", "open")},
        "q_esc": {k: esc[k]["q"] for k in ("closed", "open")},
        "alpha": d["alpha"],
        "alpha_esc": 0.30,
        "sha": d["checkpoint_sha256"],
        "score": d["score"],
    }


# --------------------------------------------------------------- the stream --
def episode_stream(npz, ep_index, bc, act, st, device, stride=1, max_steps=None):
    """Yield one dict per displayed timestep. Both arms are evaluated every step
    so either can be shown; the arbiter decides which one COUNTS."""
    z = np.load(npz, allow_pickle=True)
    S, A, E = z["S"], z["A"], z["E"]
    ids = np.unique(E)
    ep = ids[ep_index % len(ids)]
    sel = np.flatnonzero(E == ep)
    if max_steps:
        sel = sel[: max_steps * stride]

    dev = torch.device(device)
    s_mu, s_sd = st["s_mu"], st["s_sd"]
    a_mu, a_sd = st["a_mu"], st["a_sd"]
    Sn = ((S[sel] - s_mu) / s_sd).astype(np.float32)
    Ad = A[sel]
    h = max(1, int(getattr(act, "history", 1) or 1))

    hx = None
    with torch.no_grad():
        for t in range(0, len(sel), stride):
            # BC-RNN is an LSTM: it must be stepped in order, carrying state.
            for u in range(max(0, t - stride + 1) if t else 0, t + 1):
                mu, _, logit, hx = bc(torch.tensor(Sn[u][None, None]).to(dev), hx)
            a_bc = gmm_mean(mu, logit)[0, 0].cpu().numpy() * a_sd + a_mu

            # ACT is stateless over a fixed history window.
            idx = np.clip(np.arange(t - h + 1, t + 1), 0, len(Sn) - 1)
            a_ac = act(torch.tensor(Sn[idx][None]).to(dev))[0][0, 0]
            a_ac = a_ac.cpu().numpy() * a_sd + a_mu

            demo = Ad[t]
            closed = bool(demo[GRIP] > 0.5)
            yield {
                "t": t,
                "joints": S[sel][t, JOINT_SLICE],
                "eef": S[sel][t, 0:3],
                "demo": demo,
                "a_bc": a_bc,
                "a_act": a_ac,
                "s_bc": float(np.linalg.norm(a_bc[:GRIP] - demo[:GRIP])),
                "s_act": float(np.linalg.norm(a_ac[:GRIP] - demo[:GRIP])),
                "phase": "closed" if closed else "open",
                "n_total": len(sel),
            }


# ------------------------------------------------------------------ the arm --
class SawyerView:
    """robosuite Sawyer used as a RENDERER: joint angles are written straight
    into the model and `sim.forward()` resolves kinematics. No dynamics, no
    controller, no contact -- the arm shows the pose the human actually held."""

    def __init__(self, res=420, camera="frontview"):
        import robosuite as rs
        from robosuite.controllers import load_controller_config

        cfg = load_controller_config(default_controller="JOINT_POSITION")
        self.env = rs.make(
            "Lift",
            robots="Sawyer",
            controller_configs=cfg,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            control_freq=20,
            horizon=10**6,
        )
        self.env.reset()
        self.sim = self.env.sim
        self.idx = self.env.robots[0]._ref_joint_pos_indexes
        self.res, self.camera = res, camera

    def render(self, joints):
        self.sim.data.qpos[self.idx] = joints
        self.sim.forward()
        return self.sim.render(
            camera_name=self.camera, width=self.res, height=self.res
        )[::-1]

    def close(self):
        self.env.close()


# ----------------------------------------------------------------- the view --
CHEAP_C, EXP_C = "#1565c0", "#ef6c00"
OK_C, HOT_C = "#2e7d32", "#c62828"


def _panel(ax, d, dec, th, meta):
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.axis("off")
    expensive = dec.compute_class == "expensive"
    col = EXP_C if expensive else CHEAP_C
    name = "ACT" if expensive else "BC-RNN"
    score = d.get("_run_max", d["s_act"] if expensive else d["s_bc"])
    q_e, q_s = th["q_esc"][d["phase"]], th["q_safe"][d["phase"]]

    ax.add_patch(Rectangle((0, 0.90), 1, 0.10, transform=ax.transAxes, color=col))
    ax.text(
        0.5,
        0.95,
        f"{name}   {'EXPENSIVE' if expensive else 'CHEAP'}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="white",
    )

    verdict = "VIOLATION" if score > q_s else "ESCALATE" if score > q_e else "SAFE"
    vcol = HOT_C if verdict == "VIOLATION" else EXP_C if verdict == "ESCALATE" else OK_C
    ax.text(
        0.5,
        0.845,
        verdict,
        transform=ax.transAxes,
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=vcol,
        family="monospace",
    )

    rows = [
        ("gripper", d["phase"].upper()),
        ("s(t) this step", f"{d.get('_step_score', score):.3f}"),
        ("phase max  S_k", f"{d.get('_run_max', score):.3f}"),
        ("escalate at", f"{q_e:.3f}"),
        ("violate at", f"{q_s:.3f}"),
        (
            "",
            f"esc {q_e:.2f}  {'<' if score > q_e else '>'}  S_k {score:.2f}"
            f"  {'>' if score > q_s else '<'}  safe {q_s:.2f}",
        ),
        ("why", dec.reason[:26]),
    ]
    y = 0.78
    for k, v in rows:
        ax.text(0.03, y, k, transform=ax.transAxes, fontsize=12, color="#555")
        ax.text(
            0.46,
            y,
            v,
            transform=ax.transAxes,
            fontsize=13,
            family="monospace",
            fontweight="bold" if k in ("phase max  S_k", "gripper") else "normal",
        )
        y -= 0.050

    # both policies side by side, so the comparison needs no explanation
    y -= 0.01
    ax.text(
        0.03,
        y,
        "BOTH ARMS",
        transform=ax.transAxes,
        fontsize=11,
        color="#555",
        fontweight="bold",
    )
    y -= 0.046
    for nm, sc, c in (("BC-RNN", d["s_bc"], CHEAP_C), ("ACT", d["s_act"], EXP_C)):
        w = 0.40 * min(sc / max(q_s, 1e-6), 1.0)
        ax.text(
            0.03, y, nm, transform=ax.transAxes, fontsize=11, color=c, fontweight="bold"
        )
        ax.barh(
            y + 0.012,
            w,
            height=0.030,
            left=0.30,
            transform=ax.transAxes,
            color=c,
            zorder=3,
        )
        ax.text(
            0.74,
            y,
            f"{sc:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            family="monospace",
        )
        y -= 0.046
    better = "ACT" if d["s_act"] < d["s_bc"] else "BC-RNN"
    ax.text(
        0.03,
        y,
        f"closer to the human: {better}",
        transform=ax.transAxes,
        fontsize=10,
        style="italic",
        color="#555",
    )

    # joint targets: what the human asked for vs what the active policy asked for
    y -= 0.050
    ax.text(
        0.03,
        y,
        "JOINT TARGETS   human | policy",
        transform=ax.transAxes,
        fontsize=11,
        color="#555",
        fontweight="bold",
    )
    y -= 0.052
    pol = d["a_act"] if expensive else d["a_bc"]
    SPAN, MID, LIM = 0.40, 0.52, 3.2  # bar half-width, centre, +/- radians
    for j in range(7):
        ax.text(
            0.03,
            y,
            f"q{j + 1}",
            transform=ax.transAxes,
            fontsize=11,
            family="monospace",
            color="#555",
        )
        # human as a wide grey bar, policy as a bright marker on top of it:
        # any visible gap between them IS the per-joint error.
        h_, p_ = float(d["demo"][j]), float(pol[j])
        for val, c, hgt, z in ((h_, "#b0b0b0", 0.026, 3), (p_, col, 0.011, 4)):
            w = SPAN * np.clip(abs(val) / LIM, 0, 1)
            ax.barh(
                y + 0.012,
                w,
                height=hgt,
                left=MID if val >= 0 else MID - w,
                transform=ax.transAxes,
                color=c,
                zorder=z,
            )
        ax.plot(
            [MID, MID],
            [y - 0.004, y + 0.028],
            transform=ax.transAxes,
            color="#888",
            lw=0.8,
            zorder=5,
        )
        ax.text(
            0.94,
            y,
            f"{h_:+5.2f}",
            transform=ax.transAxes,
            fontsize=10,
            family="monospace",
            color="#333",
        )
        y -= 0.040
    ax.text(
        0.03,
        y,
        "grey = human      colour = policy",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
        color="#777",
    )


def _plot(ax, hist, th, cache):
    """s(t) for the ACTIVE arm, against the two calibrated thresholds.

    Thresholds are piecewise constant per gripper phase, so they move only at a
    grasp/release event -- redrawing them every frame is what made the LIBERO
    viewer stutter, so the artists are cached and only their data is updated.
    """
    if not hist:
        return
    t = np.array([h["t"] for h in hist])
    s = np.array([h["s"] for h in hist])
    qs = np.array([th["q_safe"][h["phase"]] for h in hist])
    qe = np.array([th["q_esc"][h["phase"]] for h in hist])

    if "s" not in cache:
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("timestep", fontsize=11)
        ax.set_ylabel("running phase max  $S_k$", fontsize=11)
        cache["bands"], cache["cur"], cache["fill"] = [], None, None
        cache["nseg"] = 0
        (cache["qs"],) = ax.step(
            [],
            [],
            where="post",
            color=HOT_C,
            ls="--",
            lw=2,
            label="violate at  $\\hat{q}_{safe}$",
        )
        (cache["qe"],) = ax.step(
            [],
            [],
            where="post",
            color=EXP_C,
            ls=":",
            lw=1.8,
            label="escalate at  $\\hat{q}_{esc}$",
        )
        (cache["s"],) = ax.plot(
            [], [], color="#12306e", lw=2, label="$S_k$ = running max of the active arm"
        )
        ax.legend(
            bbox_to_anchor=(0, 1.01, 1, 0.12),
            loc="lower left",
            ncol=3,
            mode="expand",
            fontsize=10,
            frameon=False,
            borderaxespad=0,
        )

    bounds, st = [], 0
    for i in range(1, len(hist) + 1):
        if i == len(hist) or hist[i]["cls"] != hist[st]["cls"]:
            bounds.append((hist[st]["t"], hist[i - 1]["t"] + 1, hist[st]["cls"]))
            st = i
    for lo, hi, cl in bounds[cache["nseg"] : len(bounds) - 1]:
        cache["bands"].append(
            ax.axvspan(
                lo,
                hi,
                color=(EXP_C if cl == "expensive" else CHEAP_C),
                alpha=0.12,
                zorder=0,
            )
        )
    cache["nseg"] = max(cache["nseg"], len(bounds) - 1)
    if cache["cur"] is not None:
        cache["cur"].remove()
    lo, hi, cl = bounds[-1]
    cache["cur"] = ax.axvspan(
        lo, hi, color=(EXP_C if cl == "expensive" else CHEAP_C), alpha=0.12, zorder=0
    )

    cache["qs"].set_data(t, qs)
    cache["qe"].set_data(t, qe)
    cache["s"].set_data(t, s)
    if cache["fill"] is not None:
        cache["fill"].remove()
    cache["fill"] = ax.fill_between(
        t, qe, qs, step="post", color=EXP_C, alpha=0.15, zorder=1
    )
    ax.set_xlim(0, max(60, t[-1] + 40))
    ax.set_ylim(0, max(s.max(), qs.max()) * 1.12)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from scripts.arbiter import Arbiter

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument(
        "--stride",
        type=int,
        default=4,
        help="display every Nth control step (episodes average ~5900)",
    )
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--calm-steps", type=int, default=15)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--res", type=int, default=420)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    th = load_thresholds()
    bc, act, st, meta = load_policies(args.device)
    view = SawyerView(res=args.res)
    arb = Arbiter(
        cheap_policy="BC-RNN", expensive_policy="ACT", calm_steps=args.calm_steps
    )

    fig = plt.figure(figsize=(15.5, 10.0))
    fig.canvas.manager.set_window_title("RoboTurk Sawyer - policy arbitration")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[4.4, 1.5, 0.5],
        width_ratios=[1, 1],
        hspace=0.42,
        wspace=0.05,
        left=0.05,
        right=0.98,
        top=0.90,
        bottom=0.05,
    )
    ax_r = fig.add_subplot(gs[0, 0])
    ax_r.axis("off")
    ax_p = fig.add_subplot(gs[0, 1])
    ax_p.axis("off")
    ax_s = fig.add_subplot(gs[1, :])
    ax_l = fig.add_subplot(gs[2, :])
    ax_l.axis("off")

    hist, log, cache, im = [], [], {}, {"h": None}

    class Pkt:
        __slots__ = (
            "score",
            "q_esc",
            "q_safe",
            "scorable",
            "violated",
            "escalated",
            "stalled",
        )

    stream = episode_stream(
        RT / "roboturk_20hz.npz",
        args.episode,
        bc,
        act,
        st,
        args.device,
        stride=args.stride,
        max_steps=args.max_steps,
    )

    ctl = {"paused": False, "once": False, "quit": False, "delay": 0.001}

    def on_key(e):
        if e.key == " ":
            ctl["paused"] = not ctl["paused"]
        elif e.key in ("right", "n"):
            ctl["once"] = True
        elif e.key == "up":
            ctl["delay"] = max(0.001, ctl["delay"] / 2)
        elif e.key == "down":
            ctl["delay"] = min(1.0, max(0.002, ctl["delay"] * 2))
        elif e.key in ("q", "escape"):
            ctl["quit"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    tip = fig.text(
        0.5,
        0.012,
        "SPACE pause     → step     ↑↓ speed     Q quit",
        ha="center",
        fontsize=10,
        family="monospace",
        color="#777",
    )

    runmax = {"closed": 0.0, "open": 0.0, "phase": None}
    costs = load_compute_costs(args.device)
    # attributed cost accumulator -- see load_compute_costs.__doc__
    comp = {"total": 0.0, "always_act": 0.0, "steps": 0, "recent": []}

    def draw(d):
        p = Pkt()
        active_exp = arb.current == "ACT"
        step_score = d["s_act"] if active_exp else d["s_bc"]

        # partD_v2 calibrated on the per-episode phase MAXIMUM (n_cal = 79
        # episodes), not on pooled per-timestep scores. Comparing an
        # instantaneous score against those thresholds is a category error --
        # they sit ~10x above a typical step and could never fire. The monitor
        # therefore tracks the RUNNING MAXIMUM within the current gripper phase,
        # which is the same statistic the quantile was taken over.
        if d["phase"] != runmax["phase"]:
            runmax["phase"] = d["phase"]
            runmax[d["phase"]] = 0.0
        runmax[d["phase"]] = max(runmax[d["phase"]], step_score)
        p.score = runmax[d["phase"]]
        d["_step_score"], d["_run_max"] = step_score, p.score
        p.q_esc = th["q_esc"][d["phase"]]
        p.q_safe = th["q_safe"][d["phase"]]
        p.scorable, p.violated, p.escalated, p.stalled = True, False, False, False
        dec = arb.step(p)
        if dec.event:
            log.append(f"t={d['t']:5d}   {dec.event}")

        img = view.render(d["joints"])
        if im["h"] is None:
            im["h"] = ax_r.imshow(img)
        else:
            im["h"].set_data(img)
        ax_r.set_title(
            f"gripper {d['phase'].upper()}",
            fontsize=15,
            fontweight="bold",
            color=HOT_C if d["phase"] == "closed" else "#333",
        )

        _panel(ax_p, d, dec, th, meta)
        hist.append(
            {
                "t": d["t"],
                "s": d["_run_max"],
                "step": d["_step_score"],
                "phase": d["phase"],
                "cls": dec.compute_class,
            }
        )
        _plot(ax_s, hist, th, cache)

        ax_l.clear()
        ax_l.axis("off")
        for i, line in enumerate(log[-2:]):
            ax_l.text(
                0.01,
                0.78 - i * 0.42,
                line,
                transform=ax_l.transAxes,
                fontsize=11,
                family="monospace",
                color=EXP_C,
            )
        sm = arb.summary()
        f_exp = sm["expensive_fraction"] or 0.0
        ax_l.barh(
            0.15,
            1 - f_exp,
            height=0.30,
            left=0,
            transform=ax_l.transAxes,
            color=CHEAP_C,
        )
        ax_l.barh(
            0.15,
            f_exp,
            height=0.30,
            left=1 - f_exp,
            transform=ax_l.transAxes,
            color=EXP_C,
        )
        ax_l.text(
            0.01,
            0.15,
            f" BC-RNN {1 - f_exp:.0%}",
            transform=ax_l.transAxes,
            fontsize=11,
            color="white",
            fontweight="bold",
            va="center",
        )
        ax_l.text(
            0.99,
            0.15,
            f"ACT {f_exp:.0%}  switches {sm['switches']} ",
            transform=ax_l.transAxes,
            fontsize=11,
            color="white",
            fontweight="bold",
            va="center",
            ha="right",
        )

        if costs:
            # attributed, NOT this process's wall clock: the arena runs both arms
            # every step to draw the comparison. See load_compute_costs.__doc__.
            step_ms = costs["bc_ms"] + (costs["act_ms"] if active_exp else 0.0)
            comp["total"] += step_ms
            comp["always_act"] += costs["act_ms"]
            comp["steps"] += 1
            comp["recent"].append(step_ms)
            del comp["recent"][:-20]
            roll = sum(comp["recent"]) / len(comp["recent"])
            saved = comp["always_act"] - comp["total"]
            ax_l.text(
                0.5,
                0.78,
                f"compute  {roll:.3f} ms/step (rolling)   ·   "
                f"cumulative {comp['total']:.1f} ms over {comp['steps']} steps   ·   "
                f"vs always-ACT {saved:+.1f} ms "
                f"({saved / comp['always_act']:+.0%})",
                transform=ax_l.transAxes,
                ha="center",
                va="center",
                fontsize=10.5,
                family="monospace",
                color=OK_C if saved > 0 else HOT_C,
            )
            ax_l.text(
                0.5,
                0.52,
                f"attributed from measured {costs['device'].upper()} throughput "
                f"(BC-RNN {costs['bc_ms']:.3f} + ACT {costs['act_ms']:.3f} ms/step "
                f"chunked ÷{costs['chunk']}); BC-RNN runs every step — the monitor "
                f"needs its score",
                transform=ax_l.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#777",
            )

        fig.suptitle(
            f"SAWYER  ·  RoboTurk “create tower”  ·  "
            f"episode {args.episode}  ·  step {d['t']}/{d['n_total']}"
            f"   —  recorded human motion, policies predicting live",
            fontsize=14,
            fontweight="bold",
            ha="left",
            x=0.02,
        )

    if args.save:
        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(
            fig,
            draw,
            frames=stream,
            repeat=False,
            cache_frame_data=False,
            save_count=10**5,
        )
        anim.save(args.save, writer="ffmpeg", fps=20, dpi=100)
        print(f"wrote {args.save}")
    else:
        plt.ion()
        fig.show()
        for d in stream:
            while ctl["paused"] and not ctl["once"] and not ctl["quit"]:
                if not plt.fignum_exists(fig.number):
                    break
                tip.set_text(
                    "PAUSED    SPACE resume     → step     ↑↓ speed     Q quit"
                )
                plt.pause(0.05)
            if ctl["quit"] or not plt.fignum_exists(fig.number):
                break
            ctl["once"] = False
            tip.set_text("SPACE pause     → step     ↑↓ speed     Q quit")
            draw(d)
            fig.canvas.draw_idle()
            plt.pause(ctl["delay"])
        plt.ioff()

    sm = arb.summary()
    print("\n" + "=" * 62)
    print(f"  episode {args.episode}   steps shown {len(hist)}")
    print(f"  switches            {sm['switches']}")
    print(f"  BC-RNN (cheap)      {1 - (sm['expensive_fraction'] or 0):.1%} of steps")
    print(f"  ACT (expensive)     {sm['expensive_fraction'] or 0:.1%} of steps")
    print(
        f"  ACT params          {meta['act_params'] / 1e6:.1f}M   epoch {meta['act_epoch']}"
    )
    print(
        f"  thresholds          partD_v2.json, alpha {th['alpha']} / "
        f"alpha_esc {th['alpha_esc']}"
    )
    if costs and comp["steps"]:
        saved = comp["always_act"] - comp["total"]
        print(
            f"  compute/step        {comp['total'] / comp['steps']:.3f} ms "
            f"(attributed, {costs['device'].upper()})"
        )
        print(
            f"  always-ACT/step     {costs['act_ms']:.3f} ms  "
            f"(chunked / {costs['chunk']})"
        )
        print(
            f"  vs always-ACT       {saved:+.1f} ms over {comp['steps']} steps "
            f"({saved / comp['always_act']:+.1%})"
        )
        if saved < 0:
            print("        arbitration cost MORE than always running ACT here: chunked")
            print("        ACT is cheaper per step than BC-RNN on this device, and")
            print(
                "        BC-RNN runs on every step regardless. See benchmark/README.md."
            )
    print("  NOTE: the arm is recorded human motion. MuJoCo renders it; no")
    print("        physics is simulated and no task success is claimed.")
    print("  NOTE: compute is ATTRIBUTED from measured per-step costs, not this")
    print("        process's wall clock -- the arena runs both arms every step.")
    print("=" * 62)
    view.close()


if __name__ == "__main__":
    main()
