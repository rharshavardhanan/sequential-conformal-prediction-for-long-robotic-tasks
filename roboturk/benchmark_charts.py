"""Render the Part 5.3 compute-efficiency charts from compute_benchmark.json.

Reads only the JSON -- never re-measures, so the charts and the table can never
disagree. Palette and mark conventions follow the `dataviz` skill; the three
categorical hues used here were validated with its checker (all PASS, light mode).

Light mode only: these are PNG deliverables, so there is no viewer theme to follow.

    .venv-lerobot/bin/python roboturk/benchmark_charts.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BM = HERE / "benchmark"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

PRETTY = {
    "score_max_a0.30_arena": "score / per-episode max\n(arena, α=0.30)",
    "p75_cross_conformal": "p75 cross-conformal",
    "predicted_sigma_a0.10": "predicted σ\n(α=0.10)",
}


def _style(ax, title, xlabel, ylabel):
    """Recessive grid and axes; text in ink tokens, never a series colour."""
    ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left", fontweight="medium")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(BM / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {BM / name}")


def chart_latency(d):
    """Magnitude comparison, two series. Linear y: bar length must be proportional to
    value, and the chunked range is only ~3x, so a log axis would both misencode and
    hide the finding."""
    devs = list(d["devices"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(devs))
    w = 0.34
    bc = [d["devices"][k]["bcrnn"]["throughput_ms"] for k in devs]
    ac = [d["devices"][k]["act"]["throughput_ms"] for k in devs]
    ch = [d["devices"][k]["act"]["chunk"] for k in devs]
    acs = [a / c for a, c in zip(ac, ch)]

    b1 = ax.bar(
        x - w / 2, bc, w, color=BLUE, zorder=3, label="BC-RNN (1.15M) — per step"
    )
    b2 = ax.bar(
        x + w / 2,
        acs,
        w,
        color=ORANGE,
        zorder=3,
        label=f"ACT v2 (9.81M) — per step, chunked ÷{ch[0]}",
    )
    for bars, vals in ((b1, bc), (b2, acs)):
        for r, v in zip(bars, vals):
            ax.annotate(
                f"{v:.3f}",
                (r.get_x() + r.get_width() / 2, v),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x, [k.upper() for k in devs])
    _style(ax, "Per-control-step inference cost", "", "ms per control step")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.text(
        0.01,
        -0.02,
        "Throughput timing (sync once per run), not per-call latency: on MPS a per-call "
        "sync adds ~0.20 ms to BC-RNN and ~0.65 ms to ACT that a real pipelined loop "
        "never pays.",
        fontsize=7.5,
        color=INK2,
        wrap=True,
    )
    _save(fig, "latency.png")


def chart_cost_vs_escalation(d):
    """The actual finding, and it is a curve: cost as a function of escalation fraction."""
    devs = list(d["devices"])
    fig, axes = plt.subplots(1, len(devs), figsize=(11.5, 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, dn in zip(axes, devs):
        m = d["devices"][dn]
        sw = m["cost_sweep"]
        f = np.array(sw["f"]) * 100
        c = m["costs_ms_per_step"]

        ax.plot(
            f,
            sw["arb_chunked"],
            color=BLUE,
            lw=2,
            zorder=4,
            label="arbitrated (ACT chunked)",
        )
        ax.axhline(c["bcrnn_alone"], color=INK2, lw=1.4, ls=":", zorder=3)
        ax.axhline(c["act_alone_chunked"], color=ORANGE, lw=1.4, ls="--", zorder=3)
        ax.annotate(
            "BC-RNN alone",
            (99, c["bcrnn_alone"]),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=INK2,
        )
        ax.annotate(
            "ACT alone (chunked)",
            (99, c["act_alone_chunked"]),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=INK2,
        )

        # all three points sit below 12%, so stack their labels vertically by x order
        pts = sorted(
            m["operating_points"].items(), key=lambda kv: kv[1]["escalation_fraction"]
        )
        for i, (k, v) in enumerate(pts):
            xf = v["escalation_fraction"] * 100
            ax.plot(
                [xf],
                [v["arb_chunked"]],
                "o",
                ms=8,
                color=AQUA,
                mec=SURFACE,
                mew=2,
                zorder=6,
            )
            ax.annotate(
                f"{PRETTY.get(k, k).replace(chr(10), ' ')} — "
                f"{xf:.2f}% → {v['speedup_vs_act_chunked']:.2f}×",
                (xf, v["arb_chunked"]),
                xytext=(40, 62 - 26 * i),
                textcoords="offset points",
                fontsize=7.5,
                color=INK,
                va="center",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"),
                arrowprops=dict(
                    arrowstyle="-", color=INK2, lw=0.7, shrinkA=0, shrinkB=4
                ),
            )
        inverted = c["act_alone_chunked"] < c["bcrnn_alone"]
        _style(
            ax,
            f"{dn.upper()}" + ("  — framing INVERTED" if inverted else ""),
            "escalation fraction (% of control steps on ACT)",
            "ms per control step",
        )
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.suptitle(
        "Arbitrated cost vs escalation fraction",
        color=INK,
        fontsize=13,
        x=0.01,
        ha="left",
        fontweight="medium",
    )
    fig.text(
        0.01,
        -0.03,
        "Where the ACT-alone line sits BELOW BC-RNN-alone (MPS), arbitration is more "
        "expensive than always running ACT at every escalation fraction — BC-RNN is "
        "charged on every step because the monitor needs its score.",
        fontsize=7.5,
        color=INK2,
    )
    _save(fig, "cost_vs_escalation.png")


def chart_efficiency_summary(d):
    """Headline ratio at each real operating point, against a 1.0x break-even line."""
    devs = list(d["devices"])
    keys = list(d["devices"][devs[0]]["operating_points"])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(keys))
    w = 0.34
    cols = [BLUE, ORANGE]
    for i, dn in enumerate(devs):
        op = d["devices"][dn]["operating_points"]
        vals = [op[k]["speedup_vs_act_chunked"] for k in keys]
        bars = ax.bar(
            x + (i - (len(devs) - 1) / 2) * w,
            vals,
            w,
            color=cols[i % 2],
            zorder=3,
            label=dn.upper(),
        )
        for r, v in zip(bars, vals):
            ax.annotate(
                f"{v:.2f}×",
                (r.get_x() + r.get_width() / 2, v),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
                color=INK,
            )
    ax.axhline(1.0, color=INK, lw=1.4, ls="--", zorder=4)
    ax.annotate(
        "1.0× break-even — below this, arbitration COSTS MORE than ACT alone",
        (-0.45, 1.0),
        xytext=(0, 6),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color=INK,
    )
    ax.set_xticks(x, [PRETTY.get(k, k) for k in keys], fontsize=8.5)
    _style(
        ax,
        "Compute efficiency: ACT-alone cost ÷ arbitrated cost",
        "",
        "speedup vs ACT alone (chunked)",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    _save(fig, "efficiency_summary.png")


def main():
    d = json.load(open(BM / "compute_benchmark.json"))
    BM.mkdir(parents=True, exist_ok=True)
    chart_latency(d)
    chart_cost_vs_escalation(d)
    chart_efficiency_summary(d)


if __name__ == "__main__":
    main()
