"""Part 5.3 -- compute efficiency for BC-RNN, ACT v2 and arbitration.

Measured back-to-back in ONE session on ONE machine, which Part 5.3 requires and
which only became possible once ACT training finished.

The chunking caveat, stated once here and carried into every output: ACT emits 16
actions per forward. Used as designed its per-step cost is cost_forward/16
("chunked", the headline). scripts/roboturk_arena.py runs a full forward per step
and keeps 1 of the 16 ("per_step"), which is correct for a viewer that needs ACT's
opinion every step but overstates deployed cost by up to 16x.

    .venv-lerobot/bin/python roboturk/compute_benchmark.py
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
OUT = HERE / "benchmark"
sys.path.insert(0, str(HERE))
from train_act import ACT  # noqa: E402

WARMUP, N_TIMED = 50, 500
HISTORY_FALLBACK = 20


class BCRNN(nn.Module):
    """Same shape as the checkpoint's state_dict -- layer names are its keys."""

    def __init__(self, sdim=37, adim=8, hidden=256, layers=2, modes=5):
        super().__init__()
        self.modes, self.adim = modes, adim
        self.enc = nn.Sequential(
            nn.Linear(sdim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.rnn = nn.LSTM(256, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, modes * (2 * adim + 1))

    def forward(self, s, hx=None):
        B, T, _ = s.shape
        o, hx = self.rnn(self.enc(s.reshape(B * T, -1)).reshape(B, T, -1), hx)
        p = self.head(o).reshape(B, T, self.modes, 2 * self.adim + 1)
        return p[..., : self.adim], p[..., -1], hx


def sync(dev):
    """MPS dispatch is async: without this we would time enqueue, not compute."""
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_calls(fn, dev, warmup=WARMUP, n=N_TIMED):
    """Per-call latency in ms. Each call is timed individually so we get a real sd."""
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts[i] = (time.perf_counter() - t0) * 1e3
    return dict(
        mean=float(ts.mean()),
        sd=float(ts.std()),
        n=int(n),
        p50=float(np.percentile(ts, 50)),
        p95=float(np.percentile(ts, 95)),
    )


@torch.no_grad()
def time_throughput(fn, dev, warmup=WARMUP, n=N_TIMED):
    """Amortised per-call cost in a continuously-running loop: sync ONCE at the end.

    This, not per-call latency, is the cost-relevant number. Syncing after every
    call serialises the pipeline and charges a fixed synchronisation tax per call
    -- measured at ~0.18 ms for BC-RNN on MPS, roughly half its apparent per-call
    time. A real control loop enqueues the next op while the GPU works, so that tax
    is not paid. Feeding per-call numbers into an additive cost model over-predicts
    the measured replay by 39% on MPS; feeding throughput numbers predicts it to 2%.

    Per-call latency is still reported separately: it answers "can one inference meet
    a deadline", which is a different question from "what does a step cost".
    """
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) * 1e3 / n


def load_policies(dev):
    bc_raw = torch.load(
        HERE / "roboturk_bcrnn_state_only.pt", map_location="cpu", weights_only=False
    )
    bc = BCRNN().to(dev).eval()
    bc.load_state_dict(bc_raw["model"])

    ck = torch.load(
        HERE / "roboturk_act_best.pt", map_location="cpu", weights_only=False
    )
    c = ck["cfg"]
    act = (
        ACT(
            c["sdim"],
            c["adim"],
            chunk=c["chunk"],
            d=c.get("dim", 512),
            ff=c.get("ff", 3200),
            n_enc=c.get("enc_layers", 4),
            n_dec=c.get("dec_layers", 7),
            n_cvae=c.get("cvae_layers", 4),
            dropout=c.get("dropout", 0.1),
            history=c.get("history", HISTORY_FALLBACK),
        )
        .to(dev)
        .eval()
    )
    act.load_state_dict(ck["model"])
    return bc, act, c


def measure(dev_name):
    dev = torch.device(dev_name)
    bc, act, c = load_policies(dev)
    hist = int(c.get("history", HISTORY_FALLBACK))
    chunk = int(c["chunk"])

    s_bc = torch.zeros(1, 1, c["sdim"], device=dev)  # one LSTM step
    s_act = torch.zeros(1, hist, c["sdim"], device=dev)  # full history window
    hx = None

    def call_bc():
        nonlocal hx
        _, _, hx = bc(s_bc, hx)

    lat_bc = time_calls(call_bc, dev)
    lat_act = time_calls(lambda: act(s_act), dev)
    hx = None
    tput_bc = time_throughput(call_bc, dev)
    tput_act = time_throughput(lambda: act(s_act), dev)
    return dict(
        bcrnn=dict(
            params=sum(p.numel() for p in bc.parameters()),
            latency_ms=lat_bc,
            throughput_ms=tput_bc,
            sync_tax_ms=lat_bc["mean"] - tput_bc,
        ),
        act=dict(
            params=sum(p.numel() for p in act.parameters()),
            latency_ms=lat_act,
            throughput_ms=tput_act,
            sync_tax_ms=lat_act["mean"] - tput_act,
            chunk=chunk,
            history=hist,
        ),
    )


def cost_model(bc_ms, act_ms, chunk):
    """Per-control-step cost of each configuration, in ms.

    BC-RNN is charged on EVERY step including escalated ones: the monitor needs its
    score to decide whether to escalate, so the cheap arm always runs. Arbitration
    adds ACT on top, it does not replace BC-RNN.
    """
    return {
        "bcrnn_alone": bc_ms,
        "act_alone_chunked": act_ms / chunk,
        "act_alone_per_step": act_ms,
        "_arb": lambda f, per_step: (
            bc_ms + f * (act_ms if per_step else act_ms / chunk)
        ),
    }


def operating_points():
    """The three triggers in this repo, which disagree by two orders of magnitude.
    Reported side by side rather than silently choosing one."""
    pts = {}
    try:
        d = json.load(open(HERE / "partD_v2.json"))
        pts["score_max_a0.30_arena"] = d["escalation"]["max"]["0.30"][
            "eval_firing_rate"
        ]
    except (OSError, KeyError):
        pass
    try:
        d = json.load(open(HERE / "partD_corrected.json"))
        pts["predicted_sigma_a0.10"] = d["arbitration"]["0.10"]["act_fraction"]
    except (OSError, KeyError):
        pass
    pts["p75_cross_conformal"] = 0.0200  # measured 2026-09-11, see memory
    return pts


@torch.no_grad()
def direct_replay(dev_name, f, n_steps=2000, seed=0):
    """Time an actual arbitrated loop end to end, then divide by steps.

    BC-RNN every step (the monitor needs its score); ACT on an f-fraction of steps.
    Uses a synthetic escalation mask -- the cost model's correctness depends on HOW
    MANY steps hit each arm, not which.
    """
    dev = torch.device(dev_name)
    bc, act, c = load_policies(dev)
    hist = int(c.get("history", HISTORY_FALLBACK))
    s_bc = torch.zeros(1, 1, c["sdim"], device=dev)
    s_act = torch.zeros(1, hist, c["sdim"], device=dev)
    rng = np.random.default_rng(seed)
    esc = rng.random(n_steps) < f

    hx = None
    for _ in range(WARMUP):  # warm both paths
        _, _, hx = bc(s_bc, hx)
        act(s_act)
    sync(dev)

    hx = None
    t0 = time.perf_counter()
    for i in range(n_steps):
        _, _, hx = bc(s_bc, hx)
        if esc[i]:
            act(s_act)
    sync(dev)
    return (time.perf_counter() - t0) * 1e3 / n_steps, float(esc.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    report = {
        "machine": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "note": "ACT emits `chunk` actions per forward. 'chunked' = cost/chunk (as designed, "
        "HEADLINE). 'per_step' = one forward per step (what roboturk_arena.py does; "
        "an UPPER BOUND on deployed cost, never cite without that label).",
        "warmup": WARMUP,
        "n_timed": N_TIMED,
        "devices": {},
    }

    for dn in devs:
        m = measure(dn)
        # throughput, NOT per-call latency -- see time_throughput.__doc__
        bc_ms = m["bcrnn"]["throughput_ms"]
        act_ms = m["act"]["throughput_ms"]
        chunk = m["act"]["chunk"]
        cm = cost_model(bc_ms, act_ms, chunk)
        arb = cm.pop("_arb")

        pts = operating_points()
        m["costs_ms_per_step"] = cm
        m["operating_points"] = {
            k: {
                "escalation_fraction": f,
                "arb_chunked": arb(f, False),
                "arb_per_step": arb(f, True),
                "speedup_vs_act_chunked": cm["act_alone_chunked"] / arb(f, False),
                "speedup_vs_act_per_step": cm["act_alone_per_step"] / arb(f, True),
            }
            for k, f in pts.items()
        }
        fs = np.linspace(0, 1, 101)
        m["cost_sweep"] = {
            "f": fs.tolist(),
            "arb_chunked": [arb(f, False) for f in fs],
            "arb_per_step": [arb(f, True) for f in fs],
        }

        f_chk = 0.10
        direct, f_act = direct_replay(dn, f_chk)
        est = arb(f_act, True)  # per_step: the replay runs a full ACT forward
        rel = abs(direct - est) / est
        m["estimate_vs_direct"] = dict(
            f_requested=f_chk,
            f_actual=f_act,
            estimate_ms_per_step=est,
            direct_ms_per_step=direct,
            rel_error=rel,
            agrees=bool(rel < 0.25),
        )

        report["devices"][dn] = m
        print(f"\n=== {dn} ===")
        print(
            f"  BC-RNN {bc_ms:8.3f} ms/step  ({m['bcrnn']['params'] / 1e6:.2f} M params)"
            f"   [latency {m['bcrnn']['latency_ms']['mean']:.3f}, "
            f"sync tax {m['bcrnn']['sync_tax_ms']:+.3f}]"
        )
        print(
            f"  ACT    {act_ms:8.3f} ms/fwd   ({m['act']['params'] / 1e6:.2f} M params, "
            f"chunk {chunk}) -> {act_ms / chunk:.3f} ms/step chunked"
            f"   [latency {m['act']['latency_ms']['mean']:.3f}, "
            f"sync tax {m['act']['sync_tax_ms']:+.3f}]"
        )
        if act_ms / chunk < bc_ms:
            print(
                f"  !! chunked ACT ({act_ms / chunk:.3f}) is CHEAPER per step than "
                f"BC-RNN ({bc_ms:.3f}) on {dn} -- the cheap/expensive framing is "
                f"INVERTED here, so H3 cannot hold at any escalation fraction."
            )
        for k, v in m["operating_points"].items():
            print(
                f"  f={v['escalation_fraction']:7.2%}  {k:<26} "
                f"arb {v['arb_chunked']:7.3f} ms/step  "
                f"speedup vs ACT-alone {v['speedup_vs_act_chunked']:5.2f}x"
            )
        print(
            f"  cross-check f={f_act:.2%}: estimate {est:.3f} vs direct {direct:.3f} "
            f"ms/step  rel err {rel:.1%}  {'AGREES' if rel < 0.25 else 'DISAGREES'}"
        )

    json.dump(report, open(OUT / "compute_benchmark.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'compute_benchmark.json'}")


if __name__ == "__main__":
    main()
