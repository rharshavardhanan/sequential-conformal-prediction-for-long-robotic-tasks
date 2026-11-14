# Compute-Efficiency Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure BC-RNN, ACT v2 and arbitrated inference cost back-to-back on this machine, report cost as a function of escalation fraction with charts, and add a live compute readout to the arena viewer.

**Architecture:** One measurement script (`roboturk/compute_benchmark.py`) writes `roboturk/benchmark/compute_benchmark.json`; one chart script (`roboturk/benchmark_charts.py`) renders PNGs from that JSON; the arena gets an additive readout panel.

**Tech Stack:** `.venv-lerobot/bin/python`, torch 2.10 (MPS + CPU), numpy, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-12-compute-efficiency-benchmark-design.md`

## Global Constraints

- **Environment:** `.venv-lerobot/bin/python` only. Never `uv run`.
- **Quiet machine.** ACT training has finished. Do not start background work while timing; a competing process invalidates Part 5.3's back-to-back requirement.
- **`torch.mps.synchronize()`** before starting and before stopping every MPS timer. Without it you measure enqueue, not compute.
- **Both checkpoints are frozen.** Load, `.eval()`, `torch.no_grad()`. No training, no re-fingerprinting.
- **ACT v2 only** — `roboturk/roboturk_act_best.pt`, 9,805,128 params, chunk 16, history 20. ACT v1 is out of scope.
- **Report chunked ACT as the headline, per-step as the stated upper bound.** Never cite per-step without that label.
- **BC-RNN is charged on every step**, escalated ones included — the monitor needs its score to decide.
- **Cost is per control step**, never per displayed frame. The arena's `--stride` is display-only.
- **Do not modify** the arena's scoring, thresholds or arbiter logic. Readout is additive.
- **Do not commit.** No commits unless asked.

---

## File Structure

| file | responsibility | status |
|---|---|---|
| `roboturk/compute_benchmark.py` | measure latency, build the cost sweep, cross-check estimate vs direct | create |
| `roboturk/benchmark/compute_benchmark.json` | every number | generated |
| `roboturk/benchmark_charts.py` | render PNGs from the JSON | create |
| `roboturk/benchmark/*.png` | the three charts | generated |
| `roboturk/benchmark/README.md` | table, headline, caveats | create (Task 4) |
| `scripts/roboturk_arena.py` | live readout panel | modify, additive only |

---

### Task 1: Latency measurement + cost sweep

**Goal:** `roboturk/benchmark/compute_benchmark.json` holds measured per-call latency for both policies on both devices, plus arbitrated cost as a function of escalation fraction.

**Why:** Every downstream claim is a ratio of these two numbers. If the MPS timing is unsynchronised or the warmup is skipped, the numbers are wrong in a way nothing downstream can detect.

**Files:**
- Create: `roboturk/compute_benchmark.py`
- Read only: `roboturk/roboturk_act_best.pt`, `roboturk/roboturk_bcrnn_state_only.pt`, `roboturk/partD_v2.json`, `roboturk/partD_corrected.json`, `roboturk/cross_conformal.json`

**Interfaces:**
- Consumes: `train_act.ACT(sdim, adim, chunk, d, ff, n_enc, n_dec, n_cvae, dropout, history)`; the arena's `BCRNN` shape (`enc`/`rnn`/`head`, forward returns `(mu, logstd, logit, hx)`).
- Produces: JSON keys `machine`, `policies` (`params`, `latency_ms` per device with `mean`/`sd`/`n`), `act_regimes` (`chunked`/`per_step` per-step cost), `operating_points`, `cost_sweep`, `estimate_vs_direct`.

- [x] **Step 1: Write the script — model loading and the timer**

```python
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
import json, platform, sys, time
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
        self.enc = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU())
        self.rnn = nn.LSTM(256, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, modes * (2 * adim + 1))

    def forward(self, s, hx=None):
        B, T, _ = s.shape
        o, hx = self.rnn(self.enc(s.reshape(B * T, -1)).reshape(B, T, -1), hx)
        p = self.head(o).reshape(B, T, self.modes, 2 * self.adim + 1)
        return p[..., :self.adim], p[..., -1], hx


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
    return dict(mean=float(ts.mean()), sd=float(ts.std()), n=int(n),
                p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)))


def load_policies(dev):
    bc_raw = torch.load(HERE / "roboturk_bcrnn_state_only.pt",
                        map_location="cpu", weights_only=False)
    bc = BCRNN().to(dev).eval()
    bc.load_state_dict(bc_raw["model"])

    ck = torch.load(HERE / "roboturk_act_best.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    act = ACT(c["sdim"], c["adim"], chunk=c["chunk"], d=c.get("dim", 512),
              ff=c.get("ff", 3200), n_enc=c.get("enc_layers", 4),
              n_dec=c.get("dec_layers", 7), n_cvae=c.get("cvae_layers", 4),
              dropout=c.get("dropout", 0.1),
              history=c.get("history", HISTORY_FALLBACK)).to(dev).eval()
    act.load_state_dict(ck["model"])
    return bc, act, c
```

- [x] **Step 2: Add the per-device measurement and the cost model**

```python
def measure(dev_name):
    dev = torch.device(dev_name)
    bc, act, c = load_policies(dev)
    hist = int(c.get("history", HISTORY_FALLBACK))
    chunk = int(c["chunk"])

    s_bc = torch.zeros(1, 1, c["sdim"], device=dev)          # one LSTM step
    s_act = torch.zeros(1, hist, c["sdim"], device=dev)      # full history window
    hx = None

    def call_bc():
        nonlocal hx
        _, _, hx = bc(s_bc, hx)

    lat_bc = time_calls(call_bc, dev)
    lat_act = time_calls(lambda: act(s_act), dev)
    return dict(
        bcrnn=dict(params=sum(p.numel() for p in bc.parameters()), latency_ms=lat_bc),
        act=dict(params=sum(p.numel() for p in act.parameters()), latency_ms=lat_act,
                 chunk=chunk, history=hist),
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
        "_arb": lambda f, per_step: bc_ms + f * (act_ms if per_step else act_ms / chunk),
    }
```

- [x] **Step 3: Add the operating points, the sweep, and `main()`**

```python
def operating_points():
    """The three triggers in this repo, which disagree by two orders of magnitude.
    Reported side by side rather than silently choosing one."""
    pts = {}
    try:
        d = json.load(open(HERE / "partD_v2.json"))
        pts["score_max_a0.30_arena"] = d["escalation"]["max"]["0.30"]["eval_firing_rate"]
    except (OSError, KeyError):
        pass
    try:
        d = json.load(open(HERE / "partD_corrected.json"))
        pts["predicted_sigma_a0.10"] = d["arbitration"]["0.10"]["act_fraction"]
    except (OSError, KeyError):
        pass
    pts["p75_cross_conformal"] = 0.0200      # measured 2026-09-11, see memory
    return pts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    report = {
        "machine": {"platform": platform.platform(), "python": sys.version.split()[0],
                    "torch": torch.__version__},
        "note": "ACT emits `chunk` actions per forward. 'chunked' = cost/chunk (as designed, "
                "HEADLINE). 'per_step' = one forward per step (what roboturk_arena.py does; "
                "an UPPER BOUND on deployed cost, never cite without that label).",
        "warmup": WARMUP, "n_timed": N_TIMED,
        "devices": {},
    }

    for dn in devs:
        m = measure(dn)
        bc_ms = m["bcrnn"]["latency_ms"]["mean"]
        act_ms = m["act"]["latency_ms"]["mean"]
        chunk = m["act"]["chunk"]
        cm = cost_model(bc_ms, act_ms, chunk)
        arb = cm.pop("_arb")

        pts = operating_points()
        m["costs_ms_per_step"] = cm
        m["operating_points"] = {
            k: {"escalation_fraction": f,
                "arb_chunked": arb(f, False), "arb_per_step": arb(f, True),
                "speedup_vs_act_chunked": cm["act_alone_chunked"] / arb(f, False),
                "speedup_vs_act_per_step": cm["act_alone_per_step"] / arb(f, True)}
            for k, f in pts.items()
        }
        fs = np.linspace(0, 1, 101)
        m["cost_sweep"] = {
            "f": fs.tolist(),
            "arb_chunked": [arb(f, False) for f in fs],
            "arb_per_step": [arb(f, True) for f in fs],
        }
        report["devices"][dn] = m
        print(f"\n=== {dn} ===")
        print(f"  BC-RNN {bc_ms:8.3f} ms/step  ({m['bcrnn']['params']/1e6:.2f} M params)")
        print(f"  ACT    {act_ms:8.3f} ms/fwd   ({m['act']['params']/1e6:.2f} M params, "
              f"chunk {chunk}) -> {act_ms/chunk:.3f} ms/step chunked")
        for k, v in m["operating_points"].items():
            print(f"  f={v['escalation_fraction']:7.2%}  {k:<26} "
                  f"arb {v['arb_chunked']:7.3f} ms/step  "
                  f"speedup vs ACT-alone {v['speedup_vs_act_chunked']:5.2f}x")

    json.dump(report, open(OUT / "compute_benchmark.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'compute_benchmark.json'}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run it**

Run: `.venv-lerobot/bin/python roboturk/compute_benchmark.py`

Expected: a `=== cpu ===` and a `=== mps ===` block, each printing BC-RNN and ACT latency with non-zero sd, then three operating-point lines.

**Done when:** `roboturk/benchmark/compute_benchmark.json` exists and

```bash
.venv-lerobot/bin/python -c "
import json; d=json.load(open('roboturk/benchmark/compute_benchmark.json'))
for dn,m in d['devices'].items():
    b=m['bcrnn']['latency_ms']; a=m['act']['latency_ms']
    assert b['sd']>0 and a['sd']>0, dn+': zero sd means the timer never resolved'
    assert a['mean']>b['mean'], dn+': ACT (9.8M) must be slower than BC-RNN (1.15M)'
    print('%-4s bcrnn %.3f+-%.3f ms | act %.3f+-%.3f ms | ratio %.1fx'%(
        dn,b['mean'],b['sd'],a['mean'],a['sd'],a['mean']/b['mean']))
"
```

prints one line per device with ACT slower than BC-RNN and both sd > 0.

---

### Task 2: Estimate vs direct end-to-end cross-check

**Goal:** `compute_benchmark.json` gains an `estimate_vs_direct` block showing that the cost model predicts the measured wall clock of a real arbitrated replay.

**Why:** Part 5.3 asks for the estimate *and* a direct measurement as a check. A cost model that does not predict the measured replay is wrong, and reporting only the model would hide that.

**Files:**
- Modify: `roboturk/compute_benchmark.py`

**Interfaces:**
- Consumes: `cost_model` and `measure` from Task 1.
- Produces: JSON key `estimate_vs_direct` with `f`, `estimate_ms_per_step`, `direct_ms_per_step`, `rel_error`, `agrees`.

- [x] **Step 1: Add the replay**

A synthetic escalation mask at the chosen `f` is used rather than the real per-episode
decisions: this task measures whether the *cost model* is arithmetically right, which
depends only on how many steps land on each arm, not on which ones.

```python
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
    for _ in range(WARMUP):                       # warm both paths
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
```

- [x] **Step 2: Call it from `main()` and record the comparison**

Insert inside the per-device loop, after `m["cost_sweep"]` is set:

```python
        f_chk = 0.10
        direct, f_act = direct_replay(dn, f_chk)
        est = arb(f_act, True)          # per_step: the replay runs a full ACT forward
        rel = abs(direct - est) / est
        m["estimate_vs_direct"] = dict(f_requested=f_chk, f_actual=f_act,
                                       estimate_ms_per_step=est,
                                       direct_ms_per_step=direct,
                                       rel_error=rel, agrees=bool(rel < 0.25))
        print(f"  cross-check f={f_act:.2%}: estimate {est:.3f} vs direct {direct:.3f} "
              f"ms/step  rel err {rel:.1%}  {'AGREES' if rel < 0.25 else 'DISAGREES'}")
```

- [x] **Step 3: Run and read the cross-check**

Run: `.venv-lerobot/bin/python roboturk/compute_benchmark.py`

Expected: a `cross-check` line per device. A 25% tolerance is deliberately loose — the direct
replay includes Python loop overhead the model omits, so exact agreement is not expected;
what is being tested is that the model is not off by an order of magnitude.

**Done when:** the command prints `AGREES` for both devices. If either says `DISAGREES`, stop
and report it — per the benchmark file, a failed check is a result, not something to tune away.

---

### Task 3: Charts

**Goal:** three PNGs in `roboturk/benchmark/` rendered from the JSON.

**Why:** the cost-vs-`f` relationship is the actual finding and it is a curve, not a number; a table cannot show that the three operating points sit in wildly different regimes.

**Files:**
- Create: `roboturk/benchmark_charts.py`

**Interfaces:**
- Consumes: `roboturk/benchmark/compute_benchmark.json`.
- Produces: `latency.png`, `cost_vs_escalation.png`, `efficiency_summary.png`.

- [x] **Step 1: Load the `dataviz` skill**

Per `CLAUDE.md`'s routing table, `dataviz` owns any chart work. Load it and follow its
palette, axis and annotation guidance rather than matplotlib defaults.

- [x] **Step 2: Write the chart script**

Three figures, each reading only the JSON (never re-measuring):

1. `latency.png` — grouped bars, BC-RNN vs ACT, per device, error bars = sd, log y (the two differ by ~an order of magnitude).
2. `cost_vs_escalation.png` — arbitrated ms/step vs escalation fraction `f` over [0, 1], one line per ACT regime, horizontal reference lines for BC-RNN-alone and ACT-alone, and the three operating points marked with labels.
3. `efficiency_summary.png` — `cost_ACT_alone / cost_arbitrated` at each operating point, with a `1.0x` break-even line.

Every figure must carry the chunked/per-step distinction in its labels or caption — a chart
that shows only one regime without saying which is the defect this whole spec exists to avoid.

- [x] **Step 3: Render and inspect**

Run: `.venv-lerobot/bin/python roboturk/benchmark_charts.py`

Then **read each PNG back with the Read tool** and check: axes labelled with units, no
overlapping text, the three operating points legible and distinguishable, the regime stated.

**Done when:** all three PNGs exist, and each has been visually inspected and is legible.

---

### Task 4: Live arena readout

**Goal:** `scripts/roboturk_arena.py` displays rolling per-step compute, the active arm, and cumulative compute saved versus always-ACT.

**Why:** this is the "live results" half of the request, and the number a viewer cares about is the running total, not a static table.

**Files:**
- Modify: `scripts/roboturk_arena.py` (additive only)
- Create: `roboturk/benchmark/README.md`

- [x] **Step 1: Load measured latencies into the arena**

Read `roboturk/benchmark/compute_benchmark.json` at startup for the device in use; fall back
to measuring nothing and hiding the panel if the file is absent, so the arena still runs
standalone.

- [x] **Step 2: Add the attributed-cost accumulator**

The arena evaluates **both** arms every step to draw its comparison, so its raw wall clock is
NOT the arbitrated cost. Accumulate *attributed* cost instead — the cost the arbitrated system
would have incurred given the arm the arbiter selected, from the measured per-call latencies:

```
cost_step = bc_ms + (act_ms_effective if escalated else 0)
saved     = (bc_ms + act_ms_effective) * steps - cost_total
```

- [x] **Step 3: Draw the panel**

Rolling ms/step, active arm, cumulative attributed ms, compute saved vs always-ACT, and a
label naming the ACT regime and the device.

- [x] **Step 4: Verify the arena still runs unchanged otherwise**

Run: `.venv-lerobot/bin/python scripts/roboturk_arena.py --episode 0 --max-steps 40 --save /tmp/arena_check.mp4`

Expected: completes, writes the mp4, and the pre-existing summary block still prints the same
threshold and score lines as before the change.

- [x] **Step 5: Write `roboturk/benchmark/README.md`**

The table, the headline `cost_ACT/cost_arb`, the chunked-vs-per-step caveat, the three
operating points, and the estimate-vs-direct result.

**Done when:** the arena command above completes and writes the mp4, and `README.md` exists
carrying the headline number and the chunking caveat.

---

### Task 5: Update the project docs

**Goal:** `docs/status.md` and `roboturk/RoboTurk_BCRNN_Findings.md` carry the Part 5.3 result.

**Why:** the user asked to "update after"; `docs/status.md` is the canonical status doc that any new session reads first, and a Part 5.3 result that lives only in `roboturk/benchmark/` will not be found.

**Files:**
- Modify: `docs/status.md`, `roboturk/RoboTurk_BCRNN_Findings.md`

- [x] **Step 1: Add a Part 5.3 section to the findings doc** — the latency table, headline ratio, chunking caveat, estimate-vs-direct result.
- [x] **Step 2: Add the headline to `docs/status.md`** with a pointer to `roboturk/benchmark/`.
- [x] **Step 3: Verify** both files mention the chunked/per-step distinction:

```bash
grep -l 'chunk' docs/status.md roboturk/RoboTurk_BCRNN_Findings.md
```

**Done when:** both files are listed by that grep and both carry the headline number.

---

## Interpreting the outcome — decided in advance

- **Arbitration cheaper than ACT-alone at a real operating point** → H3 holds; report the ratio.
- **Arbitration ≈ ACT-alone cost** → at chunked ACT, `cost_ACT/16` may be *below* BC-RNN's cost, in which case arbitration is **more** expensive than ACT-alone and H3 fails. This is a live possibility given BC-RNN is an LSTM stepped one frame at a time while ACT amortises over 16. Report it plainly; it would mean the cheap/expensive framing is inverted for this pair.
- **Estimate and direct disagree by >25%** → the cost model is wrong; stop and report.

## Out of scope

Re-litigating which escalation trigger is correct; ACT v1; FLOP counting; changing arena scoring/threshold/arbiter logic.
