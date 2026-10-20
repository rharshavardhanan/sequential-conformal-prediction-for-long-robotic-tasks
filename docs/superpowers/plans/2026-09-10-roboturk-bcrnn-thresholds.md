# RoboTurk BC-RNN Threshold Calculation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the safety threshold `q̂_k` and escalation threshold `q̂_k,esc` per phase for the frozen RoboTurk BC-RNN, on a score that excludes the grasp bool, and prove the pipeline correct by reproducing the published 8-dim Part D result from the same code.

**Architecture:** One script, `roboturk/calibrate_bcrnn.py`. It imports the frozen-split loader from `train_act.py`, redefines the `BCRNN` module so the checkpoint's bare `state_dict` has somewhere to load, runs one full-sequence forward per held-out episode on CPU, caches per-timestep scores to an npz, then computes order-statistic thresholds at two aggregations and writes `partD_v2.json`.

**Tech Stack:** Python 3.10 via `.venv-lerobot`, torch 2.10 (CPU), numpy 2.2.6. No MuJoCo, robosuite, LIBERO or lerobot involvement.

**Spec:** `docs/superpowers/specs/2026-09-10-roboturk-bcrnn-thresholds-design.md`

## Global Constraints

- **Environment:** `.venv-lerobot/bin/python` only. Never `uv run` — that is `.venv`, which has numpy <1.24 and cannot load this checkpoint.
- **Device: CPU.** ACT training holds MPS in the background; contending with it risks perturbing a 4-hour run whose timing Part 5.3 depends on.
- **The BC-RNN checkpoint is frozen.** No training, no re-fingerprinting, no gradient anywhere. Every forward is under `@torch.no_grad()`.
- **The split is frozen** at 316/56/79/80 from `np.random.default_rng(0)`. Never redraw it. `load_everything()` already asserts this.
- **Normalisation comes from the checkpoint**, never recomputed: `a_mu[7]=0.0`, `a_sd[7]=1.0` must hold.
- **`α = 0.10`, `K = 2`, `α_k = α/K = 0.05`.** `α_esc ∈ {0.20, 0.30, 0.40}`, primary 0.30, each split the same way (`α_esc/K`).
- **Corrected score:** `‖â[:7] − a[:7]‖₂` in **raw** (denormalised) joint-target space, matching Part D's convention. The grasp bool remains the K=2 phase variable and is excluded only from the magnitude.
- **`partD.json` is never modified.** It is the provenance record of the 8-dim result.
- **No detection-rate, false-alarm, or "catches failures" language** in any output or docstring. RoboTurk has no failure labels; every demo is a success.
- **Do not commit.** The user's global rule: no commits unless asked.

---

## File Structure

| file | responsibility | status |
|---|---|---|
| `roboturk/calibrate_bcrnn.py` | the whole calculation: score, cache, threshold, verify, report | create |
| `roboturk/bcrnn_scores.npz` | per-timestep score cache, keyed by checkpoint SHA | generated |
| `roboturk/partD_v2.json` | the thresholds and every check | generated |
| `roboturk/train_act.py` | `load_everything()`, `GRIP` — imported, **not modified** | unchanged |

No test file. The 8-dim reproduction gate in Task 1 is a strictly stronger test of the
same code path than a unit test of the rank arithmetic would be — see spec §5.

---

### Task 1: Scoring pass, cache, and the reproduction gate

**Goal:** `roboturk/bcrnn_scores.npz` holds per-timestep scores for all 159 held-out episodes, and the 8-dim thresholds recomputed from it reproduce `partD.json`.

**Why:** Every number in Task 2 is a quantile over this cache. If the model reconstruction, the normalisation or the episode slicing is wrong, the thresholds are confidently wrong with no other symptom. Reproducing the published 8-dim result is the only available proof that this code is the code that produced Part D.

**Files:**
- Create: `roboturk/calibrate_bcrnn.py`
- Read only: `roboturk/train_act.py`, `roboturk/roboturk_bcrnn_state_only.pt`, `roboturk/roboturk_20hz.npz`, `roboturk/fingerprint.json`, `roboturk/partD.json`

**Interfaces:**
- Consumes: `train_act.load_everything(npz, bcrnn_ckpt) -> (S, A, starts, eps, st)` where `eps` is `dict(train=, val=, cal=, eval=)` of episode-index arrays, `starts` is an `(n_ep+1,)` int64 offset array, and `st` is `dict(s_mu, s_sd, a_mu, a_sd)` of float32. `train_act.GRIP == 7`.
- Produces, for Task 2:
  - `conformal_threshold(scores: Sequence[float], alpha: float) -> tuple[float, int, int, bool]` returning `(q, n, rank, feasible)`
  - `load_scores(path: str, sha: str) -> dict` returning the npz contents as `{err7, err8, closed, bounds, ep_id, is_cal}`
  - npz keys, exactly: `err7` (float32, flat), `err8` (float32, flat), `closed` (bool, flat), `bounds` (int64, `(160,)`), `ep_id` (int64, `(159,)`), `is_cal` (bool, `(159,)`), `sha` (str)

- [x] **Step 1: Write the script skeleton with the model and the pure quantile function**

```python
"""RoboTurk BC-RNN thresholds -- Part 3 of RoboTurk_ACT_and_Full_Benchmark.md.

Computes the safety threshold q_hat_k and the escalation threshold q_hat_k,esc per
phase for the frozen BC-RNN, on the 7 CONTINUOUS action dims only.

Why not all 8: the 8th column is the grasp bool, which training deliberately keeps
unnormalised in {0,1} (a_mu[7]=0, a_sd[7]=1) while the 7 joint targets are z-scored.
It therefore owns a median 98.5% of the squared error at the timestep that sets each
phase maximum, making the published q_hat_k = 2.28 a readout of gripper-toggle timing
rather than of trajectory divergence -- vacuous against an eval max of 1.213. The grasp
bool stays the K=2 phase variable; it just stops setting the magnitude.

Correctness is established by reproducing partD.json's 8-dim thresholds from this same
code path before any 7-dim number is written.

    .venv-lerobot/bin/python roboturk/calibrate_bcrnn.py
"""
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_act import GRIP, load_everything          # same dir; frozen split + norm stats

HERE = Path(__file__).resolve().parent
ALPHA, K = 0.10, 2                                   # inherited from Part D, do not vary
ALPHA_ESC = (0.20, 0.30, 0.40)                       # 0.30 primary per the benchmark file


class BCRNN(nn.Module):
    """Verbatim from RoboTurk_BCRNN_Training.ipynb cell 28.

    Redefined rather than imported because the checkpoint stores a bare state_dict.
    The layer names here are load-bearing -- they are the state_dict's keys.
    """

    def __init__(self, sdim, adim, hidden=256, layers=2, modes=5):
        super().__init__()
        self.modes, self.adim = modes, adim
        self.enc = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU())
        self.rnn = nn.LSTM(256, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, modes * (2 * adim + 1))

    def forward(self, s, hx=None):
        B, T, _ = s.shape
        z = self.enc(s.reshape(B * T, -1)).reshape(B, T, -1)
        o, hx = self.rnn(z, hx)
        p = self.head(o).reshape(B, T, self.modes, 2 * self.adim + 1)
        return p[..., :self.adim], p[..., self.adim:2 * self.adim].clamp(-5, 2), p[..., -1], hx


def gmm_mean(mu, logit):
    """Mixture mean -- the point prediction Part D scores against."""
    return (torch.softmax(logit, -1).unsqueeze(-1) * mu).sum(-2)


def conformal_threshold(scores, alpha):
    """Split-conformal order statistic. Verbatim from notebook cell 34.

    Returns (q, n, rank, feasible). Infeasible (rank > n) yields inf, which is the
    correct answer -- a finite threshold does not exist at that alpha and n.
    """
    s = np.sort(np.asarray([x for x in scores if np.isfinite(x)], float))
    n = s.size
    if n == 0:
        return float("inf"), 0, 0, False
    rank = int(math.ceil((1 - alpha) * (n + 1)))
    if rank > n:
        return float("inf"), n, rank, False
    return float(s[rank - 1]), n, rank, True
```

- [x] **Step 2: Add the scoring pass and the cache**

`err8² = err7² + grasp_err²` exactly, so caching both norms makes the grasp diagnostic in Task 2 free — no third array needed.

```python
@torch.no_grad()
def score_episodes(model, S, A, starts, ep_ids, st):
    """One full-sequence forward per episode. Yields per-timestep scores in RAW space.

    Raw, not normalised, because that is what Part D scored and what the reproduction
    gate must match: cell 34 denormalises the prediction before taking the norm.
    """
    err7, err8, closed, bounds = [], [], [], [0]
    for n, i in enumerate(ep_ids, 1):
        s, a = S[starts[i]:starts[i + 1]], A[starts[i]:starts[i + 1]]
        sn = torch.from_numpy(((s - st["s_mu"]) / st["s_sd"])[None].astype(np.float32))
        mu, _, lg, _ = model(sn)
        ahat = gmm_mean(mu, lg)[0].numpy() * st["a_sd"] + st["a_mu"]
        d = ahat - a
        err7.append(np.linalg.norm(d[:, :GRIP], axis=1).astype(np.float32))
        err8.append(np.linalg.norm(d, axis=1).astype(np.float32))
        closed.append(a[:, GRIP] > 0.5)
        bounds.append(bounds[-1] + len(a))
        print(f"\r  scored {n:3d}/{len(ep_ids)} episodes", end="", flush=True)
    print()
    return (np.concatenate(err7), np.concatenate(err8),
            np.concatenate(closed), np.asarray(bounds, np.int64))


def build_cache(path, sha, model, S, A, starts, eps, st):
    """Score cal+eval once and persist, keyed by the checkpoint hash."""
    ep_id = np.concatenate([eps["cal"], eps["eval"]]).astype(np.int64)
    is_cal = np.arange(len(ep_id)) < len(eps["cal"])
    err7, err8, closed, bounds = score_episodes(model, S, A, starts, ep_id, st)
    np.savez_compressed(path, err7=err7, err8=err8, closed=closed,
                        bounds=bounds, ep_id=ep_id, is_cal=is_cal, sha=sha)
    return dict(err7=err7, err8=err8, closed=closed, bounds=bounds,
                ep_id=ep_id, is_cal=is_cal)


def load_scores(path, sha):
    """Reuse the cache only if it was built from this exact checkpoint."""
    if not Path(path).exists():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["sha"]) != sha:
        print(f"  cache was built from a different checkpoint ({str(z['sha'])[:12]}) -- recomputing")
        return None
    return {k: z[k] for k in ("err7", "err8", "closed", "bounds", "ep_id", "is_cal")}
```

- [x] **Step 3: Add per-episode aggregation and the reproduction gate**

The gate tolerance is **relative 1e-3**, not the 4 absolute decimals the spec first suggested. The published numbers came from CUDA (torch 2.6.0+cu118, RTX 3050); this runs on CPU with torch 2.10. LSTM and matmul reductions differ in association order between those backends, so bit-identity is not achievable and demanding it would fail a correct implementation. 1e-3 relative is ~0.002 on 2.28 — far tighter than any real reconstruction error (a wrong slice, a missed denormalisation, or a redrawn split all move the number by whole units, not thousandths). The measured delta is always printed and recorded.

```python
def phase_maxima(sc, which, is_cal):
    """Per-episode max of `which` within each phase -- the Part D aggregate (AGG=np.max)."""
    err, closed, b = sc[which], sc["closed"], sc["bounds"]
    out = {"closed": [], "open": []}
    for j in np.nonzero(sc["is_cal"] == is_cal)[0]:
        e, c = err[b[j]:b[j + 1]], closed[b[j]:b[j + 1]]
        if c.any():
            out["closed"].append(float(e[c].max()))
        if (~c).any():
            out["open"].append(float(e[~c].max()))
    return out


def reproduction_gate(sc, partd_path, rtol):
    """Recompute partD.json's 8-dim thresholds from this cache. Nothing is written unless
    they match -- if this code is not the code that produced Part D, its 7-dim output is
    not evidence of anything."""
    ref = json.load(open(partd_path))["per_category"]
    cal8 = phase_maxima(sc, "err8", is_cal=True)
    rows, ok = {}, True
    for c in ("closed", "open"):
        q, n, rank, feas = conformal_threshold(cal8[c], ALPHA / K)
        want = ref[c]["q"]
        delta = abs(q - want)
        passed = delta <= rtol * abs(want) and n == ref[c]["n_cal"] and rank == ref[c]["rank"]
        rows[c] = dict(partD=want, recomputed=q, delta=delta,
                       n_cal=n, rank=rank, passed=bool(passed))
        ok &= passed
        print(f"  {c:<6} partD {want:.7f}  recomputed {q:.7f}  delta {delta:.2e}  "
              f"n={n} rank={rank}  {'PASS' if passed else 'FAIL'}")
    return ok, rows
```

- [x] **Step 4: Wire `main()` far enough to run the gate**

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(HERE / "roboturk_20hz.npz"))
    p.add_argument("--bcrnn", default=str(HERE / "roboturk_bcrnn_state_only.pt"))
    p.add_argument("--cache", default=str(HERE / "bcrnn_scores.npz"))
    p.add_argument("--partd", default=str(HERE / "partD.json"))
    p.add_argument("--out", default=str(HERE / "partD_v2.json"))
    p.add_argument("--rtol", type=float, default=1e-3, help="reproduction gate tolerance")
    args = p.parse_args()

    for f in (args.npz, args.bcrnn, args.partd, HERE / "fingerprint.json"):
        if not Path(f).exists():
            raise SystemExit(f"missing required artifact: {f}")

    sha = hashlib.sha256(open(args.bcrnn, "rb").read()).hexdigest()
    fp = json.load(open(HERE / "fingerprint.json"))
    if sha != fp["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint sha {sha[:12]} != fingerprint {fp['checkpoint_sha256'][:12]}"
                         " -- the frozen checkpoint changed; stop.")

    S, A, starts, eps, st = load_everything(args.npz, args.bcrnn)
    sc = load_scores(args.cache, sha)
    if sc is None:
        torch.manual_seed(0)
        model = BCRNN(S.shape[1], A.shape[1]).eval()
        model.load_state_dict(torch.load(args.bcrnn, map_location="cpu",
                                         weights_only=False)["model"])
        print(f"scoring {len(eps['cal']) + len(eps['eval'])} held-out episodes on CPU")
        sc = build_cache(args.cache, sha, model, S, A, starts, eps, st)
    else:
        print(f"reusing score cache {args.cache}")

    print("\nreproduction gate -- 8-dim thresholds vs partD.json")
    ok, gate = reproduction_gate(sc, args.partd, args.rtol)
    if not ok:
        raise SystemExit("\nreproduction gate FAILED -- this is not the pipeline that produced "
                         "Part D, so its 7-dim output would not be trustworthy. Nothing written.")
    print("  gate PASSED\n")


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Run it and confirm the gate passes**

Run: `.venv-lerobot/bin/python roboturk/calibrate_bcrnn.py`

Expected: `scored 159/159 episodes`, then two `PASS` lines with `recomputed` within 1e-3 relative of `2.2835631` (closed) and `2.2110543` (open), `n=79 rank=76` on both, then `gate PASSED`. `roboturk/bcrnn_scores.npz` exists.

**Done when:** the command above prints `gate PASSED`, and `.venv-lerobot/bin/python -c "import numpy as np; z=np.load('roboturk/bcrnn_scores.npz'); print(z['bounds'][-1], z['ep_id'].shape, z['is_cal'].sum())"` prints a total timestep count near 940,000, `(159,)`, and `79`.

---

### Task 2: Thresholds, diagnostics, and the report

**Goal:** `roboturk/partD_v2.json` carries `q̂_k` and `q̂_k,esc` per phase at both aggregations, the dual-threshold check, the grasp diagnostic, eval coverage and eval firing rates.

**Why:** These are the numbers Part 3 exists to produce and that Part 5 is gated on. Without the firing rate in particular there is no way to know whether arbitration is viable at all — a `q_esc` that fires on 0.16% of timesteps makes H1 unfalsifiable, and that outcome must be visible in the output rather than discovered during the benchmark.

**Files:**
- Modify: `roboturk/calibrate_bcrnn.py`

**Interfaces:**
- Consumes: `conformal_threshold`, `phase_maxima`, `load_scores` and the npz keys from Task 1.
- Produces: `roboturk/partD_v2.json` (terminal artifact; nothing imports from this task).

- [x] **Step 1: Add pooled-timestep aggregation and the eval firing rate**

```python
def pooled(sc, which, is_cal):
    """All timesteps of the given split, pooled by phase.

    NOT exchangeable: ~5,900 timesteps within an episode are strongly autocorrelated and
    are not that many independent draws. A quantile of this carries NO coverage guarantee.
    It is computed because calibrating on per-episode maxima while triggering per-timestep
    is a category error that makes escalation fire on ~0% of steps -- this is the
    compute-budget heuristic that makes the arbitration benchmark runnable at all.
    """
    keep = np.repeat(sc["is_cal"] == is_cal, np.diff(sc["bounds"]))
    err, closed = sc[which][keep], sc["closed"][keep]
    return {"closed": err[closed], "open": err[~closed]}


def firing_rate(sc, q, which="err7"):
    """Fraction of eval timesteps that would escalate under per-phase thresholds q."""
    ev = pooled(sc, which, is_cal=False)
    n_hit = sum(int((ev[c] > q[c]).sum()) for c in ("closed", "open"))
    return n_hit / sum(len(ev[c]) for c in ("closed", "open"))
```

- [x] **Step 2: Add the grasp-bool diagnostic**

```python
def grasp_diagnostic(sc):
    """Share of squared error owned by the grasp bool at each phase-max timestep.

    err8^2 = err7^2 + grasp_err^2 exactly, so the share needs no extra cached array.
    Evaluated at the timestep that SETS each phase maximum under the 8-dim score --
    the timestep that actually determined the published threshold.
    """
    err7, err8, closed, b = sc["err7"], sc["err8"], sc["closed"], sc["bounds"]
    shares = []
    for j in np.nonzero(sc["is_cal"])[0]:
        e7, e8, c = err7[b[j]:b[j + 1]], err8[b[j]:b[j + 1]], closed[b[j]:b[j + 1]]
        for m in (c, ~c):
            if not m.any():
                continue
            t = np.nonzero(m)[0][e8[m].argmax()]
            if e8[t] > 0:
                shares.append(float((e8[t] ** 2 - e7[t] ** 2) / e8[t] ** 2))
    shares = np.asarray(shares)
    return dict(median_share_of_sq_err=float(np.median(shares)),
                frac_dominated=float(np.mean(shares > 0.5)),
                n=int(shares.size))
```

- [x] **Step 3: Add the threshold table builder**

```python
def thresholds(sc, alpha_total, agg):
    """Per-phase thresholds at alpha_total/K, plus eval coverage under the Part D convention.

    agg='max'    -> per-episode phase maxima (exchangeable; carries the guarantee)
    agg='pooled' -> every timestep (no guarantee; see pooled.__doc__)
    """
    a_k = alpha_total / K
    get = phase_maxima if agg == "max" else pooled
    cal, ev = get(sc, "err7", True), get(sc, "err7", False)
    out = {}
    for c in ("closed", "open"):
        q, n, rank, feas = conformal_threshold(np.asarray(cal[c], float), a_k)
        e = np.asarray(ev[c], float)
        out[c] = dict(q=q, n_cal=n, rank=rank, feasible=feas, n_eval=int(e.size),
                      coverage=float(np.mean(e <= q)) if np.isfinite(q) else 1.0)
    return out
```

- [x] **Step 4: Extend `main()` to compute everything and write the report**

Insert after the `gate PASSED` print, replacing nothing above it:

```python
    report = {
        "supersedes": "partD.json",
        "score": "||a_pred - a_demo||_2 over the 7 CONTINUOUS dims (user_control[:,0:7]) in raw "
                 "joint-target space; the grasp bool is excluded from the magnitude and retained "
                 "as the K=2 phase variable",
        "alpha": ALPHA, "K": K, "alpha_k": ALPHA / K,
        "checkpoint_sha256": sha,
        "device": "cpu",
        "reproduction_gate": {"rtol": args.rtol, "passed": True, **gate},
        "grasp_diagnostic": grasp_diagnostic(sc),
        "safety": {},
        "escalation": {},
        "caveats": "Per-timestep thresholds pool autocorrelated timesteps, break exchangeability, "
                   "and carry NO coverage guarantee -- they are a compute-budget heuristic for "
                   "choosing an escalation operating point. The safety guarantee lives in the "
                   "per-episode-max q_hat_k alone. All demos are successes; no failure labels "
                   "exist in RoboTurk, so nothing here is a detection or false-alarm rate.",
    }

    print("grasp-bool diagnostic (8-dim score, at each phase-max timestep)")
    g = report["grasp_diagnostic"]
    print(f"  median share of squared error owned by the grasp bool: {g['median_share_of_sq_err']:.3f}")
    print(f"  phase-maxima where it owns >50%: {g['frac_dominated']:.1%}  (n={g['n']})\n")

    for agg in ("max", "pooled"):
        report["safety"][agg] = thresholds(sc, ALPHA, agg)
        label = "per-episode max (GUARANTEED)" if agg == "max" else "per-timestep (heuristic)"
        print(f"safety q_hat_k, alpha={ALPHA} -- {label}")
        for c in ("closed", "open"):
            r = report["safety"][agg][c]
            print(f"  {c:<6} q={r['q']:.4f}  n_cal={r['n_cal']:>6} rank={r['rank']:>6}  "
                  f"feasible={r['feasible']}  eval coverage={r['coverage']:.3f}")

        for ae in ALPHA_ESC:
            t = thresholds(sc, ae, agg)
            q = {c: t[c]["q"] for c in ("closed", "open")}
            dual = {c: bool(q[c] < report["safety"][agg][c]["q"]) for c in ("closed", "open")}
            fire = firing_rate(sc, q)
            report["escalation"].setdefault(agg, {})[f"{ae:.2f}"] = dict(
                alpha_esc=ae, alpha_esc_k=ae / K, per_phase=t,
                dual_threshold_ok=dual, eval_firing_rate=fire,
                primary=(ae == 0.30))
            star = " *primary" if ae == 0.30 else ""
            print(f"  alpha_esc={ae:.2f}  q_closed={q['closed']:.4f} q_open={q['open']:.4f}  "
                  f"dual-threshold {'OK' if all(dual.values()) else 'FAILED'}  "
                  f"eval firing {fire:.2%}{star}")
        print()

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
```

- [x] **Step 5: Run and read the numbers**

Run: `.venv-lerobot/bin/python roboturk/calibrate_bcrnn.py`

Expected: gate still passes (cache is reused, so this is fast), the grasp diagnostic prints a median share near **0.985** confirming the defect independently, both safety blocks print, and every `alpha_esc` line reports `dual-threshold OK`.

**Done when:**

```bash
.venv-lerobot/bin/python -c "
import json; r=json.load(open('roboturk/partD_v2.json'))
assert r['reproduction_gate']['passed']
s=r['safety']['max']; e=r['escalation']
print('safety  max    closed %.4f open %.4f  cov %.3f/%.3f' % (
    s['closed']['q'], s['open']['q'], s['closed']['coverage'], s['open']['coverage']))
for agg in ('max','pooled'):
    x=e[agg]['0.30']
    print('esc .30 %-6s closed %.4f open %.4f  dual %s  firing %.2f%%' % (
        agg, x['per_phase']['closed']['q'], x['per_phase']['open']['q'],
        all(x['dual_threshold_ok'].values()), 100*x['eval_firing_rate']))
"
```

prints a safety `q` well below `partD.json`'s 2.28 (expected order ~0.1–1.0, since the grasp bool no longer sets it), eval coverage `≥ 0.90` in both phases, `dual True` on both aggregations, and a pooled firing rate materially above the 0.16% the 8-dim score gave.

---

## Interpreting the outcome — decided in advance, not after seeing the numbers

Three outcomes are possible and all three are results. Recording the reading rule now
prevents fitting the interpretation to whichever number arrives.

1. **Pooled firing rate comfortably non-trivial (say 1–15%) and dual-threshold OK.**
   Part 3 is satisfied; Part 5 is viable using the pooled `q_esc` as the operating point,
   labelled as a heuristic.
2. **Pooled firing rate still near zero.** The grasp bool was not the only thing wrong with
   the score, and Part 5 would produce arbitration ≈ BC-RNN-alone. Report it plainly and
   stop — per the benchmark file, "a failed or ambiguous check is a result."
3. **Dual-threshold check fails in either phase.** Part 3 says stop. Do not proceed to
   Part 5 and do not adjust `α_esc` until it passes — that would be choosing the threshold
   to make the check pass, which is exactly what the check is for.

A firing rate above ~50% also fails, differently: escalating on most timesteps means H3
(arbitration is cheaper than ACT-alone) cannot hold, and the operating point is wrong.

## Out of scope

- Part 5's three-way benchmark and Part 5.3's inference timing.
- Any ACT involvement. ACT training is running; this task must not touch it.
- Updating `docs/status.md` or `roboturk/RoboTurk_BCRNN_Findings.md`. Those record what the
  8-dim pass produced; revising them is a separate documentation decision once these
  numbers are read.
