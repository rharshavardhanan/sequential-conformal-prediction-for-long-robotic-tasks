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

from train_act import GRIP, load_everything  # same dir; frozen split + norm stats

HERE = Path(__file__).resolve().parent
ALPHA, K = 0.10, 2  # inherited from Part D, do not vary
ALPHA_ESC = (0.20, 0.30, 0.40)  # 0.30 primary per the benchmark file


class BCRNN(nn.Module):
    """Verbatim from RoboTurk_BCRNN_Training.ipynb cell 28.

    Redefined rather than imported because the checkpoint stores a bare state_dict.
    The layer names here are load-bearing -- they are the state_dict's keys.
    """

    def __init__(self, sdim, adim, hidden=256, layers=2, modes=5):
        super().__init__()
        self.modes, self.adim = modes, adim
        self.enc = nn.Sequential(
            nn.Linear(sdim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.rnn = nn.LSTM(256, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, modes * (2 * adim + 1))

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


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_episodes(model, S, A, starts, ep_ids, st):
    """One full-sequence forward per episode. Yields per-timestep scores in RAW space.

    Raw, not normalised, because that is what Part D scored and what the reproduction
    gate must match: cell 34 denormalises the prediction before taking the norm.
    """
    err7, err8, closed, bounds = [], [], [], [0]
    for n, i in enumerate(ep_ids, 1):
        s, a = S[starts[i] : starts[i + 1]], A[starts[i] : starts[i + 1]]
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
    return (
        np.concatenate(err7),
        np.concatenate(err8),
        np.concatenate(closed),
        np.asarray(bounds, np.int64),
    )


def build_cache(path, sha, model, S, A, starts, eps, st):
    """Score cal+eval once and persist, keyed by the checkpoint hash."""
    ep_id = np.concatenate([eps["cal"], eps["eval"]]).astype(np.int64)
    is_cal = np.arange(len(ep_id)) < len(eps["cal"])
    err7, err8, closed, bounds = score_episodes(model, S, A, starts, ep_id, st)
    np.savez_compressed(
        path,
        err7=err7,
        err8=err8,
        closed=closed,
        bounds=bounds,
        ep_id=ep_id,
        is_cal=is_cal,
        sha=sha,
    )
    return dict(
        err7=err7, err8=err8, closed=closed, bounds=bounds, ep_id=ep_id, is_cal=is_cal
    )


def load_scores(path, sha):
    """Reuse the cache only if it was built from this exact checkpoint."""
    if not Path(path).exists():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["sha"]) != sha:
        print(
            f"  cache was built from a different checkpoint ({str(z['sha'])[:12]}) -- recomputing"
        )
        return None
    return {k: z[k] for k in ("err7", "err8", "closed", "bounds", "ep_id", "is_cal")}


# ---------------------------------------------------------------------- aggregation
def phase_maxima(sc, which, is_cal):
    """Per-episode max of `which` within each phase -- the Part D aggregate (AGG=np.max)."""
    err, closed, b = sc[which], sc["closed"], sc["bounds"]
    out = {"closed": [], "open": []}
    for j in np.nonzero(sc["is_cal"] == is_cal)[0]:
        e, c = err[b[j] : b[j + 1]], closed[b[j] : b[j + 1]]
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
        q, n, rank, _feas = conformal_threshold(cal8[c], ALPHA / K)
        want = ref[c]["q"]
        delta = abs(q - want)
        passed = (
            delta <= rtol * abs(want)
            and n == ref[c]["n_cal"]
            and rank == ref[c]["rank"]
        )
        rows[c] = dict(
            partD=want,
            recomputed=q,
            delta=delta,
            n_cal=n,
            rank=rank,
            passed=bool(passed),
        )
        ok &= passed
        print(
            f"  {c:<6} partD {want:.7f}  recomputed {q:.7f}  delta {delta:.2e}  "
            f"n={n} rank={rank}  {'PASS' if passed else 'FAIL'}"
        )
    return ok, rows


def pooled(sc, which, is_cal):
    """All timesteps of the given split, pooled by phase.

    NOT exchangeable: ~5,000 timesteps within an episode are strongly autocorrelated and
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


def grasp_diagnostic(sc):
    """Share of squared error owned by the grasp bool at each phase-max timestep.

    err8^2 = err7^2 + grasp_err^2 exactly, so the share needs no extra cached array.
    Evaluated at the timestep that SETS each phase maximum under the 8-dim score --
    the timestep that actually determined the published threshold.
    """
    err7, err8, closed, b = sc["err7"], sc["err8"], sc["closed"], sc["bounds"]
    shares = []
    for j in np.nonzero(sc["is_cal"])[0]:
        e7, e8, c = (
            err7[b[j] : b[j + 1]],
            err8[b[j] : b[j + 1]],
            closed[b[j] : b[j + 1]],
        )
        for m in (c, ~c):
            if not m.any():
                continue
            t = np.nonzero(m)[0][e8[m].argmax()]
            if e8[t] > 0:
                shares.append(float((e8[t] ** 2 - e7[t] ** 2) / e8[t] ** 2))
    shares = np.asarray(shares)
    return dict(
        median_share_of_sq_err=float(np.median(shares)),
        frac_dominated=float(np.mean(shares > 0.5)),
        n=int(shares.size),
    )


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
        out[c] = dict(
            q=q,
            n_cal=n,
            rank=rank,
            feasible=feas,
            n_eval=int(e.size),
            coverage=float(np.mean(e <= q)) if np.isfinite(q) else 1.0,
        )
    return out


# ---------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(HERE / "roboturk_20hz.npz"))
    p.add_argument("--bcrnn", default=str(HERE / "roboturk_bcrnn_state_only.pt"))
    p.add_argument("--cache", default=str(HERE / "bcrnn_scores.npz"))
    p.add_argument("--partd", default=str(HERE / "partD.json"))
    p.add_argument("--out", default=str(HERE / "partD_v2.json"))
    p.add_argument(
        "--rtol", type=float, default=1e-3, help="reproduction gate tolerance"
    )
    args = p.parse_args()

    for f in (args.npz, args.bcrnn, args.partd, HERE / "fingerprint.json"):
        if not Path(f).exists():
            raise SystemExit(f"missing required artifact: {f}")

    sha = hashlib.sha256(open(args.bcrnn, "rb").read()).hexdigest()
    fp = json.load(open(HERE / "fingerprint.json"))
    if sha != fp["checkpoint_sha256"]:
        raise SystemExit(
            f"checkpoint sha {sha[:12]} != fingerprint {fp['checkpoint_sha256'][:12]}"
            " -- the frozen checkpoint changed; stop."
        )

    S, A, starts, eps, st = load_everything(args.npz, args.bcrnn)
    sc = load_scores(args.cache, sha)
    if sc is None:
        torch.manual_seed(0)
        model = BCRNN(S.shape[1], A.shape[1]).eval()
        model.load_state_dict(
            torch.load(args.bcrnn, map_location="cpu", weights_only=False)["model"]
        )
        print(f"scoring {len(eps['cal']) + len(eps['eval'])} held-out episodes on CPU")
        sc = build_cache(args.cache, sha, model, S, A, starts, eps, st)
    else:
        print(f"reusing score cache {args.cache}")

    print("\nreproduction gate -- 8-dim thresholds vs partD.json")
    ok, gate = reproduction_gate(sc, args.partd, args.rtol)
    if not ok:
        raise SystemExit(
            "\nreproduction gate FAILED -- this is not the pipeline that produced "
            "Part D, so its 7-dim output would not be trustworthy. Nothing written."
        )
    print("  gate PASSED\n")

    report = {
        "supersedes": "partD.json",
        "score": "||a_pred - a_demo||_2 over the 7 CONTINUOUS dims (user_control[:,0:7]) in raw "
        "joint-target space; the grasp bool is excluded from the magnitude and retained "
        "as the K=2 phase variable",
        "alpha": ALPHA,
        "K": K,
        "alpha_k": ALPHA / K,
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
    print(
        f"  median share of squared error owned by the grasp bool: {g['median_share_of_sq_err']:.3f}"
    )
    print(
        f"  phase-maxima where it owns >50%: {g['frac_dominated']:.1%}  (n={g['n']})\n"
    )

    for agg in ("max", "pooled"):
        report["safety"][agg] = thresholds(sc, ALPHA, agg)
        label = (
            "per-episode max (GUARANTEED)"
            if agg == "max"
            else "per-timestep (heuristic)"
        )
        print(f"safety q_hat_k, alpha={ALPHA} -- {label}")
        for c in ("closed", "open"):
            r = report["safety"][agg][c]
            print(
                f"  {c:<6} q={r['q']:.4f}  n_cal={r['n_cal']:>6} rank={r['rank']:>6}  "
                f"feasible={r['feasible']}  eval coverage={r['coverage']:.3f}"
            )

        for ae in ALPHA_ESC:
            t = thresholds(sc, ae, agg)
            q = {c: t[c]["q"] for c in ("closed", "open")}
            dual = {
                c: bool(q[c] < report["safety"][agg][c]["q"])
                for c in ("closed", "open")
            }
            fire = firing_rate(sc, q)
            report["escalation"].setdefault(agg, {})[f"{ae:.2f}"] = dict(
                alpha_esc=ae,
                alpha_esc_k=ae / K,
                per_phase=t,
                dual_threshold_ok=dual,
                eval_firing_rate=fire,
                primary=(ae == 0.30),
            )
            star = " *primary" if ae == 0.30 else ""
            print(
                f"  alpha_esc={ae:.2f}  q_closed={q['closed']:.4f} q_open={q['open']:.4f}  "
                f"dual-threshold {'OK' if all(dual.values()) else 'FAILED'}  "
                f"eval firing {fire:.2%}{star}"
            )
        print()

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
