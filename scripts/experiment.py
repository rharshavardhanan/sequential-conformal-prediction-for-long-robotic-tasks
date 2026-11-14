"""Run one episode N times and aggregate. Visualization/sanity, not a benchmark.

N=5 gives a 95% CI of roughly +/-0.44 on a success rate near 0.5 -- wide enough
to contain almost any hypothesis. The per-task success rates this project
reports come from 150 rollouts each, not from this. The summary says so out loud
so a screenshot of it cannot be mistaken for evidence.
"""

import math


def summarize_runs(runs, label="runs"):
    """runs: list of dicts with success, steps, verdict, violation_step, ...

    Returns an aggregate dict and prints a per-run table.
    """
    n = len(runs)
    ok = sum(1 for r in runs if r.get("success"))
    rate = ok / n if n else float("nan")
    # normal-approximation interval, honest about being crude at small n
    ci = 1.96 * math.sqrt(rate * (1 - rate) / n) if n else float("nan")

    print("\n" + "=" * 72)
    print(f"  {n} {label.upper()}")
    print("=" * 72)
    print(f"  {'run':>4} {'outcome':>8} {'steps':>6} {'monitor':>14} "
          f"{'first viol':>11} {'first esc':>10} {'stall':>7} {'switches':>9} "
          f"{'max score':>10}")
    for i, r in enumerate(runs, 1):
        print(f"  {i:>4} {'SUCCESS' if r.get('success') else 'fail':>8} "
              f"{r.get('steps', 0):>6} {str(r.get('verdict')):>14} "
              f"{str(r.get('violation_step')):>11} {str(r.get('escalation_step')):>10} "
              f"{str(r.get('stall_step')):>7} {str(r.get('switches', '-')):>9} "
              f"{r.get('max_err_norm', float('nan')):>10.3f}")

    print("-" * 72)
    print(f"  policy outcome   : {ok}/{n} success   empirical rate "
          f"{rate:.2f} +/-{ci:.2f}")
    caught = sum(1 for r in runs if r.get("verdict") not in (None, "SAFE"))
    print(f"  monitor outcome  : {caught}/{n} flagged (VIOLATION/ESCALATE/STUCK)")

    # Flat-CP baseline, when the runs were produced with --compare-flat. This is
    # the head-to-head the whole per-phase argument rests on, counted rather
    # than asserted: same episodes, same score, two ways of thresholding it.
    if any("flat_verdict" in r for r in runs):
        sub_hit = [r.get("violation_step") is not None for r in runs]
        flat_hit = [r.get("flat_verdict") == "VIOLATION" for r in runs]
        only_sub = sum(a and not b for a, b in zip(sub_hit, flat_hit))
        only_flat = sum(b and not a for a, b in zip(sub_hit, flat_hit))
        both = sum(a and b for a, b in zip(sub_hit, flat_hit))
        neither = sum((not a) and (not b) for a, b in zip(sub_hit, flat_hit))
        print("-" * 72)
        print(f"  SUBTASK CP flagged : {sum(sub_hit)}/{n}")
        print(f"  FLAT CP    flagged : {sum(flat_hit)}/{n}"
              f"   (single threshold {runs[0].get('flat_threshold', float('nan')):.3f})")
        print(f"  both {both}   subtask only {only_sub}   "
              f"flat only {only_flat}   neither {neither}")
        if only_sub or only_flat:
            print("  the disagreements are what per-phase conditioning buys/costs")
    print(f"  NOTE: policy outcome and monitor outcome are DIFFERENT things --")
    print(f"        a run can succeed and still be flagged, or fail unflagged.")
    sw = [r["switches"] for r in runs if r.get("switches") is not None]
    if sw:
        print(f"  policy switches  : {sum(sw)} total, {sum(sw)/len(sw):.1f} per run")
    frac = [r["expensive_fraction"] for r in runs
            if r.get("expensive_fraction") is not None]
    if frac:
        print(f"  expensive arm    : {sum(frac)/len(frac):.1%} of steps (mean)")
    print(f"\n  {n} runs is a SANITY CHECK, not a benchmark. The success rates")
    print(f"  reported for this project come from 150 rollouts per task.")
    print("=" * 72)

    return {"n_runs": n, "n_success": ok, "success_rate": rate, "ci95": ci,
            "n_flagged": caught,
            "switches_total": sum(sw) if sw else None,
            "expensive_fraction_mean": (sum(frac)/len(frac)) if frac else None,
            "runs": runs}
