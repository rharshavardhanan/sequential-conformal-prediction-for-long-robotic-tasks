"""D.1 Option A: extend a task's init-state pool to >=150 distinct valid states.

LIBERO ships 100 states in `*.init` (and 50 in `*.pruned_init`). Collecting 150
rollouts from 100 states with `i % 100` reuses states 0-49 twice and 50-99 once
-- a *structured* reuse pattern, which is not exchangeable. Extending the pool so
every rollout gets its own state removes the problem at the source.

New states come from `env.reset()`, which runs robosuite's placement sampler, then
`sim.get_state().flatten()`. The shipped file is named *pruned*, implying upstream
filtered invalid configurations, so every generated state is validated before use:

  1. finite, correct shape
  2. task NOT already satisfied at t=0  (else the rollout is trivially a success)
  3. no deep contact interpenetration after sim.forward()
  4. every object within a generous workspace box
  5. not a duplicate of an existing state

  .venv-lerobot/bin/python -m scripts.gen_init_states --task-id 0 --target 160
"""

import argparse
import os

import numpy as np

# Objects rest on the table with a constant negative contact distance under
# MuJoCo's soft contacts (-14.59 mm on every one of task 0's shipped states, an
# unnamed table geom against an object). That is resting contact, not
# interpenetration, so an absolute threshold rejects everything including the
# ground-truth states. The threshold is therefore CALIBRATED from the shipped
# pool at run time: those states define validity by construction.
PENETRATION_MARGIN = 5e-3   # metres allowed beyond the worst shipped state
WORKSPACE = dict(x=(-1.5, 1.5), y=(-1.5, 1.5), z=(0.0, 2.0))


def deepest_contact(sim):
    return min([sim.data.contact[i].dist for i in range(sim.data.ncon)] + [0.0])


def validate(env, state, n_expect, max_pen):
    """Return (ok, reason). `max_pen` is calibrated from the shipped pool."""
    if state.shape[0] != n_expect:
        return False, f"shape {state.shape[0]} != {n_expect}"
    if not np.all(np.isfinite(state)):
        return False, "non-finite"
    try:
        env.set_init_state(state)
    except Exception as e:
        return False, f"set_init_state raised {type(e).__name__}"
    sim = env.env.sim
    if env.env._check_success():
        return False, "task already satisfied at t=0"
    pen = -deepest_contact(sim)
    if pen > max_pen:
        return False, f"interpenetration {pen*1000:.1f} mm > {max_pen*1000:.1f} mm"
    for name, obj in env.env.objects_dict.items():
        try:
            p = sim.data.body_xpos[sim.model.body_name2id(obj.root_body)]
        except Exception:
            continue
        for ax, lo_hi in zip(range(3), (WORKSPACE["x"], WORKSPACE["y"], WORKSPACE["z"])):
            if not (lo_hi[0] <= p[ax] <= lo_hi[1]):
                return False, f"{name} outside workspace on axis {ax}: {p[ax]:.3f}"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--target", type=int, default=160)
    ap.add_argument("--max-attempts", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--outdir", default="data/init_states")
    args = ap.parse_args()

    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    base_name = task.init_states_file.replace(".pruned_init", ".init")
    base = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder,
                                   base_name), weights_only=False)
    base = np.asarray(base)
    n_expect = base.shape[1]
    print(f"[task {args.task_id}] {task.language}")
    print(f"[base] {base_name}  {base.shape}")

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(args.seed)

    # Calibrate the penetration threshold from the shipped states themselves.
    pens = []
    for i in range(base.shape[0]):
        env.set_init_state(base[i])
        pens.append(-deepest_contact(env.env.sim))
    pens = np.asarray(pens)
    max_pen = pens.max() + PENETRATION_MARGIN
    print(f"[calib] shipped penetration: min={pens.min()*1000:.2f} "
          f"median={np.median(pens)*1000:.2f} max={pens.max()*1000:.2f} mm")
    print(f"[calib] threshold = {max_pen*1000:.2f} mm")

    # keep the shipped states first, in order, so index k < 100 is unchanged
    pool = [base[i] for i in range(base.shape[0])]
    kept_base = sum(validate(env, base[i], n_expect, max_pen)[0]
                    for i in range(base.shape[0]))
    print(f"[base] {kept_base}/{base.shape[0]} shipped states pass the calibrated validator")
    if kept_base < base.shape[0]:
        print("[WARN] validator still rejects ground-truth states -- it is wrong, not them")

    rejects = {}
    attempts = 0
    rng = np.random.default_rng(args.seed)
    while len(pool) < args.target and attempts < args.max_attempts:
        attempts += 1
        env.reset()
        s = np.asarray(env.env.sim.get_state().flatten())
        ok, why = validate(env, s, n_expect, max_pen)
        if not ok:
            rejects[why.split(":")[0]] = rejects.get(why.split(":")[0], 0) + 1
            continue
        if any(np.allclose(s, p, atol=1e-6) for p in pool):
            rejects["duplicate"] = rejects.get("duplicate", 0) + 1
            continue
        pool.append(s)
        if len(pool) % 10 == 0:
            print(f"  pool {len(pool)}/{args.target}  (attempts {attempts})", flush=True)

    arr = np.stack(pool)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"task{args.task_id}_init_{arr.shape[0]}.npy")
    np.save(out, arr)
    print(f"\n[result] pool {arr.shape}  attempts={attempts}")
    print(f"[result] generated {arr.shape[0]-base.shape[0]} new states")
    print(f"[reject] {rejects if rejects else 'none'}")
    print(f"[out] {out}")
    if arr.shape[0] < args.target:
        print(f"[WARN] target {args.target} NOT reached -- fall back to D.1 Option B")
    env.close()


if __name__ == "__main__":
    main()
