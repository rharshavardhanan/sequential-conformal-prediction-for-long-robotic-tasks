"""PART 2 -- composed-task checks with the two verified experts, UNMODIFIED.

1. Composed success rate measured directly (n=200) vs the product of sub-skill rates.
2. Phase-boundary validity: puck disturbance by phase A, puck on table / in bounds,
   ee position, puck speed at the boundary; conditional slide success given boundary state.
3. Rendered frames at the phase boundary for a handful of episodes.
"""
import json
import os

import numpy as np

import compose as K
import common as C

N = 200
SEED0 = 130_000
V = json.load(open(os.path.join(K.HERE, "results", "part1_verify.json")))
pa = V["candidates"][K.EXPERT_A[1]]["success_rate"]
pb = (V["candidates"][K.EXPERT_B[1]]["success_rate"] if K.TASK == "slide" else json.load(open(os.path.join(os.path.dirname(K.RESULTS), "results", "push_ft_verify.json")))["fresh_verification"]["success_rate"])

env = K.Composed()
ex = K.Experts()
recs = [K.run_episode(env, SEED0 + i, None, ex, "expert") for i in range(N)]
k = sum(r["success"] for r in recs)
bd = [r["boundary"] for r in recs if r["boundary"]]
disp = np.array([b["puck_displacement_in_phaseA"] for b in bd])
disturbed = disp > 0.01
slide_ok = np.array([r["success"] for r in recs if r["boundary"]])
out = {
    "n": N, "composed_success": k, "composed_success_rate": k / N, "wilson95": C.wilson(k, N),
    "product_of_subskill_rates": pa * pb, "subskill_rates": {"reach": pa, K.PHASES[1]: pb},
    "phaseA_success_rate": float(np.mean([r["phaseA_success"] for r in recs])),
    "fail_phase_counts": {p: int(sum(r["fail_phase"] == p for r in recs)) for p in K.PHASES},
    "episode_len": {"mean": float(np.mean([r["T"] for r in recs])), "median": float(np.median([r["T"] for r in recs])),
                    "max": int(max(r["T"] for r in recs))},
    "phaseA_steps_median": float(np.median([r["t_switch"] for r in recs if r["t_switch"]])),
    "phaseB_steps_to_success_median": float(np.median([r["T"] - r["t_switch"] for r in recs if r["success"]])),
    "boundary": {
        "puck_displacement_in_phaseA": {"median": float(np.median(disp)), "p90": float(np.quantile(disp, .9)),
                                        "max": float(disp.max()), "frac_gt_1cm": float(disturbed.mean())},
        "puck_z_range": [float(min(b["puck_z"] for b in bd)), float(max(b["puck_z"] for b in bd))],
        # puck is a cylinder of height object_size/2 = 0.03, so its centre rests at z = 0.015
        "puck_off_table_or_airborne": int(sum(abs(b["puck_z"] - K.REST_Z) > 0.01 for b in bd)),
        "puck_speed_at_boundary_p90": float(np.quantile([b["puck_speed"] for b in bd], .9)),
        "ee_z_range": [float(min(b["ee"][2] for b in bd)), float(max(b["ee"][2] for b in bd))],
        "slide_success_given_puck_undisturbed": float(slide_ok[~disturbed].mean()) if (~disturbed).any() else None,
        "slide_success_given_puck_disturbed": float(slide_ok[disturbed].mean()) if disturbed.any() else None,
        "n_disturbed": int(disturbed.sum()),
    },
    "example_boundaries": bd[:5],
}

# Boundary distribution-shift test: SAME seeds (same puck + goal, since the slide env
# reset is seeded identically), slide expert from its NATIVE start pose, no reach phase.
import gymnasium as gym  # noqa: E402
senv = gym.make(K.SCENE)
native = []
for i in range(N):
    obs, _ = senv.reset(seed=SEED0 + i)
    done, ok = False, False
    while not done:
        obs, _, term, trunc, info = senv.step(ex.b(obs))
        ok, done = bool(info["is_success"]), term or trunc
    native.append(ok)
native = np.array(native)
comp = np.array([r["success"] for r in recs])
n10, n01 = int(np.sum(native & ~comp)), int(np.sum(~native & comp))
from scipy.stats import binomtest  # noqa: E402
ee_home = np.array(senv.unwrapped.robot.get_ee_position())
senv.reset(seed=0)
ee_home = np.array(senv.unwrapped.robot.get_ee_position())
shift = np.array([np.linalg.norm(np.array(r["boundary"]["ee"]) - ee_home) for r in recs])
out["distribution_shift_test"] = {
    "native_slide_success_same_seeds": float(native.mean()), "composed_success_same_seeds": float(comp.mean()),
    "native_only_success": n10, "composed_only_success": n01,
    "mcnemar_exact_p": float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0,
    "ee_home": ee_home.tolist(),
    "ee_shift_from_home_median": float(np.median(shift)),
    "composed_success_by_ee_shift_tercile": [float(comp[(shift >= lo) & (shift <= hi)].mean()) for lo, hi in
                                             zip(np.quantile(shift, [0, 1 / 3, 2 / 3]), np.quantile(shift, [1 / 3, 2 / 3, 1]))],
}

# frames at the boundary (and at reset) for 4 episodes, incl. a disturbed one if any
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    renv = K.Composed(render=True)
    pick = [r["seed"] for r in recs if r["boundary"]][:3]
    dis = [r["seed"] for r in recs if r["boundary"] and r["boundary"]["puck_displacement_in_phaseA"] > 0.01]
    pick += dis[:1]
    fig, axs = plt.subplots(len(pick), 3, figsize=(10, 2.8 * len(pick)))
    for row, sd in zip(np.atleast_2d(axs), pick):
        renv.reset(sd)
        row[0].imshow(renv.env.render()); row[0].set_title("seed %d reset" % sd, fontsize=8)
        ev = "continue"
        while ev == "continue":
            ev, _ = renv.step(ex(renv))
        row[1].imshow(renv.env.render()); row[1].set_title("phase boundary (%s) t=%d" % (ev, renv.t), fontsize=8)
        while ev not in ("success", "fail"):
            ev, _ = renv.step(ex(renv))
        row[2].imshow(renv.env.render()); row[2].set_title("end: %s t=%d" % (ev, renv.t), fontsize=8)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.join(K.RESULTS, "figures"), exist_ok=True)
    fig.savefig(os.path.join(K.RESULTS, "figures", "phase_boundary_frames.png"), dpi=110)
    out["frames"] = {"file": "figures/phase_boundary_frames.png", "seeds": pick}
except Exception as e:
    out["frames"] = {"error": repr(e)}

json.dump(out, open(os.path.join(K.RESULTS, "part2_compose.json"), "w"), indent=2)
print(json.dumps({k_: v for k_, v in out.items() if k_ != "example_boundaries"}, indent=1))

