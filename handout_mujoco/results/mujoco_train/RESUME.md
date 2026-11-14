# Where Phase 2/3 stands — stopped 2026-09-14

Training was stopped mid-run at the user's request (machine going offline). Nothing here
is a gate result. Regenerate any number below by re-running the command that produced it.

## Complete and trustworthy

| item | state |
|---|---|
| Phase 1 + 1.5 (both envs) | **done**, Gate 1.5 passed 37/37 (`PYTHONPATH=live ../.venv-lerobot/bin/python live/test_envs.py`) |
| Push expert **seed 0**, 300k | **done** — 0.930 on 200 held-out seeds, CI95 [0.895, 0.965], median ep_len 5 |
| Step 2a sanity (incl. HER relabeling) | **done**, 10/10 |
| B1 push difficulty breakdown | **done** — EASY 1.000 / HARD 0.900, gap +0.100, r = −0.218 |
| B2 inference latency | **done** — CPU 0.106 ms vs MPS 2.658 ms p50; **CPU is the Phase-9 device** |
| B3 param accounting | **done** — actor 541,190 vs total 2,751,082; `n_params` removed everywhere |
| Contention check | **done** — 3-way gives 18.0 fps/job vs 38.0 solo; reach was killed per the rule |

## Incomplete — must be re-run from scratch

| item | reached | why it cannot be resumed |
|---|---|---|
| Push expert **seed 1** | 60k / 300k, success 0.620 | replay buffer is not checkpointed |
| Push expert **seed 2** | 60k / 300k, success 0.590 | same |
| Reach expert seed 0 (Phase 3) | not started | killed during the contention check, never relaunched |

`seed1/best_model.zip` and `seed2/best_model.zip` exist at the 60k eval. They are usable as
*policies* but are NOT a resume point: `model.learn()` on them starts with an empty replay
buffer, which is a different training trajectory, not a continuation. Treat a restart as a
restart.

## To resume (one job at a time — 3-way contention was measured to be a net loss)

```bash
cd handout_mujoco
# ~3.5 h each at the measured 38 steps/s solo
PYTHONPATH=live ../.venv-lerobot/bin/python -u training/train_push_expert.py --steps 300000 --seed 1
PYTHONPATH=live ../.venv-lerobot/bin/python -u training/train_push_expert.py --steps 300000 --seed 2
PYTHONPATH=live ../.venv-lerobot/bin/python -u training/train_reach_expert.py --steps 300000 --seed 0
```

Then Gate 2 (push ≥ 0.85 on seeds 400,000–400,199, per seed **and** mean ± sd across the
three) and Gate 3 (reach ≥ 0.95).

## Carried forward — do not lose

1. **0.930 is conditional on the P-servo hand-off.** PushExpert has only ever seen starts
   drawn from `reach_handoff_sample.npy`, fitted to a P-servo, not to ReachExpert. The
   Phase 4 checkpoint in `docs/superpowers/plans/2026-09-14-mujoco-split-envs.md` must
   measure what ReachExpert actually delivers and refit if it differs. **Never quote 0.930
   as a composed-success prediction.**
2. **3.5% of push failures are the cube leaving the table**, concentrated in Q3 (8%). Phase 1
   found that branch unreachable under a P-servo; a trained policy does fling the cube.
   Keep the branch, keep reporting the rate.
3. **Phase 9 must time every arm batch-1 on CPU.** The device ranking inverts with network
   size; mixing devices would turn a device switch into a fake compute saving.
4. **Phase 10's shift is modest.** EASY−HARD is 10 points with the expert at ceiling on both
   easy quartiles — the rate-control claim has something to measure, but not much.
