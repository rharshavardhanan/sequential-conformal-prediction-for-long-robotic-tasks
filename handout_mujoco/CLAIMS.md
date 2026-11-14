# What this arbitration demo can and cannot claim

Task: composed **Reach → Push**, panda-gym 3.0.7 (PyBullet), deterministic resets, **500 paired held-out seeds** per condition. Primary setting α_esc = 0.30. Source: `results/part45_benchmark.json`, `results/efficiency_summary.json`.

## The setup
- **Cheap policy:** a 5.9k-parameter MLP distilled from the experts.
- **Expert pair:**
  - Reach: TQC, 5.4k-parameter actor, 100% success.
  - Push: TQC, 541k-parameter actor. Fine-tuned from a community checkpoint; 0.90 success on 200 fresh seeds.
- **Arbiter:** a phase-conditioned conformal trigger on the commanded-vs-achieved end-effector score.
  - Calibration: n = 256 successful cheap episodes; all thresholds feasible.
  - Rule: escalate to the expert when s(t) > q̂_esc[phase]; hand back after 8 quiet steps.

## Headline numbers (α_esc = 0.30)

| | Cheap only | **Arbitrated** | Expert only |
|---|---|---|---|
| Task success (Wilson 95%) | 0.436 [0.39, 0.48] | **0.530 [0.49, 0.57]** | 0.840 [0.81, 0.87] |
| Expert calls per episode | 0 | **2.9** | 19.7 |
| Share of steps on the expert | 0% | **9%** (pooled) / 13% (per-episode mean) | 100% |

The success/cost trade-off is tunable through α_esc:

| α_esc | Success | Expert steps (pooled) |
|---|---|---|
| 0.20 | 0.494 | 5% |
| **0.30** | **0.530** | **9%** |
| 0.40 | 0.554 | 12% |

## ✅ Claims supported by the data
1. **A working closed-loop arbitration system.** Two policies switch *live* inside the simulator, driven by a calibrated conformal trigger. The live switching was cross-checked against the offline rule on every episode.
2. **Arbitration improves on the cheap policy:** +9.4 points success (paired bootstrap 95% CI +6.6 to +12.4; exact McNemar p = 2e−10).
3. **It does so with few expert calls:** about 2.9 per episode, **85% fewer** than expert-only.
4. **Each expert call buys more success than in expert-only mode:**
   - Arbitration: +9.4 points over cheap from 2.9 calls, ≈ 3.2 points per call.
   - Expert-only: +40.4 points from 19.7 calls, ≈ 2.1 points per call.
   - That is about **1.6× the success gain per expert call.**
5. **The trade-off is controllable** through a single calibrated parameter, α_esc (table above).
6. **The calibration is valid** under the conformal machinery. Thresholds are feasible at α = 0.10 and K = 2 with n = 256, and escalation thresholds are below the safety thresholds in both phases.

## ❌ Do not claim
- **Near-parity with the expert.** Arbitrated 0.53 is well below expert-only 0.84.
- **That the calibrated trigger picks the right moments better than chance.** At matched expert usage, random escalation achieves statistically indistinguishable success (+0.026, CI −0.014 to +0.066, p = 0.24).
  - It is fine to show the arbiter working without this comparison.
  - Do not state or imply that the *timing* is what creates the efficiency.
- **Wall-clock or energy savings in this demo.**
  - Measured CPU inference per step: cheap 0.08 ms, Reach expert 0.32 ms, Push expert 0.41 ms. The physics step (about 1.3 ms) dominates.
  - The efficiency claim is about **expert-model invocations**. That matters when the expert is expensive, such as a large policy or a remote model; it is not a speed-up of this simulation.
- **That the policies run in MuJoCo.** The MuJoCo videos are **kinematic replays** of PyBullet runs (see README).

## Disclosures to keep with any presentation
- **Gate override.** A pre-registered suitability gate for this task (G3.1) failed at its boundary (median irreversibility step 15 vs "> 15"). The rescue-curve gate (G3.2) passed. The calibration and benchmark were run on explicit user authorisation.
- **Expert fine-tuning.** The Push expert was fine-tuned locally (continued TQC + HER training) before verification.
- **Scope.** One simulated task, not pooled with the LIBERO or RoboTurk results.
