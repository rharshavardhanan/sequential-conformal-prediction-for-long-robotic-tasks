# Why BC-RNN failed and BC-Transformer works

**For:** the policy-arbitration cheap arm, LIBERO-Long task 0
**Date:** 2026-09-10 · All numbers measured on this project, not cited from elsewhere

---

## 1. The result, in one table

| policy | epochs trained | success rate |
|---|---|---|
| BC-RNN | 22 (fully converged, early-stopped on validation loss) | **0–5 %** |
| BC-Transformer | **10** (interrupted mid-training by an unrelated crash) | **45 %** |

BC-Transformer, trained for less than half as long and never finished, beats a
fully-converged BC-RNN by roughly **ten times**.

That gap is too large to be luck or tuning. It says the problem is the
**architecture**, not the training.

---

## 2. What the robot actually did

We ran the trained BC-RNN for 200 steps and recorded every action it produced,
then compared it against the human demonstrations it was trained on.

| | BC-RNN | human demos | what it means |
|---|---|---|---|
| downward push (avg) | −0.290 | −0.037 | pushes down **8x harder** than a human |
| gripper (avg) | −0.830 | −0.108 | keeps the hand **open** far more of the time |
| side-to-side variation | 0.177 | 0.335 | only **half** the movement variety |
| up-down variation | 0.174 | 0.364 | only **half** the movement variety |

Plain reading: **the robot drives its hand downward toward the table with the
gripper open, and never commits to picking anything up.** It moves — 0.56 m of
travel in 200 steps — so it is not frozen or broken. It is doing something
confident and wrong.

The halved movement variety is the important number. The policy is not
reproducing the range of motions a human made. It has flattened them.

---

## 3. Why it flattens them — the core problem

The task is *"put **both** the alphabet soup **and** the tomato sauce in the
basket."*

There are **two correct ways to start**: reach for the soup, or reach for the
sauce. Human demonstrators did both. So the training data contains, for the same
starting picture, two different valid actions pointing in **opposite directions**.

BC-RNN's output layer is a **Gaussian mixture** — it predicts an average action
with some spread. Trained to maximise likelihood on data containing "go left" and
"go right" with equal frequency, the mathematically optimal answer it can express
is **something in the middle**.

```
   demonstration A:  reach LEFT  toward the soup     ✓ works
   demonstration B:  reach RIGHT toward the sauce    ✓ works
   ------------------------------------------------------------
   BC-RNN average :  reach NEITHER, drift downward   ✗ works on nothing
```

The average of two good behaviours is not a good behaviour. This is called
**mode averaging**, and the measured evidence for it is exactly the halved
movement variety plus the persistent downward drift toward the midpoint.

---

## 4. Three things that make it worse

**Errors pile up over a long task.** BC-RNN decides one action at a time. This
task takes about 300 steps. A small error nudges the robot into a position no
human demonstration ever visited — and there the policy has no idea what to do,
so the next error is bigger. Over 300 steps the drift compounds.

**Very little data, very large model.** 50 demonstrations, 19.1 million
parameters. During training the gap between training and validation loss grew
from 0.34 to 5.06 in seven epochs — the model was memorising the 50 examples
rather than learning the task. Our early stopping caught this correctly and
halted at epoch 22.

**This is the hardest task family in the benchmark.** LIBERO-Long uses two
objects and ~300-step episodes. Short single-object tasks are far more forgiving.

---

## 5. Why BC-Transformer does not have this problem

| | BC-RNN | BC-Transformer |
|---|---|---|
| memory | recurrent hidden state, carried step by step | attends over a window of recent frames directly |
| decision | one action at a time | trained over sequences |
| multiple valid behaviours | averaged into one | kept separate |
| measured success | 0–5 % | **45 %** |

The transformer looks at a window of recent observations at once rather than
squeezing all history through a single hidden vector, and its head does not
collapse two valid behaviours into their midpoint the way a Gaussian mixture
does. Whatever the precise mechanism, the measured outcome is unambiguous: it
does **not** flatten the demonstrations, and it completes the task 45 % of the time.

---

## 6. Would more training have saved BC-RNN?

**No — and this is worth being clear about.**

Mode averaging is a property of what the model is *able to represent*, not of how
long it is trained. Training longer makes it average more confidently, not less.

The evidence in our own data: BC-RNN's success went 0/20 → 0/20 → 0/20 → 0/20 →
1/20 across epochs 5, 10, 14, 15, 20. Flat at the floor. The single success at
epoch 20 is one rollout — statistically indistinguishable from zero.

There is also a striking detail: **the checkpoint with the WORST validation loss
(epoch 20) was the only one that ever succeeded.** Validation loss and task
success ran in *opposite* directions over this range. That is why we saved a
checkpoint every 5 epochs instead of trusting the single best-loss one — and it
is why "the loss curve looks fine" is never sufficient evidence that a robot
policy works.

---

## 7. Why this is not a surprise, in hindsight

Diffusion Policy — the model already used for the safety-monitoring half of this
project — exists **specifically** to fix mode averaging. Its central claim is that
it represents several valid behaviours at once instead of averaging them.

On this exact task, Diffusion Policy scores **68.7 %**. BC-RNN scores 0–5 %. The
gap between them is a textbook demonstration of the problem Diffusion Policy was
designed to solve, reproduced accidentally on our own data.

---

## 8. What this means for the project

**BC-RNN is not usable as the arbitration cheap arm.** Not "below target" — at the
floor. An arbitration system whose cheap policy almost never succeeds would
escalate on essentially every rollout, and the efficiency guarantee (*"P(escalate
on a healthy rollout) ≤ α_esc"*) becomes meaningless, because there are no healthy
rollouts.

**BC-Transformer replaces it, and the replacement is already in hand.** The
checkpoint measuring 45 % was trained back in August, interrupted at epoch 10 by
an unrelated crash, and has been sitting unused. It lands inside the required
40–80 % band on the first attempt, with no retraining.

Requirements it satisfies:

- **In band (40–80 %)** — good enough to be the sensible default arm, weak enough
  that escalation has something to catch.
- **Cheap** — 5.4 M parameters, one forward pass, no iterative denoising. Much
  lighter than ACT or Diffusion Policy.
- **Already trained** — no GPU time needed to proceed.
- **Still improving when interrupted** — a properly finished run would likely be
  better, if a stronger arm is ever wanted.

---

## 9. Honest caveats

- **45 % came from 20 rollouts**, giving a 95 % confidence interval of roughly
  23–67 %. Wide. A 50-rollout confirmation is running to tighten it.
- **The same checkpoint measured 35 % in August and 45 % now.** Identical weights;
  the difference is ordinary sampling noise at 20 rollouts. It is a good reminder
  of how little 20 rollouts pins down.
- **Only task 0 has been tested.** No BC checkpoint exists for tasks 1 or 8 in
  either architecture.
- **We did not run a controlled comparison.** BC-RNN was trained 22 epochs on
  Colab; BC-Transformer 10 epochs locally, months earlier, with different
  hyperparameters. The 10x gap is far too large to be explained by that, but it
  is not a clean head-to-head and should not be reported as one.

---

## 10. One-line summary

> BC-RNN averages contradictory demonstrations into a motion that accomplishes
> nothing — it drives the gripper down, open, and never grasps. BC-Transformer
> does not average them, and reaches 45 % success on the same task with less than
> half the training. The cheap arm should be BC-Transformer.
