"""Verify the fine-tuned Push checkpoint on FRESH seeds, same protocol and bar as Part 1.

Chosen checkpoint = best eval-seed checkpoint from push_finetune_log.json (selection seeds
170000..170099).  Verification: 200 fresh seeds (200000..200199), sparse PandaPush-v3,
native TimeLimit + success-termination, deterministic.  Bar unchanged: success >= 0.80.
The ORIGINAL checkpoint (step0 = enaitzb/TQC unmodified) is re-run on the same fresh seeds
for a paired comparison.
"""
import json
import os

import numpy as np
from scipy.stats import binomtest

from verify_subskills import MODELS, RESULTS, Policy, verify

LOG1 = json.load(open(os.path.join(RESULTS, "push_finetune_log_run1.json")))
LOG = json.load(open(os.path.join(RESULTS, "push_finetune_log_run2.json")))
best = LOG["best"]["file"]                  # run 2 best (0.90) > run 1 best (0.78 = untrained start)
D = os.path.join(MODELS, "push_ft2")
N, SEED0 = 200, 200_000


def run(tag):
    if tag == "original":                    # enaitzb/TQC-PandaPush-v3 exactly as downloaded
        pol = Policy("PandaPush-v3", "enaitzb/TQC-PandaPush-v3", "TQC-PandaPush-v3.zip", "tqc")
    else:
        pol = Policy("PandaPush-v3", "local:" + os.path.join(D, tag + "_vecnorm.pkl"), os.path.join(D, tag + ".zip"), "tqc")
    import gymnasium as gym
    env = gym.make("PandaPush-v3")
    s = []
    for i in range(N):
        obs, _ = env.reset(seed=SEED0 + i)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(pol(obs))
            done = term or trunc
        s.append(bool(info["is_success"]))
    return pol, np.array(s)


pol, s_best = run(best)
r = verify("PandaPush-v3", pol, n=N, seed0=SEED0)
_, s_orig = run("original")
n10, n01 = int(np.sum(s_best & ~s_orig)), int(np.sum(~s_best & s_orig))
out = {"doc": __doc__, "chosen": best, "chosen_dir": "push_ft2", "run1_curve": [(c["extra_steps"], c["eval_success"]) for c in LOG1["chunks"]], "eval_seed_success": LOG["best"]["eval_success"],
       "fresh_verification": r, "passes_bar": r["success_rate"] >= 0.80,
       "original_same_fresh_seeds": float(s_orig.mean()),
       "paired_vs_original": {"ft_only": n10, "orig_only": n01,
                              "mcnemar_exact_p": float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0},
       "actor_params": pol.params(),
       "learning_curve": [(c["extra_steps"], c["eval_success"]) for c in LOG["chunks"]]}
json.dump(out, open(os.path.join(RESULTS, "push_ft_verify.json"), "w"), indent=2, default=bool)
print(json.dumps({k: v for k, v in out.items() if k not in ("doc", "learning_curve")}, indent=1, default=bool))

