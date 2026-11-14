"""Re-emit eval JSONs with ACTOR parameters separated from total policy parameters.

`sum(policy.parameters())` counts the actor plus 2 critics plus 2 target critics. Only
the actor runs at inference, so reporting the total as "n_params" overstates deployed
cost by ~5x and would corrupt Phase 9's compute ratio between the expert and the cheap
policy. This rewrites existing eval JSONs in place, idempotently: `n_params_actor` is
read from the checkpoint next to each file, `total_params` is preserved, and the
ambiguous `n_params` key is dropped.

    cd handout_mujoco
    PYTHONPATH=live ../.venv-lerobot/bin/python analysis/fix_param_accounting.py
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))

from envs import PushOnly, ReachOnly
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

TRAIN_ROOT = ROOT / "results" / "mujoco_train"


def counts(ckpt, env_cls):
    m = TQC.load(ckpt, env=DummyVecEnv([env_cls]), device="cpu")
    return (int(sum(p.numel() for p in m.actor.parameters())),
            int(sum(p.numel() for p in m.policy.parameters())))


def main():
    targets = sorted(TRAIN_ROOT.glob("*/*/push_eval.json")) + \
              sorted(TRAIN_ROOT.glob("*/*/reach_eval.json"))
    if not targets:
        print("no eval JSONs found")
        return
    for f in targets:
        env_cls = PushOnly if f.name.startswith("push") else ReachOnly
        ckpt = next((f.parent / n for n in ("final_model.zip", "best_model.zip")
                     if (f.parent / n).exists()), None)
        if ckpt is None:
            print(f"  SKIP {f.relative_to(ROOT)} -- no checkpoint next to it")
            continue
        d = json.loads(f.read_text())
        actor, total = counts(ckpt, env_cls)
        before = d.pop("n_params", None)
        d["n_params_actor"] = actor
        d["total_params"] = total
        f.write_text(json.dumps(d, indent=2))
        note = f"was n_params={before:,}" if before is not None else "already corrected"
        print(f"  {f.relative_to(ROOT)}: actor {actor:,}  total {total:,}   ({note})")


if __name__ == "__main__":
    main()
