"""Fine-tune the best Push checkpoint (enaitzb/TQC-PandaPush-v3: 125k env steps, measured 0.715).

Smoke test (logged in the report): resuming with the checkpoint's own settings (n_envs=1,
batch 2048, lr 1e-3, empty replay buffer -- the buffer is not stored in the checkpoint)
collapsed eval success 0.78 -> 0.18 within 3k steps, at 24 env-steps/s.  Profiling showed
~30 gradient updates/s regardless of batch size (HER sampling dominated), so this version:
  * 8 subprocess envs, 2 gradient steps per 8 env steps (update-to-data 0.25)
  * PREFILL 50k env steps with the current (stochastic) policy before any update
  * lr 1e-4, batch 1024; everything else from the checkpoint (gamma 0.95, tau 0.05,
    3x512, 2 critics, top-2 quantile drop, HER future k=4, VecNormalize stats continue updating)
Selection: every CHUNK env steps, 100 deterministic eval episodes (seeds 170000..170099,
frozen normalisation copy); keep the best; stop at eval >= 0.90 or MAX_EXTRA env steps.
The chosen checkpoint is then verified on FRESH seeds (verify_push_ft.py), bar 0.80.
"""
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify_subskills import MODELS, RESULTS, _buffered_torch_load  # noqa: E402

import gymnasium as gym  # noqa: E402
import panda_gym  # noqa: E402,F401
from sb3_contrib import TQC  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

N_ENVS, BATCH = 8, 1024
GRAD_STEPS = int(os.environ.get("FT_GRAD_STEPS", 2))
LR = float(os.environ.get("FT_LR", 1e-4))
PREFILL = int(os.environ.get("FT_PREFILL", 50_000))
RUN = os.environ.get("FT_RUN", "")                      # "" = run 1, "2" = run 2 (resumed)
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
MAX_EXTRA = int(sys.argv[2]) if len(sys.argv) > 2 else 1_200_000
EVAL_N, EVAL_SEED0, TARGET = 100, 170_000, 0.90
SRC = os.path.join(MODELS, "enaitzb__TQC-PandaPush-v3")
SRC_ZIP = os.environ.get("FT_SRC_ZIP", os.path.join(SRC, "TQC-PandaPush-v3.zip"))
SRC_VN = os.environ.get("FT_SRC_VN", os.path.join(SRC, "vec_normalize.pkl"))
OUT = os.path.join(MODELS, "push_ft" + RUN)
LOGF = os.path.join(RESULTS, "push_finetune_log%s.json" % (("_run" + RUN) if RUN else ""))


def main():
    os.makedirs(OUT, exist_ok=True)
    _buffered_torch_load()
    venv = make_vec_env("PandaPush-v3", n_envs=N_ENVS, seed=190_000, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize.load(SRC_VN, venv)
    venv.training = True
    model = TQC.load(SRC_ZIP, env=venv, device="cuda",
                     custom_objects={"learning_rate": LR, "lr_schedule": lambda _: LR, "batch_size": BATCH,
                                     "gradient_steps": GRAD_STEPS, "learning_starts": 10 ** 9})
    model.tensorboard_log = None
    model.verbose = 0
    eval_env = gym.make("PandaPush-v3")

    def evaluate():
        vn = copy.deepcopy(venv)
        vn.training = False
        succ = []
        for i in range(EVAL_N):
            obs, _ = eval_env.reset(seed=EVAL_SEED0 + i)
            done = False
            while not done:
                a, _ = model.predict(vn.normalize_obs({k: v.copy() for k, v in obs.items()}), deterministic=True)
                obs, _, term, trunc, info = eval_env.step(a)
                done = term or trunc
            succ.append(bool(info["is_success"]))
        return float(np.mean(succ))

    log = {"doc": __doc__, "run": RUN or "1", "src": SRC_ZIP, "lr": LR, "grad_steps": GRAD_STEPS, "prefill": PREFILL, "chunks": []}
    t0 = time.time()
    e0 = evaluate()
    model.learn(PREFILL, reset_num_timesteps=True)            # collection only (learning_starts huge)
    model.learning_starts = 0
    log["chunks"].append({"extra_steps": 0, "eval_success": e0, "prefill": PREFILL, "wall_min": (time.time() - t0) / 60})
    best, extra = (e0, 0), 0
    model.save(os.path.join(OUT, "step0.zip")); venv.save(os.path.join(OUT, "step0_vecnorm.pkl"))
    print("start eval=%.2f (prefill done, %.1f min)" % (e0, (time.time() - t0) / 60), flush=True)
    while extra < MAX_EXTRA:
        model.learn(CHUNK, reset_num_timesteps=False)
        extra += CHUNK
        sr = evaluate()
        tag = "step%d" % extra
        model.save(os.path.join(OUT, tag + ".zip")); venv.save(os.path.join(OUT, tag + "_vecnorm.pkl"))
        if sr > best[0]:
            best = (sr, extra)
        log["chunks"].append({"extra_steps": extra, "eval_success": sr, "wall_min": (time.time() - t0) / 60,
                              "ent_coef": float(model.log_ent_coef.exp().item()) if model.log_ent_coef is not None else None})
        log["best"] = {"eval_success": best[0], "extra_steps": best[1], "file": "step%d" % best[1]}
        json.dump(log, open(LOGF, "w"), indent=2)
        print("extra=%d eval=%.2f best=%.2f@%d (%.0f min)" % (extra, sr, best[0], best[1], (time.time() - t0) / 60), flush=True)
        if sr >= TARGET:
            break
    json.dump(log, open(LOGF, "w"), indent=2)
    print("DONE", log.get("best"), flush=True)
    venv.close()


if __name__ == "__main__":
    main()

