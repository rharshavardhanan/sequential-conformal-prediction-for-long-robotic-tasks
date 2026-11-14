"""Throughput profile for Push fine-tuning: collection-only vs update cost at several batch sizes / n_envs."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify_subskills import MODELS, _buffered_torch_load  # noqa: E402

import gymnasium as gym  # noqa: E402
import panda_gym  # noqa: E402,F401
from sb3_contrib import TQC  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv  # noqa: E402

SRC = os.path.join(MODELS, "enaitzb__TQC-PandaPush-v3")
_buffered_torch_load()

if __name__ == "__main__":
    for n_envs, batch, grad_steps, starts in [(1, 2048, 1, 10**9), (1, 2048, 1, 0), (1, 512, 1, 0),
                                              (8, 1024, 2, 10**9), (8, 1024, 2, 0)]:
        venv = make_vec_env("PandaPush-v3", n_envs=n_envs, seed=1,
                            vec_env_cls=SubprocVecEnv if n_envs > 1 else None)
        venv = VecNormalize.load(os.path.join(SRC, "vec_normalize.pkl"), venv)
        m = TQC.load(os.path.join(SRC, "TQC-PandaPush-v3.zip"), env=venv, device="cuda",
                     custom_objects={"learning_rate": 1e-4, "lr_schedule": lambda _: 1e-4,
                                     "batch_size": batch, "gradient_steps": grad_steps,
                                     "learning_starts": 10**9})
        m.tensorboard_log = None
        m.verbose = 0
        m.learn(600, reset_num_timesteps=True)          # warmup + buffer
        m.learning_starts = starts
        t = time.time()
        m.learn(3000, reset_num_timesteps=False)
        dt = time.time() - t
        print("n_envs=%d batch=%d grad_steps=%d updates=%s  -> %.0f env-steps/s" % (
            n_envs, batch, grad_steps, "off" if starts else "on", 3000 / dt), flush=True)
        venv.close()
