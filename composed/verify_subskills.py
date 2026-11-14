"""PART 1 -- verify candidate sub-skill checkpoints independently (success rate measured, not inferred).

Each checkpoint is rolled out for 50 episodes in the SPARSE v3 env of its task with
gymnasium's native TimeLimit (max_episode_steps) and success-termination, so steps-to-
success is directly measurable.  Dense-trained checkpoints are evaluated on the sparse
env: same observations and success test, only the reward differs.  Deterministic actions.
Acceptance bar (pre-registered, from the brief): success rate >= 0.80.
"""
import io
import json
import os
import pickle
import sys

import numpy as np
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402  (wilson, torch-load shim via ExpertPolicy import side effects)

import gymnasium as gym  # noqa: E402
import panda_gym  # noqa: E402,F401
import torch  # noqa: E402

MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
N = 50
SEED0 = 110_000

CANDIDATES = [  # (task env, repo, zip filename, algo)
    ("PandaReach-v3", "chencliu/tqc-PandaReach-v3", "tqc-PandaReach-v3.zip", "tqc"),
    ("PandaReach-v3", "ThomasSimonini/a2c-PandaReachDense-v3", "a2c-PandaReachDense-v3.zip", "a2c"),
    ("PandaReach-v3", "sagarsdesai/a2c-PandaReachDense-v3", "a2c-PandaReachDense-v3.zip", "a2c"),
    ("PandaPush-v3", "enaitzb/TQC-PandaPush-v3", "TQC-PandaPush-v3.zip", "tqc"),
    ("PandaPush-v3", "enaitzb/TQC_no_envs-PandaPush-v3", "TQC-PandaPush-v3.zip", "tqc"),
    ("PandaPush-v3", "Mriganka1999/PandaPushDense-v3usingSAC-PandaPushDense-v3", "sac-PandaPushDense-v3.zip", "sac"),
    ("PandaPush-v3", "Mriganka1999/PandaPushDense-v3usingTD3-PandaPushDense-v3", "td3-PandaPushDense-v3.zip", "td3"),
]
if "--slide" in sys.argv:
    CANDIDATES.append(("PandaSlide-v3", "chencliu/tqc-PandaSlide-v3", "tqc-PandaSlide-v3.zip", "tqc"))
    CANDIDATES.append(("PandaPush-v3", "enaitzb/TQC-PandaPush-v3#no_load_env", "TQC_no_load_env-PandaPush-v3.zip", "tqc"))


def _buffered_torch_load():
    if getattr(torch.load, "_buffered", False):
        return
    orig = torch.load

    def _load(f, *a, **kw):
        if hasattr(f, "read") and not isinstance(f, io.BytesIO):
            f = io.BytesIO(f.read())
        return orig(f, *a, **kw)
    _load._buffered = True
    torch.load = _load


class Policy:
    """Generic SB3 / sb3-contrib checkpoint + optional VecNormalize obs stats."""

    def __init__(self, env_id, repo, zipname, algo):
        _buffered_torch_load()
        import stable_baselines3 as sb3
        import sb3_contrib
        cls = {"tqc": sb3_contrib.TQC, "sac": sb3.SAC, "td3": sb3.TD3, "a2c": sb3.A2C}[algo]
        if repo.startswith("local:"):               # locally fine-tuned checkpoint: "local:<vecnorm.pkl>"
            z, vn = zipname, repo[len("local:"):]
        else:
            repo = repo.split("#")[0]
            d = os.path.join(MODELS, repo.replace("/", "__"))
            z = hf_hub_download(repo, zipname, local_dir=d)
            try:
                vn = hf_hub_download(repo, "vec_normalize.pkl", local_dir=d)
            except Exception:
                vn = None
        kw = dict(device="cpu", custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                                 "clip_range": lambda _: 0.0})
        try:
            self.model = cls.load(z, **kw)
        except Exception as e:                      # HER buffers need an env to load
            if "env" not in str(e):
                raise
            self.model = cls.load(z, env=gym.make(env_id), **kw)
        self.vn = None
        if vn:
            with open(vn, "rb") as fh:
                self.vn = pickle.load(fh)
            self.vn.training = False
        self.algo, self.repo = algo, repo

    def __call__(self, obs):
        o = obs
        if self.vn is not None and getattr(self.vn, "norm_obs", False):
            o = self.vn.normalize_obs({k: v.copy() for k, v in obs.items()})
        a, _ = self.model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def params(self):
        net = getattr(self.model, "actor", None) or self.model.policy
        return int(sum(p.numel() for p in net.parameters()))


def verify(env_id, pol, n=N, seed0=SEED0):
    env = gym.make(env_id)
    succ, lens = [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        t, done, ok = 0, False, False
        while not done:
            obs, _, term, trunc, info = env.step(pol(obs))
            t += 1
            ok = bool(info["is_success"])
            done = term or trunc
        succ.append(ok)
        lens.append(t)
    lens, succ = np.array(lens), np.array(succ)
    k = int(succ.sum())
    return {"success": k, "n": n, "success_rate": k / n, "wilson95": C.wilson(k, n),
            "max_episode_steps": int(env.spec.max_episode_steps),
            "steps_to_success_mean": float(lens[succ].mean()) if k else None,
            "steps_to_success_median": float(np.median(lens[succ])) if k else None,
            "steps_to_success_p90": float(np.quantile(lens[succ], 0.9)) if k else None,
            "steps_to_success_max": int(lens[succ].max()) if k else None}


if __name__ == "__main__":
    path = os.path.join(RESULTS, "part1_verify.json")
    report = json.load(open(path)) if os.path.exists(path) else {"bar": 0.80, "n": N, "candidates": {}}
    for env_id, repo, zipname, algo in CANDIDATES:
        if repo in report["candidates"] and "--force" not in sys.argv:
            continue
        try:
            pol = Policy(env_id, repo, zipname, algo)
            r = verify(env_id, pol)
            r.update({"task": env_id, "algo": algo, "actor_params": pol.params(), "zip": zipname,
                      "vec_normalize": pol.vn is not None, "passes_bar": r["success_rate"] >= 0.80})
        except Exception as e:
            r = {"task": env_id, "algo": algo, "error": repr(e)[:400], "passes_bar": False}
        report["candidates"][repo] = r
        print(repo, {k: v for k, v in r.items() if k not in ("zip",)})
        json.dump(report, open(path, "w"), indent=2)
    # Borderline re-measurement: enaitzb TQC scored 38/50 = 0.76.  Re-measured on 200 FRESH
    # seeds; the bar is unchanged (point estimate >= 0.80 required).
    if "--recheck" in sys.argv:
        pol = Policy("PandaPush-v3", "enaitzb/TQC-PandaPush-v3", "TQC-PandaPush-v3.zip", "tqc")
        r = verify("PandaPush-v3", pol, n=200, seed0=120_000)
        r["passes_bar"] = r["success_rate"] >= 0.80
        report["recheck_enaitzb_TQC-PandaPush-v3_n200"] = r
        print("recheck n=200", r)
        json.dump(report, open(path, "w"), indent=2)
