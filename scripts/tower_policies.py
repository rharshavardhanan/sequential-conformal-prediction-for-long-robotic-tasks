"""Closed-loop adapters for the tower policies, plus an honest evaluator.

Both arms consume the 32-dim Route-B observation (proprioception + cube
positions) and emit 7 absolute joint targets + a grasp bool. The environment
receives exactly one action per control step, as always.
"""

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "roboturk"))
GRIP = 7


class TowerPolicy:
    """Common shape: reset(), act(obs32) -> (joint_target7, grip)."""

    def __init__(self, ckpt, device="mps"):
        self.ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.dev = torch.device(device)
        self.s_mu, self.s_sd = self.ck["s_mu"], self.ck["s_sd"]
        self.a_mu, self.a_sd = self.ck["a_mu"], self.ck["a_sd"]
        self.cfg = self.ck["cfg"]
        self.name = self.cfg["arch"]
        self.path = str(ckpt)

    def _norm(self, o):
        return ((np.asarray(o) - self.s_mu) / self.s_sd).astype(np.float32)

    def _denorm(self, a):
        return np.asarray(a) * self.a_sd + self.a_mu

    def fingerprint(self):
        import hashlib
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()


class BCRNNPolicy(TowerPolicy):
    """Cheap arm. LSTM, so it must be stepped in order carrying hidden state."""

    compute_class = "cheap"

    def __init__(self, ckpt="data/tower/tower_bcrnn.pt", device="mps"):
        super().__init__(ckpt, device)
        from scripts.tower_train import BCRNN, gmm_mean
        self._gmm_mean = gmm_mean
        self.m = BCRNN(self.cfg["sdim"], self.cfg["adim"]).to(self.dev)
        self.m.load_state_dict(self.ck["model"]); self.m.eval()
        self.hx = None

    def reset(self):
        self.hx = None

    @torch.no_grad()
    def act(self, obs):
        s = torch.tensor(self._norm(obs)[None, None]).to(self.dev)
        mu, _, logit, self.hx = self.m(s, self.hx)
        a = self._denorm(self._gmm_mean(mu, logit)[0, 0].cpu().numpy())
        return a[:GRIP], float(a[GRIP] > 0.5)


class ACTPolicy(TowerPolicy):
    """Expensive arm. Stateless over a fixed history window; emits a chunk and
    the first action of it is executed."""

    compute_class = "expensive"

    def __init__(self, ckpt="data/tower/tower_act.pt", device="mps",
                 exec_len=None):
        super().__init__(ckpt, device)
        from train_act import ACT
        c = self.cfg
        self.h = int(c["history"])
        self.m = ACT(c["sdim"], c["adim"], chunk=c["chunk"], d=c["dim"],
                     ff=c["ff"], n_enc=c["enc_layers"], n_dec=c["dec_layers"],
                     n_cvae=c["cvae_layers"], dropout=c["dropout"],
                     history=self.h).to(self.dev)
        self.m.load_state_dict(self.ck["model"]); self.m.eval()
        self.exec_len = int(exec_len or c["chunk"])
        self.buf, self.queue, self.chunk = [], [], None

    def __init_exec__(self):
        pass

    def reset(self):
        self.buf, self.queue, self.chunk = [], [], None

    @torch.no_grad()
    def act(self, obs):
        """Executes the chunk it planned, re-querying every `exec_len` steps.

        This is what ACT is FOR. Re-planning every step and throwing away 15 of
        16 predicted actions gives up chunking's main benefit: each inference
        commits to a short open-loop burst, so per-step errors have fewer
        opportunities to compound into a state the policy has never seen.
        Measured here: executing 1 action per inference gives 0/15 towers.
        """
        self.buf.append(self._norm(obs))
        self.buf = self.buf[-self.h:]
        if not self.queue:
            pad = [self.buf[0]] * (self.h - len(self.buf)) + self.buf
            s = torch.tensor(np.stack(pad)[None]).to(self.dev)
            chunk = self.m(s)[0][0].cpu().numpy()
            self.chunk = np.array([self._denorm(x) for x in chunk])
            self.queue = list(self.chunk[: self.exec_len])
        a = self.queue.pop(0)
        return a[:GRIP], float(a[GRIP] > 0.5)


def evaluate(policy, n=20, max_steps=800, seed=0, device="mps", verbose=True):
    """Closed loop: policy -> joint targets -> controller -> MuJoCo -> obs.
    Reports the only number that matters: does it actually build the tower."""
    from scripts.tower_collect import make_env, observation
    from scripts.tower_control import joint_action

    env = make_env(seed=seed)
    ok, heights, steps = 0, [], []
    for i in range(n):
        env.reset(); policy.reset()
        for t in range(max_steps):
            q, g = policy.act(observation(env))
            env.step(joint_action(env, q, -1.0 if g < 0.5 else 1.0))
            if env._check_success():
                break
        s = env._check_success()
        ok += s; heights.append(env.tower_height()); steps.append(t + 1)
        if verbose:
            print(f"    ep{i:3d}  height {env.tower_height()}  "
                  f"steps {t+1:4d}  {'SUCCESS' if s else 'fail'}", flush=True)
    env.close()
    return {"policy": policy.name, "n": n, "success": ok, "rate": ok / n,
            "mean_height": float(np.mean(heights)), "mean_steps": float(np.mean(steps)),
            "heights": heights}
