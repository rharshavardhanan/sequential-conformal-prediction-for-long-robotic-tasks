"""The cheap arm: a small MLP distilled from the composed expert, one net for both phases.

It sees the same 26-dim observation in REACH and PUSH and must do both jobs from ~6k
parameters. It is expected to be weak at reach (inferring a 3-D waypoint) and mediocre at
push. That is the point: escalations then fire in BOTH phases, which is what makes a
K=2 phase-conditioned calibration meaningful rather than decorative.

Input standardisation is stored WITH the weights. The cheap policy must see exactly the
transform it was distilled under, in training, in calibration and in the live demo.
"""

import numpy as np
import torch
from torch import nn

OBS_DIM, ACT_DIM = 26, 3


class CheapMLP(nn.Module):
    def __init__(self, hidden=(64, 64)):
        super().__init__()
        dims = [OBS_DIM, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(dims[-1], ACT_DIM), nn.Tanh()]  # tanh -> actions in [-1,1]
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
        self.register_buffer("obs_std", torch.ones(OBS_DIM))

    def forward(self, x):
        return self.net((x - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs26):
        x = torch.as_tensor(np.asarray(obs26, dtype=np.float32)).reshape(1, OBS_DIM)
        return self.forward(x).numpy().ravel().astype(np.float64)

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


class CheapMLPGaussian(nn.Module):
    """Heteroscedastic BC head: predicts a mean AND a per-dimension log-std.

    Candidate (c) for the informativeness gate. The std is what the policy itself says
    about how uncertain the demonstrated action was at this observation -- a quantity
    available at decision time with no extra forward passes, unlike an ensemble.
    """

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, hidden=(16, 16)):
        super().__init__()
        dims = [OBS_DIM, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(dims[-1], ACT_DIM)
        self.log_std = nn.Linear(dims[-1], ACT_DIM)
        self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
        self.register_buffer("obs_std", torch.ones(OBS_DIM))

    def forward(self, x):
        h = self.trunk((x - self.obs_mean) / self.obs_std)
        return torch.tanh(self.mu(h)), self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    @torch.no_grad()
    def act_and_std(self, obs26):
        x = torch.as_tensor(np.asarray(obs26, dtype=np.float32)).reshape(1, OBS_DIM)
        mu, ls = self.forward(x)
        return (mu.numpy().ravel().astype(np.float64),
                float(ls.exp().mean().item()))

    @torch.no_grad()
    def act(self, obs26):
        return self.act_and_std(obs26)[0]

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters())
                   - self.obs_mean.numel() - self.obs_std.numel())


def save(model, path, meta=None):
    if isinstance(model, CheapMLPGaussian):
        hidden = [m.out_features for m in model.trunk if isinstance(m, nn.Linear)]
        kind = "gaussian"
    else:
        hidden = [m.out_features for m in model.net if isinstance(m, nn.Linear)][:-1]
        kind = "deterministic"
    torch.save({"hidden": hidden, "kind": kind, "state_dict": model.state_dict(),
                "meta": meta or {}}, path)


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cls = CheapMLPGaussian if blob.get("kind") == "gaussian" else CheapMLP
    m = cls(tuple(blob["hidden"]))
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m, blob.get("meta", {})
