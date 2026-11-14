"""D.5 environment fingerprint — enforced at every session start.

Batching across nights is safe only if the data-generating distribution is
identical between sessions. Nothing in the stack raises an error when it is not:
a pip upgrade, an OS update, or a different checkpoint silently changes the
distribution and the conformal guarantee quietly stops holding.

This computes a hash over everything that defines the distribution. It is written
into the output file on the first session and re-checked on every subsequent one.
A mismatch aborts before a single rollout is appended.
"""

import hashlib
import json
import subprocess


def _ver(mod):
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "unknown"))
    except Exception as e:
        return f"IMPORT_FAIL:{type(e).__name__}"


def git_revision():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        rev = r.stdout.strip() or "unknown"
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return rev + ("-dirty" if d.stdout.strip() else "")
    except Exception:
        return "unknown"


def file_sha256(path, _chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_chunk), b""):
            h.update(b)
    return h.hexdigest()


def array_sha256(arr):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build(policy_repo, policy_sha, init_pool_sha, env_config, perturbation=None):
    """Return (fields: dict[str,str], digest: str)."""
    f = {
        "policy_repo": policy_repo,
        "policy_sha256": policy_sha,
        "init_pool_sha256": init_pool_sha,
        "env_config": json.dumps(env_config, sort_keys=True),
        "perturbation": json.dumps(perturbation or {}, sort_keys=True),
        "git_revision": git_revision(),
        "torch": _ver("torch"),
        "numpy": _ver("numpy"),
        "mujoco": _ver("mujoco"),
        "robosuite": _ver("robosuite"),
        "lerobot": _ver("lerobot"),
        "python": _ver("platform") and __import__("platform").python_version(),
    }
    digest = hashlib.sha256(
        json.dumps(f, sort_keys=True).encode()
    ).hexdigest()
    return f, digest


def compare(stored, current):
    """Return list of (key, stored, current) for differing keys."""
    keys = set(stored) | set(current)
    return [
        (k, stored.get(k, "<absent>"), current.get(k, "<absent>"))
        for k in sorted(keys)
        if stored.get(k) != current.get(k)
    ]
