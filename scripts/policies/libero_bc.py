"""Adapter for LIBERO/robomimic checkpoints -- BC-Transformer and BC-RNN.

These need no hydra and no BERT at run time:

  * the checkpoint pickles its own `cfg` (EasyDict) including `cfg.shape_meta`,
    so the policy is constructible from the file alone;
  * the task embedding is a FIXED vector per task, precomputed once into
    `data/task_embs_libero10_bert.npz` by a script run in `.venv`.

That is what keeps `transformers` and `hydra` out of `.venv-lerobot`, whose
package set is fingerprinted by `scripts/fingerprint.py`.

Image sizes differ from DiffusionPolicy: `shape_meta.all_shapes` asks for
[3,128,128], while the env renders 256. The `[::2, ::2]` downsample used here is
the same one `collect_rollouts.py` applies when storing frames.
"""

import hashlib

import numpy as np
import torch

from scripts.policies.base import PolicyAdapter, PolicyOutput

TASK_EMBS = "data/task_embs_libero10_bert.npz"
_POLICY_CLASSES = {
    "BCTransformerPolicy": "bc_transformer_policy",
    "BCRNNPolicy": "bc_rnn_policy",
}


def _sha256_file(path, _chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_chunk), b""):
            h.update(b)
    return h.hexdigest()


class LiberoBCAdapter(PolicyAdapter):
    """BC-Transformer / BC-RNN: 128px images, caller-held history, ONE action."""

    def __init__(self, checkpoint, task_id=0, device="mps", name=None,
                 display_name=None, compute_class="cheap",
                 task_embs=TASK_EMBS):
        import importlib

        self._ckpt = checkpoint
        self._task_id = int(task_id)
        self._device = torch.device(device)
        self._compute_class = compute_class

        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = sd["cfg"]
        ptype = str(cfg.policy.policy_type)
        if ptype not in _POLICY_CLASSES:
            raise ValueError(f"unsupported LIBERO policy_type {ptype!r}")
        mod = importlib.import_module(
            f"libero.lifelong.models.{_POLICY_CLASSES[ptype]}")
        cls = getattr(mod, ptype)

        self._policy = cls(cfg, cfg.shape_meta).to(self._device).eval()
        missing, unexpected = self._policy.load_state_dict(sd["state_dict"],
                                                           strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{checkpoint}: {len(missing)} missing / {len(unexpected)} "
                f"unexpected state_dict keys -- refusing to run a partially "
                f"loaded policy")

        self._cfg = cfg
        self._ptype = ptype
        self._seq_len = int(cfg.data.seq_len)
        self._key_map = dict(cfg.data.obs_key_mapping)
        self._img_keys = list(cfg.data.obs.modality["rgb"])
        self._img_size = int(cfg.shape_meta["all_shapes"][self._img_keys[0]][-1])
        self._low_keys = list(cfg.data.obs.modality["low_dim"])
        self._name = name or f"{ptype.replace('Policy','').lower()}_task{task_id}"
        self._display = display_name or (
            "BC-Transformer" if ptype == "BCTransformerPolicy" else "BC-RNN")
        self._sha = _sha256_file(checkpoint)

        z = np.load(task_embs, allow_pickle=True)
        self._task_emb = torch.from_numpy(
            np.asarray(z["embs"][self._task_id], np.float32)
        ).unsqueeze(0).to(self._device)

        # robomimic keeps a GLOBAL obs_key -> modality registry, and
        # ObsUtils.process_obs() reads it. LIBERO's own eval initialises it as a
        # side effect of building the dataset; we do it directly so no 13 GB of
        # demonstrations has to be opened just to run inference.
        import robomimic.utils.obs_utils as ObsUtils

        self._ObsUtils = ObsUtils
        ObsUtils.initialize_obs_utils_with_obs_specs(
            {"obs": {k: list(v) for k, v in cfg.data.obs.modality.items() if v}})

    def reset(self):
        """Clears the policy's internal latent queue. MUST run between episodes:
        BCTransformerPolicy accumulates history itself, so a missed reset leaks
        the previous episode's context into the next one."""
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @torch.no_grad()
    def act(self, obs):
        """One raw LIBERO obs -> one action.

        Shapes matter: `get_action` calls `preprocess_input(train_mode=False)`,
        which inserts the TIME dimension itself via unsqueeze(dim=1). So each
        tensor is handed over as (B=1, ...) with no time axis, and the policy's
        own `latent_queue` supplies the history -- this adapter holds none.
        """
        data = {"obs": {}, "task_emb": self._task_emb}
        for k in self._img_keys + self._low_keys:
            raw = np.asarray(obs[self._key_map[k]])
            if k in self._img_keys and raw.shape[0] != self._img_size:
                raw = raw[::2, ::2]      # env renders 256 for DiffusionPolicy
            t = self._ObsUtils.process_obs(
                torch.from_numpy(np.ascontiguousarray(raw)), obs_key=k).float()
            data["obs"][k] = t.unsqueeze(0).to(self._device)

        a = self._policy.get_action(data)
        a = np.asarray(a, np.float32).reshape(-1)[:7]
        return PolicyOutput(action=np.clip(a, -1.0, 1.0), chunk=None, chunk_len=1,
                            is_new=True, meta={"checkpoint": self._ckpt})

    def fingerprint(self):
        return self._sha

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return self._display

    @property
    def compute_class(self):
        return self._compute_class

    @property
    def observation_spec(self):
        return {
            "agentview_rgb": "(3,128,128) <- agentview_image[::2,::2]",
            "eye_in_hand_rgb": "(3,128,128) <- robot0_eye_in_hand_image[::2,::2]",
            "gripper_states": "(2,) <- robot0_gripper_qpos",
            "joint_states": "(7,) <- robot0_joint_pos",
            "task_emb": "(768,) precomputed BERT, frozen",
            "_history": f"{self._seq_len} steps, held by this adapter",
        }
