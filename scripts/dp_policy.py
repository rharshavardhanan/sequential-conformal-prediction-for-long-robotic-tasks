"""Adapter for a pretrained LeRobot diffusion policy driving a LIBERO env.

Runs in .venv-lerobot only (lerobot needs numpy 2.x).

The normalization here is applied by hand on purpose. These checkpoints store
their stats as `normalize_inputs.*` / `unnormalize_outputs.*` inside
model.safetensors (pre-0.4 LeRobot layout); lerobot 0.4.4 moved to external
processor files and DISCARDS those keys on load, leaving the policy with no
stats. A naive from_pretrained() therefore returns plausible-looking 7-vectors
that are pure garbage. Keeping it explicit also means the action space the
monitor is built on is stated, not assumed.

Verified on yzembodied/libero10_task0_diffusion: unnormalized actions span
[-0.938, 0.938] with the gripper at exactly +/-1.0 -- LIBERO's native 7-dim
delta space, so `action[:3] * 0.05` gives the commanded setpoint directly.
"""

import numpy as np
import torch

# LIBERO stores agentview bottom-up (macros_image_convention: opengl) and the
# LeRobot conversions flip it. Confirmed empirically: unflipped, the robot hangs
# from the ceiling, and rollouts score 0.
FLIP_IMAGES = True


def minmax_norm(x, lo, hi):
    return ((x - lo) / (hi - lo)) * 2.0 - 1.0


def minmax_denorm(x, lo, hi):
    return (x + 1.0) / 2.0 * (hi - lo) + lo


class DiffusionPolicyRunner:
    """Wraps a LeRobot DiffusionPolicy so it consumes raw LIBERO observations."""

    def __init__(self, repo, device="mps", flip_images=FLIP_IMAGES):
        from huggingface_hub import hf_hub_download
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from safetensors.torch import load_file

        self.repo = repo
        self.device = torch.device(device)
        self.flip = flip_images
        self.policy = DiffusionPolicy.from_pretrained(repo).to(self.device).eval()

        raw = load_file(hf_hub_download(repo, "model.safetensors"))
        S = {
            k: v.float().to(self.device)
            for k, v in raw.items()
            if k.startswith(("normalize_", "unnormalize_"))
        }
        if not S:
            raise RuntimeError(f"{repo} has no in-model normalization stats")
        self.stats = S
        self.img_mean = S["normalize_inputs.buffer_observation_images_image.mean"]
        self.img_std = S["normalize_inputs.buffer_observation_images_image.std"]
        self.st_lo = S["normalize_inputs.buffer_observation_state.min"]
        self.st_hi = S["normalize_inputs.buffer_observation_state.max"]
        self.jt_lo = S["normalize_inputs.buffer_observation_state_joint.min"]
        self.jt_hi = S["normalize_inputs.buffer_observation_state_joint.max"]
        self.act_lo = S["unnormalize_outputs.buffer_action.min"]
        self.act_hi = S["unnormalize_outputs.buffer_action.max"]

    def reset(self):
        """Clear the n_obs_steps observation queue between rollouts."""
        self.policy.reset()

    def _img(self, a):
        if self.flip:
            a = a[::-1]
        t = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        t = t.permute(2, 0, 1).float() / 255.0
        return ((t - self.img_mean) / self.img_std).unsqueeze(0)

    @staticmethod
    def build_state(obs):
        """[eef_pos(3), axis-angle(3), gripper_qpos(2)] -- validated against the
        dataset's recorded min/max."""
        from robosuite.utils.transform_utils import quat2axisangle

        return np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32)

    @torch.no_grad()
    def act(self, obs):
        """Raw LIBERO obs dict -> 7-dim action in LIBERO's native [-1, 1] space."""
        st = torch.from_numpy(self.build_state(obs)).to(self.device)
        jt = torch.from_numpy(obs["robot0_joint_pos"].astype(np.float32)).to(
            self.device
        )
        batch = {
            "observation.images.image": self._img(obs["agentview_image"]),
            "observation.images.wrist_image": self._img(
                obs["robot0_eye_in_hand_image"]
            ),
            "observation.state": minmax_norm(st, self.st_lo, self.st_hi).unsqueeze(0),
            "observation.state.joint": minmax_norm(
                jt, self.jt_lo, self.jt_hi
            ).unsqueeze(0),
        }
        a = self.policy.select_action(batch).squeeze(0)
        return minmax_denorm(a, self.act_lo, self.act_hi).cpu().numpy()

    def fingerprint(self):
        """Identity of the exact weights, for stamping into every rollout.

        CP's guarantee assumes one data-generating process; this makes a
        mid-collection swap detectable after the fact rather than silent.
        """
        import hashlib

        h = hashlib.sha256()
        for k in sorted(self.policy.state_dict()):
            h.update(k.encode())
            h.update(self.policy.state_dict()[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()
