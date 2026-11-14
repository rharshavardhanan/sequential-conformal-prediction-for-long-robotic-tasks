"""Adapter for lerobot DiffusionPolicy checkpoints.

Wraps the EXISTING `dp_policy.DiffusionPolicyRunner` rather than reimplementing
it. That runner carries hard-won behaviour -- notably the manual normalization
that lerobot 0.4.4 would otherwise discard, leaving the policy with no stats and
returning plausible-looking garbage. None of that is duplicated here.
"""

import numpy as np

from scripts.policies.base import PolicyAdapter, PolicyOutput

# Upstream misspells tasks 3 and 8 as "libeo".
def task_repo(task_id):
    stem = "libeo10" if task_id in (3, 8) else "libero10"
    return f"yzembodied/{stem}_task{task_id}_diffusion"


class DiffusionAdapter(PolicyAdapter):
    """DiffusionPolicy: 256px images, internal 5-step obs queue, 8-action chunk."""

    def __init__(self, task_id=0, repo=None, device="mps", name=None):
        from scripts.dp_policy import DiffusionPolicyRunner

        self._name = name or f"diffusion_task{task_id}"
        self._repo = repo or task_repo(task_id)
        self._runner = DiffusionPolicyRunner(self._repo, device=device)

    def reset(self):
        self._runner.reset()

    def act(self, obs):
        action, chunk, is_new = self._runner.act_with_chunk(obs)
        return PolicyOutput(action=action, chunk=chunk, chunk_len=int(len(chunk)),
                            is_new=bool(is_new), meta={"repo": self._repo})

    def fingerprint(self):
        return self._runner.fingerprint()

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return "Diffusion Policy"

    @property
    def action_horizon(self):
        return int(self._runner.n_action_steps)

    @property
    def compute_class(self):
        return "expensive"

    @property
    def observation_spec(self):
        return {
            "agentview_image": "(256,256,3) uint8",
            "robot0_eye_in_hand_image": "(256,256,3) uint8",
            "robot0_eef_pos": "(3,)",
            "robot0_eef_quat": "(4,) -> axis-angle",
            "robot0_gripper_qpos": "(2,)",
            "robot0_joint_pos": "(7,)",
            "_history": "5 obs steps, held inside lerobot queues",
        }
