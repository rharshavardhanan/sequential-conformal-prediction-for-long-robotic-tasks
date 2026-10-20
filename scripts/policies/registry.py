"""The ONLY place a policy name is branched on.

Every entry declares what is *measured*, never what is hoped for. `success` is
None until a real evaluation has produced a number, and `available` is computed
from whether the checkpoint actually exists on disk. The GUI reads these fields
directly, which is what stops it inventing "ACT: 85%" for a policy that has
never been trained.

Measured success rates, task 0:
  diffusion_task0  0.687  n=150  (data/rollouts/task0_v10.hdf5)
  bct_task0        0.44   n=50   (data/results/bct_eval_task0_n50.json)
  bcrnn_task0      0.05   n=20   (data/results/bcrnn_eval_task0.json) -- unusable
  act_task0        None          -- no checkpoint, never trained
"""

import os
from dataclasses import dataclass, field

BCT_CKPT = "data/checkpoints_bct/bct_task0_ep010.pth"
BCRNN_CKPT = "data/checkpoints/bcrnn_task0_ep020.pth"


@dataclass
class PolicyEntry:
    name: str
    display_name: str
    family: str                       # lerobot | libero
    compute_class: str                # cheap | expensive
    task_id: int
    success: "float | None"           # MEASURED, or None. Never estimated.
    success_n: "int | None"
    checkpoint: "str | None"          # local path, or None for HF-hosted
    note: str = ""
    _factory: object = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        """HF-hosted policies are assumed fetchable; local ones must exist."""
        return self.checkpoint is None or os.path.exists(self.checkpoint)

    def build(self, device="mps"):
        if not self.available:
            raise SystemExit(
                f"[abort] policy {self.name!r} is not available: "
                f"checkpoint {self.checkpoint} does not exist")
        return self._factory(self, device)


def _build_diffusion(e, device):
    from scripts.policies.diffusion import DiffusionAdapter
    return DiffusionAdapter(task_id=e.task_id, device=device, name=e.name)


def _build_libero(e, device):
    from scripts.policies.libero_bc import LiberoBCAdapter
    return LiberoBCAdapter(e.checkpoint, task_id=e.task_id, device=device,
                           name=e.name, display_name=e.display_name,
                           compute_class=e.compute_class)


def _build_act(e, device):
    raise SystemExit(
        "[abort] ACT has no checkpoint in this project. lerobot 0.4.4 provides "
        "ACTPolicy, but nothing has been trained, so there is nothing to load. "
        "Train one first; the adapter slot is reserved.")


REGISTRY = {}


def _add(**kw):
    e = PolicyEntry(**kw)
    REGISTRY[e.name] = e


for _t in (0, 1, 2, 3, 8):
    _add(name=f"diffusion_task{_t}", display_name="Diffusion Policy",
         family="lerobot", compute_class="expensive", task_id=_t,
         success=({0: 0.687, 1: 0.727, 8: 0.853}.get(_t)),
         success_n=(150 if _t in (0, 1, 8) else None),
         checkpoint=None, note="frozen pretrained, HF-hosted",
         _factory=_build_diffusion)

_add(name="bct_task0", display_name="BC-Transformer", family="libero",
     compute_class="cheap", task_id=0, success=0.44, success_n=50,
     checkpoint=BCT_CKPT, note="ep010, interrupted mid-training",
     _factory=_build_libero)

_add(name="bcrnn_task0", display_name="BC-RNN", family="libero",
     compute_class="cheap", task_id=0, success=0.05, success_n=20,
     checkpoint=BCRNN_CKPT, note="converged but unusable -- see "
                                 "docs/bc-rnn-vs-bc-transformer.md",
     _factory=_build_libero)

_add(name="act_task0", display_name="ACT", family="lerobot",
     compute_class="expensive", task_id=0, success=None, success_n=None,
     checkpoint="data/checkpoints_act/act_task0.pth",
     note="NOT TRAINED -- slot reserved", _factory=_build_act)


def get(name):
    if name not in REGISTRY:
        raise SystemExit(f"[abort] unknown policy {name!r}. "
                         f"Known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def build(name, device="mps"):
    return get(name).build(device=device)


def list_policies(task_id=None):
    return [e for e in REGISTRY.values()
            if task_id is None or e.task_id == task_id]
