"""The one interface every policy family satisfies.

The simulator, monitor, logger and viewer see only `PolicyAdapter`. They never
learn which library produced an action. Adding a policy means adding an adapter
and a registry entry -- never an `if` in the control loop.

Two families exist today and they disagree about almost everything:

  DiffusionPolicy (lerobot)   256px images, 5-step internal queue, emits an
                              8-action CHUNK, no task embedding
  BC-Transformer  (LIBERO)    128px images, 10-step caller-held history, emits
                              ONE action, needs a BERT task embedding

`PolicyOutput` is what reconciles them: every adapter returns exactly one
executable action, and optionally the chunk it came from. The chunk is kept
rather than discarded because FAIL-Detect and the arbiter's chunk-boundary
switching both need it, and it is NOT reconstructable afterwards -- diffusion
sampling is stochastic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

ACTION_DIM = 7


@dataclass
class PolicyOutput:
    """One control step's worth of policy output.

    `action` is the only thing the environment ever receives. `chunk` is
    provenance: the action about to be executed followed by everything still
    queued, or None for single-action policies.
    """

    action: np.ndarray                       # (7,) float32, LIBERO native [-1,1]
    chunk: "np.ndarray | None" = None        # (k, 7) or None
    chunk_len: int = 1
    is_new: bool = True                      # did the network actually run this step?
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.action = np.asarray(self.action, np.float32).reshape(-1)
        if self.action.shape != (ACTION_DIM,):
            raise ValueError(
                f"action must be ({ACTION_DIM},), got {self.action.shape}")
        if self.chunk is not None:
            self.chunk = np.asarray(self.chunk, np.float32)
            if self.chunk.ndim != 2 or self.chunk.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"chunk must be (k,{ACTION_DIM}), got {self.chunk.shape}")


class PolicyAdapter(ABC):
    """Contract: reset between episodes, act once per control step."""

    @abstractmethod
    def reset(self):
        """Clear any observation history. MUST be called before each episode."""

    @abstractmethod
    def act(self, obs) -> PolicyOutput:
        """Raw LIBERO obs dict -> one executable action (+ optional chunk)."""

    @abstractmethod
    def fingerprint(self) -> str:
        """SHA-256 of the exact weights. Stamped into every rollout so a
        mid-collection checkpoint swap is detectable after the fact."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'bct_task0'."""

    @property
    def display_name(self) -> str:
        """Human-facing label for the GUI, e.g. 'BC-Transformer'."""
        return self.name

    @property
    def action_horizon(self) -> int:
        """How many actions the network emits per inference. 1 = no chunking."""
        return 1

    @property
    def compute_class(self) -> str:
        """'cheap' or 'expensive' -- what the arbiter selects between."""
        return "cheap"

    @property
    @abstractmethod
    def observation_spec(self) -> dict:
        """Exactly which obs keys this policy consumes, and at what shape.
        Documented rather than inferred, so the audit trail is explicit."""


class ChunkBuffer:
    """Turns an action chunk into exactly one action per control step.

    The environment must receive one action per step regardless of whether the
    policy emitted 1 or 8. This is the only place that reconciliation happens.
    """

    def __init__(self, horizon=1):
        self.horizon = int(horizon)
        self._q = []

    def push(self, chunk):
        """Replace the queue with a fresh chunk, oldest-first."""
        chunk = np.asarray(chunk, np.float32).reshape(-1, ACTION_DIM)
        self._q = [row.copy() for row in chunk]

    def pop(self):
        """Next executable action. Raises if empty -- callers must check."""
        if not self._q:
            raise IndexError("ChunkBuffer empty; push() a chunk first")
        return self._q.pop(0)

    def clear(self):
        self._q = []

    @property
    def at_boundary(self) -> bool:
        """True when the queue is exhausted -- the only safe moment for the
        arbiter to swap policies without truncating a planned chunk."""
        return not self._q

    def __len__(self):
        return len(self._q)
