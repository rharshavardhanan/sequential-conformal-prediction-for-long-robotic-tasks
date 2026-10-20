"""Policy adapter tests. Only the registry/shape tests need a checkpoint."""

import numpy as np
import pytest

from scripts.policies import ACTION_DIM, ChunkBuffer, PolicyOutput
from scripts.policies import registry as R


def test_policy_output_rejects_a_wrong_action_shape():
    PolicyOutput(np.zeros(7))
    with pytest.raises(ValueError):
        PolicyOutput(np.zeros(6))
    with pytest.raises(ValueError):
        PolicyOutput(np.zeros(7), chunk=np.zeros((8, 6)))


def test_chunk_buffer_yields_one_action_per_step_in_order():
    b = ChunkBuffer(8)
    chunk = np.arange(8 * ACTION_DIM, dtype=np.float32).reshape(8, ACTION_DIM)
    b.push(chunk)
    assert len(b) == 8 and not b.at_boundary
    for i in range(8):
        np.testing.assert_array_equal(b.pop(), chunk[i])
    assert b.at_boundary and len(b) == 0
    with pytest.raises(IndexError):
        b.pop()


def test_chunk_buffer_preserves_contents_exactly():
    """Chunk data must survive: FAIL-Detect needs it and it is not
    reconstructable afterwards -- diffusion sampling is stochastic."""
    b = ChunkBuffer(4)
    chunk = np.random.RandomState(0).randn(4, ACTION_DIM).astype(np.float32)
    b.push(chunk)
    out = np.stack([b.pop() for _ in range(4)])
    np.testing.assert_allclose(out, chunk)


def test_registry_reports_measured_success_and_never_invents_it():
    e = R.get("bct_task0")
    assert e.success == 0.44 and e.success_n == 50 and e.compute_class == "cheap"
    d = R.get("diffusion_task0")
    assert d.success == 0.687 and d.compute_class == "expensive"
    act = R.get("act_task0")
    assert act.success is None, "ACT has never been trained -- must not carry a number"
    assert not act.available


def test_unavailable_policy_refuses_to_build():
    with pytest.raises(SystemExit) as ei:
        R.build("act_task0")
    assert "not available" in str(ei.value) or "no checkpoint" in str(ei.value).lower()


def test_unknown_policy_name_is_rejected():
    with pytest.raises(SystemExit):
        R.get("no_such_policy")


def test_bct_adapter_loads_and_declares_its_contract():
    a = R.build("bct_task0", device="cpu")
    assert a.fingerprint().startswith("dfb20c44")
    assert a.display_name == "BC-Transformer"
    assert a.action_horizon == 1
    spec = a.observation_spec
    for k in ("agentview_rgb", "eye_in_hand_rgb", "gripper_states",
              "joint_states", "task_emb"):
        assert k in spec


def test_bct_adapter_emits_a_valid_action_from_a_synthetic_observation():
    """Shape + range, without paying for a MuJoCo episode."""
    a = R.build("bct_task0", device="cpu")
    a.reset()
    obs = {
        "agentview_image": np.zeros((256, 256, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), np.uint8),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_joint_pos": np.zeros(7),
    }
    out = a.act(obs)
    assert out.action.shape == (ACTION_DIM,)
    assert out.action.dtype == np.float32
    assert np.all(np.abs(out.action) <= 1.0)
    assert out.chunk is None and out.chunk_len == 1
