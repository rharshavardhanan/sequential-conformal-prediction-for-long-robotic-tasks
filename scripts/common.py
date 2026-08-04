"""Shared config assembly + checkpoint identity.

LIBERO's `main.py` builds every task's dataset and then trains all of them in
sequence; there is no switch for "just task k". Rather than fight hydra we
compose the same config here and drive `learn_one_task` directly.
"""

import hashlib
import os
from pathlib import Path

import yaml
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "third_party" / "LIBERO" / "libero" / "configs"


def sha256_file(path, _chunk=1 << 20):
    """Checkpoint identity, stamped into every rollout file.

    CP's guarantee assumes one data-generating process. Recording this per
    rollout makes a mid-collection policy swap detectable after the fact
    instead of silently corrupting the calibration set.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_cfg(overrides=None):
    """Compose LIBERO's hydra config with our overrides, as an EasyDict."""
    from libero.libero import get_libero_path

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        hydra_cfg = compose(config_name="config", overrides=overrides or [])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))

    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    return cfg


def pick_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_one_task(cfg, task_id):
    """Dataset + shape_meta for a single task, plus task embeddings for all.

    Embeddings are computed for every task description because `benchmark`
    indexes them by global task id, but only one hdf5 is opened.
    """
    from libero.lifelong.datasets import SequenceVLDataset, get_dataset
    from libero.lifelong.utils import get_task_embs
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)

    dataset, shape_meta = get_dataset(
        dataset_path=os.path.join(
            cfg.folder, benchmark.get_task_demonstration(task_id)
        ),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )

    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    return benchmark, SequenceVLDataset(dataset, task_embs[task_id]), shape_meta
