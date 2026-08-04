#!/usr/bin/env bash
# Bootstrap LIBERO. Run once per machine (including the GPU box).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d third_party/LIBERO ]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
fi

# Upstream ships no libero/__init__.py. The old install path used legacy
# develop-mode (repo root on sys.path) so it did not matter; modern setuptools
# does PEP 660 editable installs where find_packages() needs this file, and
# without it the install silently maps zero packages.
touch third_party/LIBERO/libero/__init__.py

# libero/libero/__init__.py calls input() on first import if this file is
# absent -- which hangs any non-interactive run. Write it up front.
mkdir -p ~/.libero
ROOT="$(pwd)/third_party/LIBERO/libero/libero"
cat >~/.libero/config.yaml <<EOF
benchmark_root: $ROOT
bddl_files: $ROOT/bddl_files
init_states: $ROOT/init_files
datasets: $ROOT/../datasets
assets: $ROOT/assets
EOF

# Source fixes. See patches/libero-macos.patch for the per-hunk rationale:
#   utils.py            safe_device() returned None for any non-cpu/cuda device
#   *.py x3             torch.load needs weights_only=False since torch 2.6
#   algos/base.py       re-init robomimic obs registry in spawned DataLoader
#                       workers (macOS spawns, Linux forks); persistent_workers
#                       is a hard error when num_workers == 0
#   transformer_modules legacy .type(t.type()) rejects 'torch.mps.FloatTensor'
git -C third_party/LIBERO apply --check ../../patches/libero-macos.patch 2>/dev/null &&
  git -C third_party/LIBERO apply ../../patches/libero-macos.patch &&
  echo "applied patches/libero-macos.patch" ||
  echo "patch already applied (or conflicts) -- skipping"

uv sync # add --group train to also install the training stack

# Demos: 13.7 GB, only libero_10. Do NOT use benchmark_scripts/download_libero_datasets.py:
#   --datasets libero_10 is not an accepted choice, and --datasets libero_100
#   --use-huggingface globs 'libero_100/*' which matches nothing in the HF repo
#   (it stores libero_10/ and libero_90/), downloading zero files silently.
if [ ! -f third_party/LIBERO/libero/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 ]; then
  uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset',
                  local_dir='third_party/LIBERO/libero/datasets',
                  allow_patterns='libero_10/*', max_workers=4)"
fi

echo "ok. verify with: uv run python scripts/check_action_space.py"
