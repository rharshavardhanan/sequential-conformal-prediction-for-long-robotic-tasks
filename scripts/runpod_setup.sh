#!/usr/bin/env bash
# One-shot setup for a RunPod (or any headless Linux + NVIDIA) box.
#
#   bash scripts/runpod_setup.sh
#
# Run it from the project root, with the project living on the PERSISTENT
# volume (/workspace on RunPod) so the 13.7 GB dataset survives a pod restart:
#
#   cd /workspace && git clone <your-remote> PROJECT_1 && cd PROJECT_1
#   # ...or from the Mac:  rsync -av --exclude .venv --exclude third_party \
#   #     ~/PROJECTS/PROJECT_1/ root@<pod>:/workspace/PROJECT_1/
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

echo "== project root: $PROJECT_ROOT"
case "$PROJECT_ROOT" in
/workspace/*) ;;
*)
  echo "   WARNING: not under /workspace. On RunPod only /workspace persists;"
  echo "            everything here is lost when the pod stops."
  ;;
esac

# --- system libs -----------------------------------------------------------
# MuJoCo renders offscreen through EGL on headless Linux. Without libEGL the
# failure shows up during EVAL, not training -- i.e. an hour into a run.
if command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    git curl ca-certificates build-essential cmake \
    libgl1 libegl1 libglx-mesa0 libosmesa6 libglib2.0-0 patchelf 2>/dev/null ||
    echo "   (some apt packages unavailable on this image; continuing)"
fi

# --- uv --------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

# --- headless rendering backend -------------------------------------------
# Unset on macOS (the native stack handles offscreen); required here.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
for rc in "$HOME/.bashrc" "$HOME/.profile"; do
  grep -q "MUJOCO_GL=egl" "$rc" 2>/dev/null || {
    echo 'export MUJOCO_GL=egl' >>"$rc"
    echo 'export PYOPENGL_PLATFORM=egl' >>"$rc"
  }
done
echo "== MUJOCO_GL=$MUJOCO_GL"

# --- LIBERO + deps ---------------------------------------------------------
# bootstrap.sh clones LIBERO, applies patches/libero-macos.patch (all five fixes
# apply identically here), writes ~/.libero/config.yaml, syncs, and downloads
# the 13.7 GB libero_10 demos. egl-probe DOES install on Linux, by design --
# override-dependencies confines it to sys_platform == 'linux'.
bash scripts/bootstrap.sh
uv sync --group train

# --- verify ----------------------------------------------------------------
echo
echo "== verification"
uv run python - <<'PY'
import torch
print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device         : {torch.cuda.get_device_name(0)}")
    print(f"vram           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("!! no CUDA -- training will fall back to CPU and be far slower than the Mac")
PY

echo
echo "== offscreen render check (this is what MUJOCO_GL guards)"
uv run python - <<'PY'
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
s = benchmark.get_benchmark_dict()["libero_10"]()
t = s.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=f"{get_libero_path('bddl_files')}/{t.problem_folder}/{t.bddl_file}",
    camera_heights=128, camera_widths=128)
env.seed(0); obs = env.reset()
k = [x for x in obs if x.endswith("_image")]
print("rendered camera keys:", k)
for key in k:
    print(f"  {key}: {obs[key].shape} {obs[key].dtype}")
env.close()
print("offscreen rendering OK")
PY

cat <<'EOF'

== next steps
  # 1. measure the real speedup before committing to a long run (~1 min)
  uv run python -m scripts.profile_train_step --device cuda --num-workers 12

  # 2. train. 4 tasks x 100 rollouts beats 10 x 50 for CP, so consider --tasks 0,1,2,3
  uv run python -m scripts.train_all_tasks --tasks 0,1,2,3 --device cuda \
      --num-workers 12 --num-procs 20 --n-epochs 50

  # on a 48 GB card (A40/A6000) each process needs ~8 GB, so tasks can overlap.
  # Total DataLoader workers = --parallel x --num-workers; keep that <= vCPUs.
  # Start at 2 and check scaling before going higher -- it only helps if one
  # process leaves the GPU underutilised.
  uv run python -m scripts.train_all_tasks --tasks 0,1,2,3 --device cuda \
      --parallel 2 --num-workers 6 --num-procs 10 --n-epochs 50

  # 3. confirm each checkpoint lands in the 40-80% band before collecting
  uv run python -m scripts.eval_checkpoint --checkpoint <path> --task-id <id> --n-eval 20

  Watch progress:  uv run python scripts/watch_training.py --log logs/train_task0.log
  Manifest:        checkpoints.json  (task -> checkpoint -> sha256)
EOF
