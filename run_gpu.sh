#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=redteam_demo
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --output=redteam_demo_%j.log
#SBATCH --error=redteam_demo_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Project directory resolution
#
# sbatch COPIES this script into /var/spool/slurm/job<ID>/ and runs the copy,
# so BASH_SOURCE[0] points at a spool path the job cannot write to. Prefer
# SLURM_SUBMIT_DIR (where sbatch was invoked from), fall back to the script's
# own location for interactive/srun use, and allow an explicit override for
# when you submit from somewhere other than the project root.
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="${REDTEAM_PROJECT_DIR:-$PROJECT_DIR}"

SCRIPT_PATH="${PROJECT_DIR}/run_gpu.sh"
GPU_VENV="${PROJECT_DIR}/.venv-gpu"

# Match whatever module was used to build .venv-gpu. A bare `python3` picks up
# the system interpreter, which may differ between your submit shell and a
# clean one, so be explicit.
PYTHON_MODULE="${REDTEAM_PYTHON_MODULE:-python/3.11.7}"

# Keep model weights off the home quota. Torch + a few checkpoints will blow
# through a modest home allocation quickly.
export HF_HOME="${REDTEAM_HF_HOME:-${PROJECT_DIR}/.hf-cache}"

load_python_module() {
    if command -v module >/dev/null 2>&1; then
        module load "${PYTHON_MODULE}" 2>/dev/null || {
            echo "Warning: could not load ${PYTHON_MODULE}; using whatever python3 is on PATH." >&2
        }
    fi
}

sanity_check_project_dir() {
    if [[ ! -f "${PROJECT_DIR}/demo.py" ]]; then
        echo "Error: demo.py not found in ${PROJECT_DIR}" >&2
        echo "Submit from the project root, or set REDTEAM_PROJECT_DIR=/path/to/red-teaming-mqp" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# One-time setup. Run this on a LOGIN NODE.
#
# Compute nodes typically have no outbound internet, so pip cannot reach
# download.pytorch.org from inside a job. Setup belongs on the login node.
# ---------------------------------------------------------------------------
setup_gpu_environment() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Error: --setup was invoked inside a Slurm job." >&2
        echo "Compute nodes usually cannot reach the internet. Run this on a login node." >&2
        exit 1
    fi

    sanity_check_project_dir
    load_python_module

    echo "Creating CUDA-enabled environment at ${GPU_VENV}"
    echo "Using: $(command -v python3) ($(python3 --version 2>&1))"

    if [[ ! -x "${GPU_VENV}/bin/python" ]]; then
        python3 -m venv "${GPU_VENV}"
    fi

    # The regular .venv holds CPU-only PyTorch. A separate environment keeps
    # the CPU and CUDA builds from overwriting one another.
    "${GPU_VENV}/bin/pip" install --upgrade pip
    "${GPU_VENV}/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu126
    "${GPU_VENV}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

    mkdir -p "${HF_HOME}"

    cat <<EOF

GPU setup complete.
  venv:     ${GPU_VENV}
  HF cache: ${HF_HOME}
  module:   ${PYTHON_MODULE}

Next steps:
  ./run_gpu.sh --max-new-tokens 1    # warm the HF cache before class
  ./run_gpu.sh --offline --pause     # interactive presentation run
  sbatch run_gpu.sh                  # batch run (no --pause)
EOF
}

if [[ "${1:-}" == "--setup" ]]; then
    setup_gpu_environment
    exit 0
fi

# ---------------------------------------------------------------------------
# Interactive path: from a login shell, grab an L40S and re-exec inside it.
# Slurm sets SLURM_JOB_ID in the job, so this branch runs exactly once.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    sanity_check_project_dir

    if ! command -v srun >/dev/null 2>&1; then
        echo "Error: srun not found. Run this from a Turing login node." >&2
        exit 1
    fi
    if [[ ! -x "${GPU_VENV}/bin/python" ]]; then
        echo "GPU environment not set up yet." >&2
        echo "Run ./run_gpu.sh --setup once, then try again." >&2
        exit 1
    fi

    echo "Requesting one L40S GPU from the short partition ..."
    SRUN_ARGS=(
        --pty
        --nodes=1
        --ntasks=1
        --cpus-per-task=4
        --mem=24G
        --partition=short
        --time=00:30:00
        --gres=gpu:L40S:1
    )
    # Pass the resolved directory through so the inner invocation agrees with
    # this one regardless of where srun lands.
    export REDTEAM_PROJECT_DIR="${PROJECT_DIR}"
    exec srun "${SRUN_ARGS[@]}" "${SCRIPT_PATH}" "$@"
fi

# ---------------------------------------------------------------------------
# Inside the allocation (sbatch or srun).
# ---------------------------------------------------------------------------
sanity_check_project_dir
cd "${PROJECT_DIR}"

load_python_module

# Do NOT build the venv here. Fail loudly instead: a job that silently tries
# to pip install will hang against an unreachable index and burn the whole
# time limit before dying.
if [[ ! -x "${GPU_VENV}/bin/python" ]]; then
    echo "Error: GPU venv missing at ${GPU_VENV}" >&2
    echo "Run ./run_gpu.sh --setup on a login node first." >&2
    exit 1
fi

source "${GPU_VENV}/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "----------------------------------------------------------------"
echo "job id:      ${SLURM_JOB_ID}"
echo "node:        $(hostname)"
echo "project dir: ${PROJECT_DIR}"
echo "venv:        ${GPU_VENV}"
echo "HF cache:    ${HF_HOME}"
echo "python:      $(command -v python) ($(python --version 2>&1))"
echo "----------------------------------------------------------------"

# Fail with a useful message rather than quietly falling back to CPU.
python -u - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "A Slurm GPU was allocated, but this PyTorch build cannot see CUDA. "
        "Rebuild with ./run_gpu.sh --setup on a login node."
    )

print(f"GPU ready: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
PY

# Arguments pass straight through to demo.py.
#   sbatch run_gpu.sh
#   sbatch run_gpu.sh --models gpt2 qwen
#   sbatch run_gpu.sh --offline
#
# Do not use --pause with sbatch: batch jobs have no stdin, so input() will
# hit EOF and crash. --pause is for the interactive ./run_gpu.sh path only.
if [[ -n "${SLURM_JOB_ID:-}" && ! -t 0 ]]; then
    for arg in "$@"; do
        if [[ "$arg" == "--pause" ]]; then
            echo "Error: --pause requires an interactive terminal. Drop it for sbatch." >&2
            exit 1
        fi
    done
fi

exec python -u demo.py "$@"