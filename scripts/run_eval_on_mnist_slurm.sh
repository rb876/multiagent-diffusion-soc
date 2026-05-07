#!/usr/bin/env bash
set -euo pipefail

# When Slurm launches a batch script it may execute a spooled copy under
# /var/spool/slurmd, so prefer the original submission directory.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

echo "[$(date -Is)] Starting MNIST eval sweep job"
echo "Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Slurm job id: ${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Load the project virtualenv when it exists.
if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
elif [[ -f "$REPO_ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/venv/bin/activate"
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

echo "Python: $(command -v python)"

exec bash "$REPO_ROOT/scripts/run_eval_on_mnist.sh" "$@"
