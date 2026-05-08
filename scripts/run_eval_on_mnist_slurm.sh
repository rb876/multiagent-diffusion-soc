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
echo "Slurm array task id: ${SLURM_ARRAY_TASK_ID:-none}"
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

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_ID="${EVAL_SWEEP_RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}}"
  echo "Array mode: combo_index=${SLURM_ARRAY_TASK_ID}, run_id=${RUN_ID}"
  exec bash "$REPO_ROOT/scripts/run_eval_on_mnist.sh" \
    --combo-index "${SLURM_ARRAY_TASK_ID}" \
    --run-id "${RUN_ID}" \
    "$@"
fi

exec bash "$REPO_ROOT/scripts/run_eval_on_mnist.sh" "$@"
