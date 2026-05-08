#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_SCRIPT="$REPO_ROOT/scripts/run_eval_on_mnist_slurm.sh"

PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
JOB_NAME="${JOB_NAME:-eval_mnist}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/slurm}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/venv}"
DRY_RUN=0
ARRAY_MODE=0
MAX_PARALLEL="${MAX_PARALLEL:-}"
EVAL_SWEEP_RUN_ID="${EVAL_SWEEP_RUN_ID:-}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/submit_eval_on_mnist_slurm.sh [slurm options] -- [run_eval_on_mnist.sh options]

Examples:
  ./scripts/submit_eval_on_mnist_slurm.sh \
    --partition workq \
    --time 08:00:00 \
    --mem 32G \
    --cpus-per-task 4 \
    -- --agents 3 --digits-list "9 3 0"

  ./scripts/submit_eval_on_mnist_slurm.sh \
    --account mylab \
    --qos normal \
    --job-name eval_mnist_a2 \
    -- --agents 2 --only workflows.learning_agent_joint

  ./scripts/submit_eval_on_mnist_slurm.sh \
    --array \
    --max-parallel 8 \
    --partition workq \
    -- --agents 3 --digits-list "9 3 0"

Launcher options:
  --partition NAME       Slurm partition to submit to.
  --account NAME         Slurm account to charge.
  --qos NAME             Slurm QoS.
  --time HH:MM:SS        Wall time limit. Default: 12:00:00
  --gpus N               Number of GPUs. Default: 1
  --cpus-per-task N      CPUs per task. Default: 4
  --mem SIZE             Memory request. Default: 32G
  --job-name NAME        Slurm job name. Default: eval_mnist
  --log-dir PATH         Directory for stdout/stderr logs.
  --venv PATH            Virtualenv to activate inside the job.
  --array                Submit one Slurm array task per eval combo.
  --max-parallel N       Limit simultaneous array tasks to N.
  --dry-run              Print the sbatch command without submitting.
  -h, --help             Show this help.

Everything after `--` is passed through to `scripts/run_eval_on_mnist.sh`.
EOF
}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch is not available in PATH." >&2
  exit 1
fi

TARGET_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --cpus-per-task)
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --array)
      ARRAY_MODE=1
      shift
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      TARGET_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown launcher option: $1" >&2
      echo "Use -- to separate launcher options from run_eval_on_mnist.sh options." >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"
export VENV_PATH

ARRAY_SPEC=""
if [[ "$ARRAY_MODE" -eq 1 ]]; then
  COUNT_CMD=(bash "$REPO_ROOT/scripts/run_eval_on_mnist.sh" --count-combos)
  if [[ "${#TARGET_ARGS[@]}" -gt 0 ]]; then
    COUNT_CMD+=("${TARGET_ARGS[@]}")
  fi

  COMBO_COUNT="$("${COUNT_CMD[@]}")"
  if ! [[ "$COMBO_COUNT" =~ ^[0-9]+$ ]] || [[ "$COMBO_COUNT" -le 0 ]]; then
    echo "Error: failed to resolve a valid combo count for array submission." >&2
    exit 1
  fi

  ARRAY_SPEC="0-$((COMBO_COUNT - 1))"
  if [[ -n "$MAX_PARALLEL" ]]; then
    if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -le 0 ]]; then
      echo "Error: --max-parallel must be a positive integer." >&2
      exit 1
    fi
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PARALLEL}"
  fi

  if [[ -z "$EVAL_SWEEP_RUN_ID" ]]; then
    EVAL_SWEEP_RUN_ID="$(date +%Y%m%d_%H%M%S)"
  fi
  export EVAL_SWEEP_RUN_ID
fi

SBATCH_CMD=(
  sbatch
  --job-name "$JOB_NAME"
  --output "$LOG_DIR/%x-%j.out"
  --error "$LOG_DIR/%x-%j.err"
  --time "$TIME_LIMIT"
  --gres "gpu:${GPUS}"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$MEMORY"
  --export=ALL
)

if [[ "$ARRAY_MODE" -eq 1 ]]; then
  SBATCH_CMD[4]="$LOG_DIR/%x-%A_%a.out"
  SBATCH_CMD[6]="$LOG_DIR/%x-%A_%a.err"
  SBATCH_CMD+=(--array "$ARRAY_SPEC")
fi

if [[ -n "$PARTITION" ]]; then
  SBATCH_CMD+=(--partition "$PARTITION")
fi

if [[ -n "$ACCOUNT" ]]; then
  SBATCH_CMD+=(--account "$ACCOUNT")
fi

if [[ -n "$QOS" ]]; then
  SBATCH_CMD+=(--qos "$QOS")
fi

SBATCH_CMD+=("$JOB_SCRIPT")

if [[ "${#TARGET_ARGS[@]}" -gt 0 ]]; then
  SBATCH_CMD+=("${TARGET_ARGS[@]}")
fi

if [[ "$ARRAY_MODE" -eq 1 ]]; then
  echo "Array submission: ${COMBO_COUNT} combos"
  echo "Array spec: ${ARRAY_SPEC}"
  echo "Sweep run id: ${EVAL_SWEEP_RUN_ID}"
fi

printf 'Submitting command:\n  '
printf '%q ' "${SBATCH_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
