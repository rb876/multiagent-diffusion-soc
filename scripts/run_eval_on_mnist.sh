#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- defaults ----
AGENTS=2

DEFAULT_BATCH_SIZE=16
REDUCED_BATCH_SIZE=8

# sweeps (space-separated lists)
DIGITS_LIST="2 3 4 5 6 7 8 9"
LAMBDA_REG_LIST="0.1 1.0 10.0"
LR_LIST="1e-4"
RUN_OPT_REG_LIST="0.1 1.0 10.0"

# which workflows/configs to run
declare -A CONFIGS=(
  [workflows.learning_agents_bptt]=exps/bptt_learning_agents_fine_tuning
#   [workflows.learning_agents_bptt_fictitious]=exps/fictitious_bptt_learning_agents_fine_tuning
)
ORDER=(
  workflows.learning_agents_bptt
#   workflows.learning_agents_bptt_fictitious
)

# ---- CLI args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agents) AGENTS="$2"; shift 2 ;;
    --digits-list) DIGITS_LIST="$2"; shift 2 ;;          # e.g. "0 1 2" or "7"
    --lambda-reg-list) LAMBDA_REG_LIST="$2"; shift 2 ;;
    --lr-list)         LR_LIST="$2"; shift 2 ;;
    --run-opt-reg-list) RUN_OPT_REG_LIST="$2"; shift 2 ;;
    --only)
      # run only one workflow key, e.g. --only workflows.learning_agents_bptt
      ORDER=("$2"); shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# ---- batch size rule ----
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
if [[ "$AGENTS" -eq 3 ]]; then
  BATCH_SIZE="$REDUCED_BATCH_SIZE"
fi

echo "Sweep: DIGITS=[${DIGITS_LIST}], AGENTS=${AGENTS}, batch_size=${BATCH_SIZE}"
echo "lambda_reg: ${LAMBDA_REG_LIST}"
echo "learning_rate: ${LR_LIST}"
echo "running_optimality_reg: ${RUN_OPT_REG_LIST}"
echo

# ---- run sweep ----
RUN_ID="$(date +%Y%m%d_%H%M%S)"

for m in "${ORDER[@]}"; do
  cfg="${CONFIGS[$m]}"
  echo "==> Workflow: $m  (config: ${cfg})"

  for digit in ${DIGITS_LIST}; do
    for lambda_reg in ${LAMBDA_REG_LIST}; do
      for lr in ${LR_LIST}; do
        for run_opt_reg in ${RUN_OPT_REG_LIST}; do
          NAME="mnist_digit${digit}_A${AGENTS}_bs${BATCH_SIZE}_lam${lambda_reg}_lr${lr}_ror${run_opt_reg}_${RUN_ID}"
          echo "→ ${NAME}"

          python -m "$m" \
            --config-path ../configs \
            --config-name "${cfg}" \
            -m \
            exps.soc.optimality_target="${digit}" \
            exps.soc.batch_size="${BATCH_SIZE}" \
            exps.soc.num_control_agents="${AGENTS}" \
            exps.soc.lambda_reg="${lambda_reg}" \
            exps.soc.learning_rate="${lr}" \
            exps.soc.running_optimality_reg="${run_opt_reg}" \
            exps.wandb.name="${NAME}" \
            exps.wandb.tags="[mnist,sweep,digit${digit},agents${AGENTS}]"
        done
      done
    done
  done

  echo
done