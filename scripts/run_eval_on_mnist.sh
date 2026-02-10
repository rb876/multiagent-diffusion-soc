#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- GPU setup ----
export CUDA_VISIBLE_DEVICES=0

# ---- defaults ----
AGENTS=2

DEFAULT_BATCH_SIZE=16
REDUCED_BATCH_SIZE=8

# sweeps (space-separated lists)
DIGITS_LIST="9 3 0"
LAMBDA_REG_LIST="10.0 1.0"
LR_LIST="1e-4"  
RUN_STATE_COST_SCALING_LIST="1.0 10.0"

# which workflows/configs to run
declare -A CONFIGS=(
  [workflows.learning_agents_bptt_fictitious]=exps/fictitious_bptt_learning_agents_fine_tuning
  [workflows.learning_agents_bptt]=exps/bptt_learning_agents_fine_tuning
)
ORDER=(
  workflows.learning_agents_bptt_fictitious
  workflows.learning_agents_bptt
)

# ---- CLI args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agents) AGENTS="$2"; shift 2 ;;
    --digits-list) DIGITS_LIST="$2"; shift 2 ;;          # e.g. "0 1 2" or "7"
    --lambda-reg-list) LAMBDA_REG_LIST="$2"; shift 2 ;;
    --lr-list)         LR_LIST="$2"; shift 2 ;;
    --run-state-cost-scaling-list) RUN_STATE_COST_SCALING_LIST="$2"; shift 2 ;;
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
echo "running_state_cost_scaling: ${RUN_STATE_COST_SCALING_LIST}"
echo

# ---- run sweep ----
RUN_ID="$(date +%Y%m%d_%H%M%S)"

for m in "${ORDER[@]}"; do
  cfg="${CONFIGS[$m]}"
  echo "==> Workflow: $m  (config: ${cfg})"

  for digit in ${DIGITS_LIST}; do
    for lambda_reg in ${LAMBDA_REG_LIST}; do
      for lr in ${LR_LIST}; do
        for run_state_cost_scaling in ${RUN_STATE_COST_SCALING_LIST}; do
          NAME="mnist_digit${digit}_A${AGENTS}_bs${BATCH_SIZE}_lam${lambda_reg}_lr${lr}_ror${run_state_cost_scaling}_${RUN_ID}"
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
            exps.soc.running_state_cost_scaling="${run_state_cost_scaling}" \
            exps.wandb.name="${NAME}" \
            exps.wandb.tags="[mnist,sweep,digit${digit},agents${AGENTS},NEW_CONTROL,SWEEP]" \
            exps.sde.name="VP" \
            exps.soc.path_to_score_model_checkpoint="checkpoints/vp/latest.ckpt"
        done
      done
    done
  done

  echo
done