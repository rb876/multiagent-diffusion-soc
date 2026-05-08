#!/usr/bin/env bash
set -euo pipefail
set -f
cd "$(dirname "$0")/.."

# ---- GPU setup ----
# Respect a scheduler-provided CUDA_VISIBLE_DEVICES if one already exists.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# ---- defaults ----
AGENTS=2

DEFAULT_BATCH_SIZE=16
REDUCED_BATCH_SIZE=8

# sweeps (space-separated lists)
DIGITS_LIST="9 3 0"
CONTROL_COST_SCALING_LIST="10.0 1.0"
LR_LIST="1e-4"
RUN_STATE_COST_SCALING_LIST="1.0 10.0"
COMBO_INDEX=""
COUNT_COMBOS=0
LIST_COMBOS=0
RUN_ID="${EVAL_SWEEP_RUN_ID:-}"

# which workflows/configs to run
declare -A CONFIGS=(
  [workflows.learning_agent_control_wise]=exps/control_wise_bptt_learning_agents_fine_tuning
  [workflows.learning_agent_joint]=exps/bptt_learning_agents_fine_tuning
)
declare -A WORKFLOW_LABELS=(
  [workflows.learning_agent_control_wise]=control_wise
  [workflows.learning_agent_joint]=joint
)
ORDER=(
  workflows.learning_agent_control_wise
  workflows.learning_agent_joint
)

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_eval_on_mnist.sh [options]

Options:
  --agents N
  --digits-list "9 3 0"
  --control-cost-scaling-list "10.0 1.0"
  --lambda-reg-list "10.0 1.0"              Legacy alias for --control-cost-scaling-list
  --lr-list "1e-4"
  --run-state-cost-scaling-list "1.0 10.0"
  --only WORKFLOW                           e.g. workflows.learning_agent_joint
  --combo-index N                           Run only the Nth combo (0-based)
  --count-combos                            Print the total combo count and exit
  --list-combos                             Print combo metadata and exit
  --run-id ID                               Override the shared run id suffix
  -h, --help
EOF
}

# ---- CLI args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agents) AGENTS="$2"; shift 2 ;;
    --digits-list) DIGITS_LIST="$2"; shift 2 ;;          # e.g. "0 1 2" or "7"
    --control-cost-scaling-list) CONTROL_COST_SCALING_LIST="$2"; shift 2 ;;
    --lambda-reg-list) LAMBDA_REG_LIST="$2"; CONTROL_COST_SCALING_LIST="$2"; shift 2 ;;  # legacy alias
    --lr-list)         LR_LIST="$2"; shift 2 ;;
    --run-state-cost-scaling-list) RUN_STATE_COST_SCALING_LIST="$2"; shift 2 ;;
    --only)
      # run only one workflow key, e.g. --only workflows.learning_agent_joint
      ORDER=("$2"); shift 2
      ;;
    --combo-index)
      COMBO_INDEX="$2"; shift 2 ;;
    --count-combos)
      COUNT_COMBOS=1; shift ;;
    --list-combos)
      LIST_COMBOS=1; shift ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ---- batch size rule ----
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
if [[ "$AGENTS" -eq 3 ]]; then
  BATCH_SIZE="$REDUCED_BATCH_SIZE"
fi

read -r -a DIGITS <<< "$DIGITS_LIST"
read -r -a CONTROL_COST_SCALINGS <<< "$CONTROL_COST_SCALING_LIST"
read -r -a LRS <<< "$LR_LIST"
read -r -a RUN_STATE_COST_SCALINGS <<< "$RUN_STATE_COST_SCALING_LIST"

TOTAL_COMBOS=$(( ${#ORDER[@]} * ${#DIGITS[@]} * ${#CONTROL_COST_SCALINGS[@]} * ${#LRS[@]} * ${#RUN_STATE_COST_SCALINGS[@]} ))

if [[ "$TOTAL_COMBOS" -eq 0 ]]; then
  echo "Error: sweep resolved to zero combinations." >&2
  exit 1
fi

if [[ "$COUNT_COMBOS" -eq 1 ]]; then
  echo "$TOTAL_COMBOS"
  exit 0
fi

if [[ -n "$COMBO_INDEX" ]]; then
  if ! [[ "$COMBO_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Error: --combo-index must be a non-negative integer." >&2
    exit 1
  fi

  if (( COMBO_INDEX >= TOTAL_COMBOS )); then
    echo "Error: --combo-index ${COMBO_INDEX} is out of range for ${TOTAL_COMBOS} combos." >&2
    exit 1
  fi
fi

if [[ "$LIST_COMBOS" -eq 0 ]]; then
  echo "Sweep: DIGITS=[${DIGITS_LIST}], AGENTS=${AGENTS}, batch_size=${BATCH_SIZE}"
  echo "control_cost_scaling: ${CONTROL_COST_SCALING_LIST}"
  echo "learning_rate: ${LR_LIST}"
  echo "running_state_cost_scaling: ${RUN_STATE_COST_SCALING_LIST}"
  echo "workflows: ${ORDER[*]}"
  echo "total_combos: ${TOTAL_COMBOS}"
  if [[ -n "$COMBO_INDEX" ]]; then
    echo "mode: combo_index=${COMBO_INDEX}"
  else
    echo "mode: full sweep"
  fi
  echo
fi

if [[ "$LIST_COMBOS" -eq 0 && -z "$RUN_ID" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

if [[ "$LIST_COMBOS" -eq 0 ]]; then
  echo "run_id: ${RUN_ID}"
  echo
fi

run_eval_combo() {
  local workflow="$1"
  local cfg="$2"
  local digit="$3"
  local control_cost_scaling="$4"
  local lr="$5"
  local run_state_cost_scaling="$6"
  local combo_id="$7"
  local name
  local workflow_label

  workflow_label="${WORKFLOW_LABELS[$workflow]}"

  name="mnist_${workflow_label}_digit${digit}_A${AGENTS}_bs${BATCH_SIZE}_ccs${control_cost_scaling}_lr${lr}_ror${run_state_cost_scaling}_${RUN_ID}"

  echo "[combo $((combo_id + 1))/${TOTAL_COMBOS}] ${workflow}"
  echo "→ ${name}"

  python -m "$workflow" \
    --config-path ../configs \
    --config-name "${cfg}" \
    -m \
    exps.soc.optimality_target="${digit}" \
    exps.soc.batch_size="${BATCH_SIZE}" \
    exps.soc.num_control_agents="${AGENTS}" \
    exps.soc.control_cost_scaling="${control_cost_scaling}" \
    exps.soc.learning_rate="${lr}" \
    exps.soc.running_state_cost_scaling="${run_state_cost_scaling}" \
    exps.wandb.name="${name}" \
    exps.wandb.tags="[mnist,sweep,workflow_${workflow_label},digit${digit},agents${AGENTS},NEW_CONTROL,SWEEP]" \
    exps.sde.name="VP" \
    exps.soc.path_to_score_model_checkpoint="checkpoints/vp/latest.ckpt"
}

combo_id=0
for m in "${ORDER[@]}"; do
  cfg="${CONFIGS[$m]}"
  if [[ "$LIST_COMBOS" -eq 0 && -z "$COMBO_INDEX" ]]; then
    echo "==> Workflow: $m  (config: ${cfg})"
  fi

  for digit in "${DIGITS[@]}"; do
    for control_cost_scaling in "${CONTROL_COST_SCALINGS[@]}"; do
      for lr in "${LRS[@]}"; do
        for run_state_cost_scaling in "${RUN_STATE_COST_SCALINGS[@]}"; do
          if [[ "$LIST_COMBOS" -eq 1 ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$combo_id" "$m" "$digit" "$control_cost_scaling" "$lr" "$run_state_cost_scaling" "$BATCH_SIZE"
            combo_id=$((combo_id + 1))
            continue
          fi

          if [[ -n "$COMBO_INDEX" && "$combo_id" -ne "$COMBO_INDEX" ]]; then
            combo_id=$((combo_id + 1))
            continue
          fi

          run_eval_combo "$m" "$cfg" "$digit" "$control_cost_scaling" "$lr" "$run_state_cost_scaling" "$combo_id"

          if [[ -n "$COMBO_INDEX" ]]; then
            exit 0
          fi

          combo_id=$((combo_id + 1))
        done
      done
    done
  done

  if [[ "$LIST_COMBOS" -eq 0 && -z "$COMBO_INDEX" ]]; then
    echo
  fi
done
