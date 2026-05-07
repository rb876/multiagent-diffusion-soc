#!/usr/bin/env bash
set -euo pipefail
set -f  # disable globbing
cd "$(dirname "$0")/.."

# Shared configs
declare -A CONFIGS=(
  [workflows.learning_agent_joint]=exps/bptt_learning_agents_fine_tuning
  [workflows.learning_agent_control_wise]=exps/control_wise_bptt_learning_agents_fine_tuning
)

run_suite () {
  local suite_name="$1"
  local agents="$2"
  local batch_size="$3"
  local targets_csv="$4"                 # e.g. "0,1,2,3,4,5,6,7,8,9"
  local running_state_cost_scaling="$5"
  local control_cost_scaling="$6"
  shift 6
  local workflows=("$@")

  echo
  echo "==================== ${suite_name} ===================="
  echo "Agents=${agents} | Batch=${batch_size} | Targets(sweep)=${targets_csv}"
  echo "running_state_cost_scaling=${running_state_cost_scaling} | control_cost_scaling=${control_cost_scaling}"
  echo "Workflows: ${workflows[*]}"
  echo "======================================================="
  echo

  for m in "${workflows[@]}"; do
    local cfg="${CONFIGS[$m]}"
    echo "→ $m (${cfg})"

    local tags="[mnist,${suite_name},agents${agents},batch${batch_size}]"

    # Note: use Hydra interpolation so each multirun job gets the right target in the name.
    # Single quotes prevent bash from expanding ${...}.
    local wandb_name="mnist_${suite_name}_A${agents}_B${batch_size}_T\${exps.soc.optimality_target}"

    python -m "$m" \
      -m \
      --config-path ../configs \
      --config-name "$cfg" \
      "exps.soc.optimality_target=${targets_csv}" \
      "exps.soc.batch_size=${batch_size}" \
      "exps.soc.num_control_agents=${agents}" \
      "exps.soc.running_state_cost_scaling=${running_state_cost_scaling}" \
      "exps.soc.control_cost_scaling=${control_cost_scaling}" \
      "exps.wandb.name=${wandb_name}" \
      "exps.wandb.tags=${tags}"
  done
}

# Baselines
BASE_TARGETS="0,1,2,3,4,5,6,7,8,9"
BASE_RUNNING_STATE_COST=10
BASE_CONTROL_COST_SCALING=1

WF_ALL=(
  workflows.learning_agent_joint
  workflows.learning_agent_control_wise
)

WF_BPTT=(
  workflows.learning_agent_joint
)

# EXP 1
run_suite "exp1" 2 16 "$BASE_TARGETS" "$BASE_RUNNING_STATE_COST" "$BASE_CONTROL_COST_SCALING" "${WF_ALL[@]}"

# EXP 2
run_suite "exp2" 3 8  "$BASE_TARGETS" "$BASE_RUNNING_STATE_COST" "$BASE_CONTROL_COST_SCALING" "${WF_ALL[@]}"

# ABLATION 1: no running state cost scaling
ABL_TARGETS="0,4"
run_suite "abl1_noRunningOptReg" 2 16 "$ABL_TARGETS" 0 "$BASE_CONTROL_COST_SCALING" "${WF_BPTT[@]}"

# ABLATION 2: high lambda + low state cost
run_suite "abl2_highLambda_lowStateCost" 2 16 "$ABL_TARGETS" 0.01 10 "${WF_BPTT[@]}"

echo
echo "All experiments completed."
