#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Runs:
#  - exp1: agents=2, batch=16, targets=0-9, workflows={bptt, fictitious}
#  - exp2: agents=3, batch=8,  targets=0-9, workflows={bptt, fictitious}
#  - abl1: agents=3, batch=8,  targets=0,4,9, workflow={bptt}, running_optimality_reg=0
#  - abl2: agents=3, batch=8,  targets=0,4,9, workflow={bptt}, lambda_reg=10, running_state_cost=0.01

# -------- shared configs --------
declare -A CONFIGS=(
  [workflows.learning_agents_bptt]=exps/bptt_learning_agents_fine_tuning
  [workflows.learning_agents_bptt_fictitious]=exps/fictitious_bptt_learning_agents_fine_tuning
)

run_suite () {
  local suite_name="$1"
  local agents="$2"
  local batch_size="$3"
  local opt_target="$4"
  local running_opt_reg="$5"
  local lambda_reg="$6"
  local running_state_cost="$7"
  shift 7
  local workflows=("$@")

  echo
  echo "==================== ${suite_name} ===================="
  echo "Agents=${agents} | Batch=${batch_size} | Targets=${opt_target}"
  echo "running_optimality_reg=${running_opt_reg} | lambda_reg=${lambda_reg} | running_state_cost=${running_state_cost}"
  echo "Workflows: ${workflows[*]}"
  echo "======================================================="
  echo

  for m in "${workflows[@]}"; do
    local cfg="${CONFIGS[$m]}"
    echo "→ $m (${cfg})"

    local name="mnist_${suite_name}_A${agents}_B${batch_size}_T${opt_target//,/}"
    local tags="[mnist,${suite_name},agents${agents},batch${batch_size}]"

    python -m "$m" \
      --config-path ../configs \
      --config-name "$cfg" \
      exps.soc.optimality_target="$opt_target" \
      exps.soc.batch_size="$batch_size" \
      exps.soc.num_control_agents="$agents" \
      exps.soc.running_optimality_reg="$running_opt_reg" \
      exps.soc.lambda_reg="$lambda_reg" \
      exps.soc.running_state_cost="$running_state_cost" \
      exps.wandb.name="$name" \
      exps.wandb.tags="$tags"
  done
}

# -------- baseline hyperparams --------
BASE_OPT_TARGET="0,1,2,3,4,5,6,7,8,9"
BASE_OPT_REG=10
BASE_LAMBDA_REG=1
BASE_STATE_COST=1.0

WF_ALL=(
  workflows.learning_agents_bptt
  workflows.learning_agents_bptt_fictitious
)

WF_BPTT=(
  workflows.learning_agents_bptt
)

# -------- EXPERIMENT 1 --------
run_suite "exp1" 2 16 "$BASE_OPT_TARGET" "$BASE_OPT_REG" "$BASE_LAMBDA_REG" "$BASE_STATE_COST" "${WF_ALL[@]}"

# -------- EXPERIMENT 2 --------
run_suite "exp2" 3 8 "$BASE_OPT_TARGET" "$BASE_OPT_REG" "$BASE_LAMBDA_REG" "$BASE_STATE_COST" "${WF_ALL[@]}"

# -------- ABLATION 1: no running optimality reg --------
ABL_TARGET="0,4,9"
run_suite "abl1_noRunningOptReg" 3 8 "$ABL_TARGET" 0 "$BASE_LAMBDA_REG" "$BASE_STATE_COST" "${WF_BPTT[@]}"

# -------- ABLATION 2: high lambda + low state cost --------
run_suite "abl2_highLambda_lowStateCost" 3 8 "$ABL_TARGET" "$BASE_OPT_REG" 10 0.01 "${WF_BPTT[@]}"

echo
echo "All experiments completed."