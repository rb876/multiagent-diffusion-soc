#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A CONFIGS=(
  [workflows.learning_agents_bptt]=exps/bptt_learning_agents_fine_tuning
  [workflows.learning_agents_bptt_fictitious]=exps/fictitious_bptt_learning_agents_fine_tuning
)

ORDER=(
  workflows.learning_agents_bptt
  workflows.learning_agents_bptt_fictitious
)

for m in "${ORDER[@]}"; do
  echo "Running $m with ${CONFIGS[$m]}"
  python -m "$m" \
    --config-path ../configs \
    --config-name "${CONFIGS[$m]}" \
    exps.soc.optimality_target='[0,1,2,3,4,5,6,7,8,9]'
done