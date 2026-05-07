#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- defaults ----
AGENTS=2
DEFAULT_BATCH_SIZE=16
REDUCED_BATCH_SIZE=8

OPT_TARGET="0,1,2,3,4,5,6,7,8,9"

# ---- CLI args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agents)
      AGENTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# ---- run experiments ----
# Decide batch size
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
if [[ "$AGENTS" -eq 3 ]]; then
  BATCH_SIZE="$REDUCED_BATCH_SIZE"
fi

echo "Running with AGENTS=${AGENTS}, optimality_target=${OPT_TARGET}, batch_size=${BATCH_SIZE}"

declare -A CONFIGS=(
  [workflows.learning_agent_joint]=exps/bptt_learning_agents_fine_tuning
  [workflows.learning_agent_control_wise]=exps/control_wise_bptt_learning_agents_fine_tuning
)

ORDER=(
  workflows.learning_agent_joint
  workflows.learning_agent_control_wise
)

for m in "${ORDER[@]}"; do
  echo "→ $m (${CONFIGS[$m]})"
  python -m "$m" \
    --config-path ../configs \
    --config-name "${CONFIGS[$m]}" \
    -m \
    exps.soc.optimality_target="${OPT_TARGET}" \
    exps.soc.batch_size="${BATCH_SIZE}" \
    exps.soc.num_control_agents="${AGENTS}"
done
