#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${PROJECT_DIR}/configs/evaluation.env}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: configuration file not found: ${CONFIG_FILE}" >&2
  echo "Copy configs/evaluation.env.example to configs/evaluation.env first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
N_EPISODES="${N_EPISODES:-10}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-smolvla_libero_eval}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_${STAMP}"
LOG_FILE="${OUTPUT_DIR}.log"
DEFAULT_RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP:-${DEFAULT_RENAME_MAP}}"

mkdir -p "${OUTPUT_ROOT}"

args=(
  "--policy.path=${MODEL_PATH}"
  "--policy.device=cuda"
  "--policy.empty_cameras=${EMPTY_CAMERAS:-1}"
  "--policy.n_action_steps=${N_ACTION_STEPS:-10}"
  "--env.type=libero"
  "--env.task=${TASK_SUITE}"
  "--env.max_parallel_tasks=${MAX_PARALLEL_TASKS:-1}"
  "--eval.batch_size=${BATCH_SIZE:-1}"
  "--eval.n_episodes=${N_EPISODES}"
  "--eval.use_async_envs=${USE_ASYNC_ENVS:-false}"
  "--rename_map=${RENAME_MAP}"
  "--seed=${SEED}"
  "--output_dir=${OUTPUT_DIR}"
)

if [[ -n "${TASK_IDS:-}" ]]; then
  args+=("--env.task_ids=${TASK_IDS}")
fi

echo "Model: ${MODEL_PATH}"
echo "Suite: ${TASK_SUITE}"
echo "Task IDs: ${TASK_IDS:-all}"
echo "Episodes per task: ${N_EPISODES}"
echo "Output: ${OUTPUT_DIR}"

set +e
set -o pipefail
lerobot-eval "${args[@]}" 2>&1 | tee "${LOG_FILE}"
eval_status="${PIPESTATUS[0]}"
set -e

if [[ "${eval_status}" -ne 0 ]]; then
  echo "ERROR: evaluation failed. Log: ${LOG_FILE}" >&2
  exit "${eval_status}"
fi

cp "${LOG_FILE}" "${OUTPUT_DIR}/eval.log"

echo
echo "Evaluation completed."
grep -E 'Output dir:|Success rate [0-9]' "${OUTPUT_DIR}/eval.log" | tail -n 5 || true
echo "Saved log: ${OUTPUT_DIR}/eval.log"
