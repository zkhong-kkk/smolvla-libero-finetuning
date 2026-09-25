#!/usr/bin/env bash
set -euo pipefail

: "${STAGE2_RUN_DIR:?Set STAGE2_RUN_DIR to the completed Stage 2 training output}"
: "${STAGE2_TASK0_EVAL_DIR:?Set STAGE2_TASK0_EVAL_DIR to the 10-episode task-0 output}"

LEROBOT_DIR="${LEROBOT_DIR:-${HOME}/lerobot}"
EXPORT_DIR="${EXPORT_DIR:-${HOME}/github_export/smolvla-libero-artifacts}"
FINAL_CHECKPOINT="$(readlink -f "${STAGE2_RUN_DIR}/checkpoints/last")"
MODEL_DIR="${FINAL_CHECKPOINT}/pretrained_model"

mkdir -p \
  "${EXPORT_DIR}/environment" \
  "${EXPORT_DIR}/evaluation" \
  "${EXPORT_DIR}/policy_metadata" \
  "${EXPORT_DIR}/training" \
  "${EXPORT_DIR}/selected_videos"

copy_if_present() {
  local source_path="$1"
  local destination_path="$2"
  if [[ -f "${source_path}" ]]; then
    cp "${source_path}" "${destination_path}"
  else
    echo "NOTICE: file not found, skipped: ${source_path}" >&2
  fi
}

copy_if_present "${STAGE2_RUN_DIR}/train.log" \
  "${EXPORT_DIR}/training/stage2_train.log"
copy_if_present "${STAGE2_TASK0_EVAL_DIR}/eval.log" \
  "${EXPORT_DIR}/evaluation/stage2_task0_eval.log"

if [[ -n "${STAGE1_RUN_DIR:-}" ]]; then
  copy_if_present "${STAGE1_RUN_DIR}/train.log" \
    "${EXPORT_DIR}/training/stage1_train.log"
fi

if [[ -n "${STAGE1_TASK0_EVAL_DIR:-}" ]]; then
  copy_if_present "${STAGE1_TASK0_EVAL_DIR}/eval.log" \
    "${EXPORT_DIR}/evaluation/stage1_task0_eval.log"
fi

if [[ -n "${FULL_SPATIAL_EVAL_DIR:-}" ]]; then
  copy_if_present "${FULL_SPATIAL_EVAL_DIR}/eval.log" \
    "${EXPORT_DIR}/evaluation/stage2_spatial_all_eval.log"
fi

if [[ -n "${OFFICIAL_EVAL_DIR:-}" ]]; then
  copy_if_present "${OFFICIAL_EVAL_DIR}/eval.log" \
    "${EXPORT_DIR}/evaluation/official_spatial_all_eval.log"
fi

for metadata_name in \
  train_config.json \
  config.json \
  policy_preprocessor.json \
  policy_postprocessor.json; do
  copy_if_present "${MODEL_DIR}/${metadata_name}" \
    "${EXPORT_DIR}/policy_metadata/${metadata_name}"
done

find "${MODEL_DIR}" -maxdepth 1 -type f \
  \( -name '*.json' -o -name '*normalizer*.safetensors' \) \
  ! -name 'model*.safetensors' \
  -exec cp -n {} "${EXPORT_DIR}/policy_metadata/" \;

copy_if_present "${FINAL_CHECKPOINT}/training_state/training_step.json" \
  "${EXPORT_DIR}/policy_metadata/training_step.json"

python -m pip freeze > "${EXPORT_DIR}/environment/pip_freeze.txt"
conda env export --no-builds > "${EXPORT_DIR}/environment/conda_environment.yml"
nvidia-smi > "${EXPORT_DIR}/environment/gpu_info.txt"

if [[ -d "${LEROBOT_DIR}/.git" ]]; then
  git -C "${LEROBOT_DIR}" rev-parse HEAD > \
    "${EXPORT_DIR}/environment/lerobot_commit.txt"
  git -C "${LEROBOT_DIR}" status --short > \
    "${EXPORT_DIR}/environment/lerobot_worktree_status.txt"
fi

if [[ -n "${SUCCESS_VIDEO:-}" ]]; then
  copy_if_present "${SUCCESS_VIDEO}" \
    "${EXPORT_DIR}/selected_videos/stage2_task0_success.mp4"
fi

if [[ -n "${FAILURE_VIDEO:-}" ]]; then
  copy_if_present "${FAILURE_VIDEO}" \
    "${EXPORT_DIR}/selected_videos/stage2_task0_failure.mp4"
fi

echo "Export completed: ${EXPORT_DIR}"
echo "Model weights and optimizer states were intentionally excluded."
du -sh "${EXPORT_DIR}"
