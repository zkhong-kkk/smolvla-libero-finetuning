#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the checkpoint pretrained_model directory}"

MODEL_DIR="$(readlink -f "${MODEL_DIR}")"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: checkpoint directory does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -s "${MODEL_DIR}/train_config.json" ]]; then
  echo "ERROR: missing or empty train_config.json" >&2
  exit 1
fi

if ! find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.safetensors' -size +1M -print -quit | grep -q .; then
  echo "ERROR: no non-empty model SafeTensors file found" >&2
  exit 1
fi

echo "Checkpoint: ${MODEL_DIR}"
du -sh "${MODEL_DIR}"

for file in \
  train_config.json \
  config.json \
  policy_preprocessor.json \
  policy_postprocessor.json; do
  if [[ -s "${MODEL_DIR}/${file}" ]]; then
    echo "OK: ${file}"
  else
    echo "NOTICE: ${file} was not found in the checkpoint root"
  fi
done

echo "CHECKPOINT_OK"
