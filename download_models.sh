#!/usr/bin/env bash
set -euxo pipefail

MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DEST_ROOT="models"
DEST_NOTEBOOK="Notebook/models"

mkdir -p "${DEST_ROOT}" "${DEST_NOTEBOOK}"

if [[ ! -f "${DEST_ROOT}/sam_vit_h_4b8939.pth" ]]; then
  echo "Downloading SAM ViT-H checkpoint to ${DEST_ROOT}..."
  curl -L "${MODEL_URL}" -o "${DEST_ROOT}/sam_vit_h_4b8939.pth"
else
  echo "Checkpoint already exists at ${DEST_ROOT}/sam_vit_h_4b8939.pth; skipping download."
fi

cp -f "${DEST_ROOT}/sam_vit_h_4b8939.pth" "${DEST_NOTEBOOK}/"
