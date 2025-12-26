#!/bin/bash

TASK_ID="qwen-test-$(date +%s)"
MODEL="gradients-io-tournaments/Qwen2-VL-7B-Instruct" # Ganti dengan repo Qwen yang mas tuju
DATASET_ZIP="https://gradients.s3.eu-north-1.amazonaws.com/dc9853fb35c40bd4_train_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20251221%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251221T212609Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=865abddfccce78e1964b0abb468c3fc7a591019820d3a3024f7d4220757da588"
MODEL_TYPE="qwen-image"
EXPECTED_REPO_NAME="test_qwen_image_1"

HUGGINGFACE_USERNAME=""
HUGGINGFACE_TOKEN=""
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"

echo "🚀 Downloading model Qwen and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET_ZIP" \
  --task-type "ImageTask" \
  --model-type "$MODEL_TYPE"

echo "🔥 Starting Qwen Image Training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=32g \
  --cpus=8 \
  --network none \
  --env TRANSFORMERS_CACHE=/cache/hf_cache \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$CHECKPOINTS_DIR/datasets:/dataset:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name image-trainer-example \
  standalone-image-toolkit-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset-zip "$DATASET_ZIP" \
  --model-type "$MODEL_TYPE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --hours-to-complete 1
