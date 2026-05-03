#!/usr/bin/env bash
#
# Build the Isaac Lab + SO-ARM101 training image and push it to ECR.
#
# Prerequisites:
#   - aws CLI configured for ap-northeast-1
#   - docker logged in to nvcr.io (NGC API key)
#   - cdk deploy has created the ECR repository
#
# Usage:
#   ./scripts/push_to_ecr.sh [tag]
#
set -euo pipefail

REGION="${AWS_REGION:-ap-northeast-1}"
PROJECT_NAME="soarm101-isaac-lab-sagemaker-rl"
TAG="${1:-latest}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_HOST="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_HOST}/${PROJECT_NAME}:${TAG}"

echo "[push_to_ecr.sh] Account : ${ACCOUNT_ID}"
echo "[push_to_ecr.sh] Region  : ${REGION}"
echo "[push_to_ecr.sh] Image   : ${IMAGE_URI}"

echo "[push_to_ecr.sh] Logging in to ECR ..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_HOST}"

echo "[push_to_ecr.sh] Building image (linux/amd64) ..."
docker build --platform linux/amd64 -t "${PROJECT_NAME}:${TAG}" .

echo "[push_to_ecr.sh] Tagging image ..."
docker tag "${PROJECT_NAME}:${TAG}" "${IMAGE_URI}"

echo "[push_to_ecr.sh] Pushing image (this can take 30-60 minutes for ~15 GB image) ..."
docker push "${IMAGE_URI}"

echo "[push_to_ecr.sh] Done. Image URI:"
echo "${IMAGE_URI}"
