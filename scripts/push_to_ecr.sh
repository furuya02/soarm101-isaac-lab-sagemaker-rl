#!/usr/bin/env bash
# Build and push the training image to ECR. Usage: ./scripts/push_to_ecr.sh [tag]
set -euo pipefail

REGION="${AWS_REGION:-ap-northeast-1}"
PROJECT_NAME="soarm101-isaac-lab-sagemaker-rl"
TAG="${1:-latest}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_HOST="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_HOST}/${PROJECT_NAME}:${TAG}"

aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_HOST}"
docker build --platform linux/amd64 -t "${PROJECT_NAME}:${TAG}" .
docker tag "${PROJECT_NAME}:${TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"
echo "${IMAGE_URI}"
