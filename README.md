# soarm101-isaac-lab-sagemaker-rl

Sample code for training the SO-ARM101 Reach task with Isaac Lab on Amazon SageMaker Training Job + Managed Spot Training.

Companion blog post: [SO-ARM101 with Isaac Lab on SageMaker Training Job (Managed Spot)](https://dev.classmethod.jp/articles/) (link to be filled in after publication).

> Japanese version: [README.ja.md](README.ja.md)

## Overview

- Base image: `nvcr.io/nvidia/isaac-lab:2.3.2` (Isaac Lab 2.3.2 + Isaac Sim 5.1.x)
- Task: `Isaac-SO-ARM101-Reach-v0` (from [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) v1.2.0)
- Training: SageMaker Training Job, ml.g6.2xlarge (NVIDIA L4 24 GB), Managed Spot
- Region: `ap-northeast-1`

## Repository layout

```
.
├── cdk/                    # AWS CDK (TypeScript): S3, ECR, IAM Role, Budget
├── scripts/
│   └── push_to_ecr.sh      # Build & push the training image to ECR
├── src/
│   ├── train.py            # SageMaker entrypoint (SIGTERM forwarding, ckpt resume)
│   └── entrypoint.sh
├── Dockerfile              # Inherits NGC isaac-lab:2.3.2
├── submit.py               # SageMaker Estimator launcher
└── README.md / README.ja.md
```

## Prerequisites

- AWS account with SageMaker / S3 / ECR / Budgets access
- AWS CLI v2 configured for `ap-northeast-1`
- Docker (with the `linux/amd64` build platform available)
- Node.js 20.x and AWS CDK v2 (`npm i -g aws-cdk`)
- An NVIDIA NGC account and API key (`docker login nvcr.io`)
- Python 3.11 with the SageMaker Python SDK (`pip install sagemaker`)

## Setup

### 1. Clone

```bash
git clone https://github.com/furuya02/soarm101-isaac-lab-sagemaker-rl.git
cd soarm101-isaac-lab-sagemaker-rl
```

### 2. Deploy AWS resources with CDK

```bash
cd cdk
npm install

export AWS_REGION=ap-northeast-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

npx cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}
npx cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c region=${AWS_REGION}
```

The stack creates:

- S3 bucket: `soarm101-isaac-lab-sagemaker-rl-<ACCOUNT_ID>`
- ECR repository: `soarm101-isaac-lab-sagemaker-rl`
- IAM role: `soarm101-isaac-lab-sagemaker-rl-sagemaker-execution-role`
- Monthly Budget alert (USD 100, 10/50/90 % thresholds, email to `hirauchi.shinichi@classmethod.jp`)

To override the bucket suffix or budget email:

```bash
npx cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c bucket_suffix=20260503 \
  -c budget_email=you@example.com
```

### 3. Build & push the training image to ECR

```bash
cd ..

# Log in to NGC (NGC API key required)
docker login nvcr.io

./scripts/push_to_ecr.sh
```

The first push transfers ~15 GB and takes 30-60 minutes depending on your uplink. SageMaker pulls the image from the same region, so there is no inter-region data transfer charge.

### 4. Submit a SageMaker Training Job

```bash
export SAGEMAKER_ROLE_ARN=$(aws iam get-role \
  --role-name soarm101-isaac-lab-sagemaker-rl-sagemaker-execution-role \
  --query 'Role.Arn' --output text)
export ECR_IMAGE_URI=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/soarm101-isaac-lab-sagemaker-rl:latest
export S3_BUCKET=soarm101-isaac-lab-sagemaker-rl-${ACCOUNT_ID}

# On-demand single run for sanity check
USE_SPOT=false MAX_RUN_HOURS=2 python submit.py

# Managed Spot run
USE_SPOT=true MAX_RUN_HOURS=2 MAX_WAIT_HOURS=6 python submit.py
```

### 5. Retrieve the trained model

```bash
JOB_NAME=<job-name-from-submit.py-output>
aws s3 cp s3://${S3_BUCKET}/output/${JOB_NAME}/output/model.tar.gz .
tar xzf model.tar.gz
# rsl_rl/<task>/<run>/model_<iter>.pt
```

## Cost estimate (ap-northeast-1, May 2026)

| Resource | Estimated cost |
|---|---|
| ml.g6.2xlarge (on-demand) | ~ $1.81 / hour |
| ml.g6.2xlarge (Managed Spot, 70 % off) | ~ $0.54 / hour |
| ECR storage (15 GB image) | ~ $1.50 / month |
| S3 (artifacts + checkpoints, < 1 GB) | < $0.10 / month |

A typical Reach training run (`--num_envs 64`, `--max_iterations 1000`) is estimated at well under USD 5 per attempt with Managed Spot.

## Cleanup

```bash
cd cdk
npx cdk destroy
# Empty the S3 bucket and delete ECR images manually if you want to remove them too.
```

## Caveats

- **Managed Spot requires checkpoints.** Without a checkpoint implementation, `max_wait` is capped at 1 hour. `src/train.py` resumes from `/opt/ml/checkpoints/model_*.pt` automatically.
- **`max_run` is mandatory.** Always set this to prevent runaway training charges.
- **Region pinning.** Keep ECR, S3 and SageMaker all in `ap-northeast-1` to avoid inter-region data transfer fees.
- **Image size.** The base image is ~15 GB. SageMaker pulls the image at job start, adding 5-10 minutes of overhead per job.

## License

This sample code is released under the MIT License.

`isaac_so_arm101` is BSD-3-Clause. NVIDIA Isaac Sim and Isaac Lab follow the [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html).
