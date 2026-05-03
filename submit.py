"""Submit a SageMaker Training Job for SO-ARM101 Reach with Managed Spot.

Required environment variables:
- SAGEMAKER_ROLE_ARN : SageMaker execution role ARN
- ECR_IMAGE_URI      : Full ECR URI of the training image
- S3_BUCKET          : S3 bucket name for output / checkpoints

Optional:
- INSTANCE_TYPE      : default ml.g5.2xlarge
- USE_SPOT           : default true
- NUM_ENVS           : default 64
- MAX_ITERATIONS     : default 1000
- MAX_RUN_HOURS      : default 2
- MAX_WAIT_HOURS     : default 6

Note: ml.g5.2xlarge is chosen as the default because new AWS accounts often
have a quota of 1 for it, while ml.g6 (L4) family quotas are frequently 0
until manually requested. To use g6, request the quota first and pass
`INSTANCE_TYPE=ml.g6.2xlarge`.
"""

from __future__ import annotations

import os
import time
from typing import Any

import sagemaker
from sagemaker.estimator import Estimator


def env_bool(name: str, default: bool) -> bool:
    raw: str = os.environ.get(name, str(default)).lower()
    return raw in ("1", "true", "yes", "on")


def main() -> None:
    role: str = os.environ["SAGEMAKER_ROLE_ARN"]
    image_uri: str = os.environ["ECR_IMAGE_URI"]
    bucket: str = os.environ["S3_BUCKET"]

    instance_type: str = os.environ.get("INSTANCE_TYPE", "ml.g5.2xlarge")
    use_spot: bool = env_bool("USE_SPOT", True)
    num_envs: str = os.environ.get("NUM_ENVS", "64")
    max_iterations: str = os.environ.get("MAX_ITERATIONS", "1000")
    max_run_hours: int = int(os.environ.get("MAX_RUN_HOURS", "2"))
    max_wait_hours: int = int(os.environ.get("MAX_WAIT_HOURS", "6"))

    job_name: str = f"soarm101-reach-{int(time.time())}"

    estimator_kwargs: dict[str, Any] = dict(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        output_path=f"s3://{bucket}/output/",
        environment={
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "TASK_NAME": "Isaac-SO-ARM101-Reach-v0",
            "NUM_ENVS": num_envs,
            "MAX_ITERATIONS": max_iterations,
            "EXPERIMENT_NAME": "so_arm101_reach",
        },
    )

    if use_spot:
        estimator_kwargs.update(
            use_spot_instances=True,
            max_run=max_run_hours * 3600,
            max_wait=max_wait_hours * 3600,
            checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/",
            checkpoint_local_path="/opt/ml/checkpoints",
        )
    else:
        estimator_kwargs.update(max_run=max_run_hours * 3600)

    estimator = Estimator(**estimator_kwargs)

    print(f"[submit.py] Job name      : {job_name}")
    print(f"[submit.py] Image         : {image_uri}")
    print(f"[submit.py] Instance      : {instance_type}")
    print(f"[submit.py] Use Spot      : {use_spot}")
    print(f"[submit.py] max_run hours : {max_run_hours}")
    if use_spot:
        print(f"[submit.py] max_wait hours: {max_wait_hours}")
    print(f"[submit.py] Output path   : s3://{bucket}/output/")

    estimator.fit(job_name=job_name, wait=False)
    print(f"[submit.py] Submitted. Track via: aws sagemaker describe-training-job --training-job-name {job_name}")


if __name__ == "__main__":
    main()
