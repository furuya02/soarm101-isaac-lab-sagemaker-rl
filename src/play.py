"""SageMaker eval entrypoint: render a trained policy to mp4 (headless).

Required environment:
- MODEL_S3_URI : full S3 URI to the model.tar.gz produced by a train run,
                 e.g. s3://bucket/output/<job>/output/model.tar.gz

Optional:
- TASK_NAME    : default Isaac-SO-ARM101-Reach-Play-v0
- NUM_ENVS     : default 4
- VIDEO_LENGTH : steps to record, default 200

Outputs:
- /opt/ml/model/videos/*.mp4  (uploaded to S3 by SageMaker on completion)
- /opt/ml/model/<source-checkpoint-name>.pt  (the checkpoint used for eval)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3

ISAACLAB_DIR: Path = Path("/workspace/isaaclab")
MODEL_DIR: Path = Path("/opt/ml/model")
WORK_DIR: Path = Path("/opt/ml/code/play_work")

TASK_NAME: str = os.environ.get("TASK_NAME", "Isaac-SO-ARM101-Reach-Play-v0")
NUM_ENVS: str = os.environ.get("NUM_ENVS", "4")
VIDEO_LENGTH: str = os.environ.get("VIDEO_LENGTH", "200")


def download_and_extract_model(s3_uri: str, dest: Path) -> Path:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"MODEL_S3_URI must be an s3:// URI, got: {s3_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / "model.tar.gz"
    print(f"[play.py] Downloading {s3_uri} -> {tarball}", flush=True)
    boto3.client("s3").download_file(bucket, key, str(tarball))

    print(f"[play.py] Extracting into {dest}", flush=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest)
    return dest


def latest_checkpoint(model_root: Path) -> Optional[Path]:
    candidates = sorted(model_root.rglob("model_*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return candidates[-1]


def find_videos_dir() -> Optional[Path]:
    for root in [ISAACLAB_DIR / "logs", Path("/opt/ml/code/logs")]:
        if not root.exists():
            continue
        for path in root.rglob("videos"):
            if path.is_dir() and any(path.iterdir()):
                return path
    return None


def main() -> int:
    model_s3_uri = os.environ.get("MODEL_S3_URI")
    if not model_s3_uri:
        print("[play.py] ERROR: MODEL_S3_URI is required.", flush=True)
        return 2

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    extracted = download_and_extract_model(model_s3_uri, WORK_DIR)
    ckpt = latest_checkpoint(extracted)
    if ckpt is None:
        print("[play.py] ERROR: No model_*.pt found in the downloaded tarball.", flush=True)
        return 3
    print(f"[play.py] Using checkpoint: {ckpt}", flush=True)

    cmd: list[str] = [
        str(ISAACLAB_DIR / "isaaclab.sh"), "-p",
        "/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py",
        "--task", TASK_NAME,
        "--headless",
        "--video",
        "--video_length", VIDEO_LENGTH,
        "--num_envs", NUM_ENVS,
        "--checkpoint", str(ckpt),
    ]
    print(f"[play.py] Launching (cwd={ISAACLAB_DIR}): {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ISAACLAB_DIR))
    print(f"[play.py] Inner play.py exited with code {proc.returncode}", flush=True)

    videos_dir = find_videos_dir()
    if videos_dir is not None:
        dst = MODEL_DIR / "videos"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(videos_dir, dst)
        print(f"[play.py] Copied {videos_dir} -> {dst}", flush=True)
    else:
        print("[play.py] WARNING: videos directory not found.", flush=True)

    shutil.copy2(ckpt, MODEL_DIR / ckpt.name)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
