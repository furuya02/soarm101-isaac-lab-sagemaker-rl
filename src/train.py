"""SageMaker Training Job entrypoint for SO-ARM101 Reach with Isaac Lab + RSL-RL.

- Resumes from /opt/ml/checkpoints/<latest> if present (Managed Spot recovery).
- Forwards SIGTERM to the inner training process so RSL-RL flushes a checkpoint
  before SageMaker terminates the container (Managed Spot 2-minute grace period).
- Copies final logs to /opt/ml/model/ for automatic upload to S3.
"""

from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

CKPT_DIR: Path = Path("/opt/ml/checkpoints")
MODEL_DIR: Path = Path("/opt/ml/model")
LOG_DIR: Path = Path("/workspace/isaaclab/logs/rsl_rl")

TASK_NAME: str = os.environ.get("TASK_NAME", "Isaac-SO-ARM101-Reach-v0")
NUM_ENVS: str = os.environ.get("NUM_ENVS", "64")
MAX_ITERATIONS: str = os.environ.get("MAX_ITERATIONS", "1000")
EXPERIMENT_NAME: str = os.environ.get("EXPERIMENT_NAME", "so_arm101_reach")


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("model_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> int:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    resume_args: list[str] = []
    ckpt = latest_checkpoint(CKPT_DIR)
    if ckpt is not None:
        print(f"[train.py] Resuming from checkpoint: {ckpt}", flush=True)
        resume_args = ["--resume", "--checkpoint", str(ckpt)]
    else:
        print("[train.py] No checkpoint found, starting fresh.", flush=True)

    cmd: list[str] = [
        "/workspace/isaaclab/isaaclab.sh", "-p",
        "/opt/isaac_so_arm101/scripts/rsl_rl/train.py",
        "--task", TASK_NAME,
        "--headless",
        "--num_envs", NUM_ENVS,
        "--max_iterations", MAX_ITERATIONS,
        "--logger", "tensorboard",
        "--experiment_name", EXPERIMENT_NAME,
        *resume_args,
    ]
    print(f"[train.py] Launching: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd)

    def forward_sigterm(signum: int, frame: object) -> None:
        print(f"[train.py] Received signal {signum}, forwarding to child.", flush=True)
        proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward_sigterm)

    return_code: int = proc.wait()
    print(f"[train.py] Inner train.py exited with code {return_code}", flush=True)

    if LOG_DIR.exists():
        dst = MODEL_DIR / "rsl_rl"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(LOG_DIR, dst)
        print(f"[train.py] Copied {LOG_DIR} -> {dst}", flush=True)

    if CKPT_DIR.exists():
        for ckpt_file in CKPT_DIR.glob("model_*.pt"):
            shutil.copy2(ckpt_file, MODEL_DIR / ckpt_file.name)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
