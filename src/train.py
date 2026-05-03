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
ISAACLAB_DIR: Path = Path("/workspace/isaaclab")

TASK_NAME: str = os.environ.get("TASK_NAME", "Isaac-SO-ARM101-Reach-v0")
NUM_ENVS: str = os.environ.get("NUM_ENVS", "64")
MAX_ITERATIONS: str = os.environ.get("MAX_ITERATIONS", "1000")
EXPERIMENT_NAME: str = os.environ.get("EXPERIMENT_NAME", "so_arm101_reach")


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("model_*.pt"))
    return ckpts[-1] if ckpts else None


def find_log_dir() -> Optional[Path]:
    candidates = [
        ISAACLAB_DIR / "logs" / "rsl_rl",
        Path("/opt/ml/code/logs/rsl_rl"),
        Path.cwd() / "logs" / "rsl_rl",
    ]
    for c in candidates:
        if c.exists() and c.is_dir() and any(c.iterdir()):
            print(f"[train.py] Found log dir at: {c}", flush=True)
            return c
    for root in [Path("/workspace"), Path("/opt/ml/code"), Path("/opt/isaac_so_arm101")]:
        if not root.exists():
            continue
        try:
            for path in root.rglob("logs/rsl_rl"):
                if path.is_dir() and any(path.iterdir()):
                    print(f"[train.py] Discovered log dir via rglob: {path}", flush=True)
                    return path
        except (PermissionError, OSError):
            continue
    return None


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
        str(ISAACLAB_DIR / "isaaclab.sh"), "-p",
        "/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py",
        "--task", TASK_NAME,
        "--headless",
        "--num_envs", NUM_ENVS,
        "--max_iterations", MAX_ITERATIONS,
        "--logger", "tensorboard",
        "--experiment_name", EXPERIMENT_NAME,
        *resume_args,
    ]
    print(f"[train.py] Launching (cwd={ISAACLAB_DIR}): {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, cwd=str(ISAACLAB_DIR))

    def forward_sigterm(signum: int, frame: object) -> None:
        print(f"[train.py] Received signal {signum}, forwarding to child.", flush=True)
        proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward_sigterm)

    return_code: int = proc.wait()
    print(f"[train.py] Inner train.py exited with code {return_code}", flush=True)

    log_dir = find_log_dir()
    if log_dir is not None:
        dst = MODEL_DIR / "rsl_rl"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(log_dir, dst)
        print(f"[train.py] Copied {log_dir} -> {dst}", flush=True)
    else:
        print("[train.py] WARNING: rsl_rl log directory not found anywhere.", flush=True)

    for ckpt_file in CKPT_DIR.glob("model_*.pt"):
        shutil.copy2(ckpt_file, MODEL_DIR / ckpt_file.name)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
