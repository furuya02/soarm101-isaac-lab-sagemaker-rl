"""Wrap isaaclab.utils.pretrained_checkpoint import in try/except for Isaac Lab 2.3.2."""

import re
from pathlib import Path

PLAY_PY = Path("/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py")
NEW = """try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*a, **k):
        return None"""

OLD_LINE_RE = re.compile(
    r"^from isaaclab\.utils\.pretrained_checkpoint import get_published_pretrained_checkpoint$",
    re.MULTILINE,
)

src = PLAY_PY.read_text()
if OLD_LINE_RE.search(src):
    PLAY_PY.write_text(OLD_LINE_RE.sub(NEW, src, count=1))
elif "except ImportError:\n    def get_published_pretrained_checkpoint" not in src:
    raise SystemExit(f"pattern not found in {PLAY_PY}")
