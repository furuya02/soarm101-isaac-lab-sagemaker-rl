"""Make pretrained_checkpoint import resilient across Isaac Lab versions.

Isaac Lab 2.3.x で `isaaclab.utils.pretrained_checkpoint` が
`isaaclab_rl.utils.pretrained_checkpoint` に移動した一方、`isaac_so_arm101`
main HEAD は移動前の旧 path を import している。base image のバージョンに
よってどちらが存在するかが変わるため、3 段フォールバック（旧 path → 新
path → ダミー）で書き換える。
"""

import re
from pathlib import Path

PLAY_PY = Path("/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py")
NEW = """try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    try:
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
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
elif "from isaaclab_rl.utils.pretrained_checkpoint" not in src:
    raise SystemExit(f"pattern not found in {PLAY_PY}")
