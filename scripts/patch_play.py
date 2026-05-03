"""Patch isaac_so_arm101 play.py for compatibility with Isaac Lab 2.3.2.

isaac_so_arm101 main HEAD imports isaaclab.utils.pretrained_checkpoint, which
does not exist in the NGC isaac-lab:2.3.2 base image (the module was added
upstream after that release). The function is only invoked when the
--use_pretrained_checkpoint flag is set, which this project does not use.

This script wraps the import in try/except so the play script loads correctly
on Isaac Lab 2.3.2.

Idempotent: re-running the script on an already-patched file is a no-op.
"""

import re
from pathlib import Path

PLAY_PY = Path("/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py")
NEW = """try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*a, **k):
        return None"""

# Match only when the import statement appears as a top-level (non-indented)
# line. After patching the same import becomes 4-space indented inside the
# try block, which this anchored pattern intentionally does not match.
OLD_LINE_RE = re.compile(
    r"^from isaaclab\.utils\.pretrained_checkpoint import get_published_pretrained_checkpoint$",
    re.MULTILINE,
)
PATCHED_MARKER = "except ImportError:\n    def get_published_pretrained_checkpoint"


def main() -> None:
    src = PLAY_PY.read_text()
    if OLD_LINE_RE.search(src):
        new_src, count = OLD_LINE_RE.subn(NEW, src, count=1)
        PLAY_PY.write_text(new_src)
        print(f"[patch_play.py] Patched ({count} occurrence): {PLAY_PY}")
        return
    if PATCHED_MARKER in src:
        print(f"[patch_play.py] Already patched: {PLAY_PY}")
        return
    raise SystemExit(
        f"[patch_play.py] Pattern not found in {PLAY_PY}. "
        "isaac_so_arm101 may have changed; update this patch script."
    )


if __name__ == "__main__":
    main()
