"""Patch isaac_so_arm101 play.py for compatibility with Isaac Lab 2.3.2.

isaac_so_arm101 main HEAD imports isaaclab.utils.pretrained_checkpoint, which
does not exist in the NGC isaac-lab:2.3.2 base image (the module was added
upstream after that release). The function is only invoked when the
--use_pretrained_checkpoint flag is set, which this project does not use.

This script wraps the import in try/except so the play script loads correctly
on Isaac Lab 2.3.2.
"""

from pathlib import Path

PLAY_PY = Path("/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py")
OLD = "from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint"
NEW = """try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*a, **k):
        return None"""


def main() -> None:
    src = PLAY_PY.read_text()
    if NEW.split("\n")[0] in src:
        print(f"[patch_play.py] Already patched: {PLAY_PY}")
        return
    if OLD not in src:
        raise SystemExit(
            f"[patch_play.py] Pattern not found in {PLAY_PY}. "
            "isaac_so_arm101 may have changed; update this patch script."
        )
    PLAY_PY.write_text(src.replace(OLD, NEW))
    print(f"[patch_play.py] Patched: {PLAY_PY}")


if __name__ == "__main__":
    main()
