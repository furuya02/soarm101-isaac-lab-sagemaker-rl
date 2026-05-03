"""Patch isaac_so_arm101 reach_env_cfg.py for a clearer playback video.

Three changes:

1. Goal pose visualizer: switch from the default frame marker (XYZ axes) to
   a magenta sphere. Reach uses position-only reward (orientation reward
   weight is 0), so a sphere conveys the goal clearly. Magenta is chosen so
   the marker does not collide with the red/green/blue/yellow already on
   screen (frame axes and the SO-ARM101 yellow body).

2. Camera position: move the default viewer from (2.5, 2.5, 1.5) closer to
   (1.0, 0.8, 0.6) and lookat (0.0, -0.15, 0.5), framing a single arm and
   its goal larger in the recorded video.

3. Current pose (end-effector frame) keeps the default frame marker so the
   arm pose is still visible relative to the goal sphere.

Idempotent: re-running on an already-patched file is a no-op.
"""

from pathlib import Path

CFG_PY = Path("/opt/isaac_so_arm101/src/isaac_so_arm101/tasks/reach/reach_env_cfg.py")

INSERT_BLOCK = """import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

GOAL_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_sphere",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    },
)


"""

ANCHOR_PRE_INSERT = "@configclass\nclass CommandsCfg:"
INLINE_KEY = "        debug_vis=True,\n"
INLINE_NEW = (
    "        debug_vis=True,\n"
    "        goal_pose_visualizer_cfg=GOAL_SPHERE_MARKER_CFG,\n"
)

VIEWER_OLD = "        self.viewer.eye = (2.5, 2.5, 1.5)"
VIEWER_NEW = (
    "        self.viewer.eye = (1.5, 1.5, 1.0)\n"
    "        self.viewer.lookat = (0.0, 0.0, 0.4)"
)

PATCHED_MARKER = "GOAL_SPHERE_MARKER_CFG"


def main() -> None:
    src = CFG_PY.read_text()
    if PATCHED_MARKER in src:
        print(f"[patch_reach_visualizer.py] Already patched: {CFG_PY}")
        return
    if ANCHOR_PRE_INSERT not in src:
        raise SystemExit(f"Anchor not found: {ANCHOR_PRE_INSERT!r}")
    if INLINE_KEY not in src:
        raise SystemExit(f"Inline anchor not found: {INLINE_KEY!r}")
    if VIEWER_OLD not in src:
        raise SystemExit(f"Viewer anchor not found: {VIEWER_OLD!r}")
    src = src.replace(ANCHOR_PRE_INSERT, INSERT_BLOCK + ANCHOR_PRE_INSERT, 1)
    src = src.replace(INLINE_KEY, INLINE_NEW, 1)
    src = src.replace(VIEWER_OLD, VIEWER_NEW, 1)
    CFG_PY.write_text(src)
    print(f"[patch_reach_visualizer.py] Patched: {CFG_PY}")


if __name__ == "__main__":
    main()
