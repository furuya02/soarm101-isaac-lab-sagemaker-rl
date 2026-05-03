"""Patch isaac_so_arm101 reach_env_cfg.py to use a sphere goal marker.

By default Isaac Lab UniformPoseCommandCfg renders the goal pose as XYZ axes
(FRAME_MARKER), which is hard to read in playback videos. The Reach task in
isaac_so_arm101 uses position-only reward (orientation reward weight is 0),
so a sphere position marker conveys the target more clearly.

Current pose (end-effector frame) keeps the default frame marker so we can
still see how the arm pose moves towards the goal.

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
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
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
    src = src.replace(ANCHOR_PRE_INSERT, INSERT_BLOCK + ANCHOR_PRE_INSERT, 1)
    src = src.replace(INLINE_KEY, INLINE_NEW, 1)
    CFG_PY.write_text(src)
    print(f"[patch_reach_visualizer.py] Patched: {CFG_PY}")


if __name__ == "__main__":
    main()
