FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV SAGEMAKER_PROGRAM=train.py

WORKDIR /opt/ml/code

# ffmpeg is required by Isaac Lab's RecordVideo wrapper (used when running
# play.py with --video to render the trained policy to mp4 in headless mode).
# The NGC isaac-lab base image does not ship ffmpeg.
#
# python3 is added to run scripts/patch_play.py at build time. The base
# image bundles Python under /workspace/isaaclab/_isaac_sim, but does not
# expose a `python3` command on PATH, which is awkward for build-time
# patching scripts.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3 \
 && rm -rf /var/lib/apt/lists/*

ARG ISAAC_SO_ARM101_REF=main

RUN git clone --depth 1 --branch ${ISAAC_SO_ARM101_REF} \
    https://github.com/MuammerBay/isaac_so_arm101.git /opt/isaac_so_arm101 \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install -e /opt/isaac_so_arm101 --no-deps \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install boto3

# Patch isaac_so_arm101 play.py: main HEAD imports
# isaaclab.utils.pretrained_checkpoint, which does not exist in
# isaac-lab:2.3.2. We do not use the --use_pretrained_checkpoint flag,
# so wrap that import in try/except. Verify the patch landed by grepping
# the unique try/except marker, so a silent skip fails the build loudly.
COPY scripts/patch_play.py /tmp/patch_play.py
RUN python3 /tmp/patch_play.py \
 && grep -q "^except ImportError:" /opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
 && rm /tmp/patch_play.py

# Replace the goal pose visualizer in the Reach env from the default frame
# marker (XYZ axes, hard to read in playback videos) with a small red
# sphere. The Reach reward is position-only, so a sphere conveys the goal
# clearly without losing information. Verified by grepping the marker name.
COPY scripts/patch_reach_visualizer.py /tmp/patch_reach_visualizer.py
RUN python3 /tmp/patch_reach_visualizer.py \
 && grep -q "GOAL_SPHERE_MARKER_CFG" /opt/isaac_so_arm101/src/isaac_so_arm101/tasks/reach/reach_env_cfg.py \
 && rm /tmp/patch_reach_visualizer.py

COPY src/train.py src/play.py src/entrypoint.sh /opt/ml/code/

RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
