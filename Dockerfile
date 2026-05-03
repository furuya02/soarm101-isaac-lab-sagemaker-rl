FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV SAGEMAKER_PROGRAM=train.py

WORKDIR /opt/ml/code

# ffmpeg is required by Isaac Lab's RecordVideo wrapper (used when running
# play.py with --video to render the trained policy to mp4 in headless mode).
# The NGC isaac-lab base image does not ship ffmpeg, so install it explicitly.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

ARG ISAAC_SO_ARM101_REF=main

RUN git clone --depth 1 --branch ${ISAAC_SO_ARM101_REF} \
    https://github.com/MuammerBay/isaac_so_arm101.git /opt/isaac_so_arm101 \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install -e /opt/isaac_so_arm101 --no-deps \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install boto3

# Patch isaac_so_arm101 play.py: main HEAD imports
# isaaclab.utils.pretrained_checkpoint, which does not exist in
# isaac-lab:2.3.2. We do not use the --use_pretrained_checkpoint flag,
# so wrap that import in try/except.
COPY scripts/patch_play.py /tmp/patch_play.py
RUN python3 /tmp/patch_play.py && rm /tmp/patch_play.py

COPY src/train.py src/play.py src/entrypoint.sh /opt/ml/code/

RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
