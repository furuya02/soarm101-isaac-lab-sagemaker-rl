FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV SAGEMAKER_PROGRAM=train.py

WORKDIR /opt/ml/code

# ffmpeg: required by Isaac Lab RecordVideo. python3: needed for build-time patches.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3 \
 && rm -rf /var/lib/apt/lists/*

ARG ISAAC_SO_ARM101_REF=main
RUN git clone --depth 1 --branch ${ISAAC_SO_ARM101_REF} \
    https://github.com/MuammerBay/isaac_so_arm101.git /opt/isaac_so_arm101 \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install -e /opt/isaac_so_arm101 --no-deps \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install boto3

COPY scripts/patch_play.py scripts/patch_reach_visualizer.py /tmp/
RUN python3 /tmp/patch_play.py \
 && python3 /tmp/patch_reach_visualizer.py \
 && rm /tmp/patch_play.py /tmp/patch_reach_visualizer.py

COPY src/train.py src/play.py src/entrypoint.sh /opt/ml/code/
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
