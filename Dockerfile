FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

WORKDIR /opt/ml/code

# ffmpeg: required by Isaac Lab RecordVideo. python3: needed for build-time patches.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3 \
 && rm -rf /var/lib/apt/lists/*

# 本記事執筆時点の main HEAD (2025-12-22) に pin。upstream の不意の更新で
# 手順が壊れないようにするため。別 ref を試したい場合は
# --build-arg ISAAC_SO_ARM101_REF=<commit_or_branch> で上書き可能。
ARG ISAAC_SO_ARM101_REF=e4624dea075b00a36dbc66bebd531d191c92e8cd
RUN git clone https://github.com/MuammerBay/isaac_so_arm101.git /opt/isaac_so_arm101 \
 && git -C /opt/isaac_so_arm101 checkout ${ISAAC_SO_ARM101_REF} \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install -e /opt/isaac_so_arm101 --no-deps \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install boto3

COPY scripts/patch_play.py scripts/patch_reach_visualizer.py /tmp/
RUN python3 /tmp/patch_play.py \
 && python3 /tmp/patch_reach_visualizer.py \
 && rm /tmp/patch_play.py /tmp/patch_reach_visualizer.py

COPY src/train.py src/play.py src/entrypoint.sh /opt/ml/code/
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
