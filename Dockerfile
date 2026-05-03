FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV SAGEMAKER_PROGRAM=train.py

WORKDIR /opt/ml/code

RUN git clone --depth 1 --branch v1.2.0 \
    https://github.com/MuammerBay/isaac_so_arm101.git /opt/isaac_so_arm101 \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install -e /opt/isaac_so_arm101 --no-deps

COPY src/train.py src/entrypoint.sh /opt/ml/code/

RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
