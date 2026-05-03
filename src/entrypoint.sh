#!/usr/bin/env bash
set -eu

exec /workspace/isaaclab/isaaclab.sh -p /opt/ml/code/train.py "$@"
