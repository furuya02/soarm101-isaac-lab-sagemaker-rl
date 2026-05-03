#!/usr/bin/env bash
set -eu

MODE="${MODE:-train}"

case "${MODE}" in
  train)
    exec /workspace/isaaclab/isaaclab.sh -p /opt/ml/code/train.py "$@"
    ;;
  play)
    exec /workspace/isaaclab/isaaclab.sh -p /opt/ml/code/play.py "$@"
    ;;
  *)
    echo "[entrypoint.sh] ERROR: unknown MODE=${MODE} (expected train|play)" >&2
    exit 2
    ;;
esac
