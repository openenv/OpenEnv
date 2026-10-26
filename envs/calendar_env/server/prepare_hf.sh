#!/usr/bin/env bash
# Calendar example notebooks include local tunnel setup notes that can trigger
# Hub content filters. They are not needed by the runtime Space.

set -euo pipefail

DOCKERFILE_PATH="$1"
STAGE_DIR=$(cd "$(dirname "$DOCKERFILE_PATH")" && pwd)

rm -rf "$STAGE_DIR/client_notebooks"
rm -rf "$STAGE_DIR/envs/calendar_env/client_notebooks"
