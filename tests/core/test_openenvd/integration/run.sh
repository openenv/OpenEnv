#!/usr/bin/env bash
# Build the integration image and run the live-unit tests twice:
#   privileged:   openenvd isolates (as it would inside a microVM), and the
#                 cgroup tests run against the real kernel;
#   unprivileged: a plain container, where openenvd must step aside and serve
#                 the environment exactly as before.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
docker build -f "$root/tests/core/test_openenvd/integration/Dockerfile" -t openenvd-it "$root"
docker run --rm --privileged --cgroupns=private \
    --tmpfs /var/lib/openenvd:exec --tmpfs /tmp:exec \
    -e OPENENVD_TEST_ADOPT_CGROUP_ROOT=1 \
    openenvd-it python -m pytest -q -p no:cacheprovider -o asyncio_mode=auto \
    /opt/tests/test_cgroups.py /opt/tests/test_live_unit.py "$@"
docker run --rm -e OPENENVD_IT_MODE=unprivileged --tmpfs /tmp:exec \
    openenvd-it python -m pytest -q -p no:cacheprovider /opt/tests/test_live_unit.py "$@"
