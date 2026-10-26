#!/usr/bin/env bash
# Test Docker builds for OpenEnv environments.
#
# Builds each env image, starts the container, polls /health, then cleans up.
# Exits non-zero if any build or health check fails.
#
# Usage:
#   bash scripts/test-docker-builds.sh                  # test all envs
#   bash scripts/test-docker-builds.sh echo-env         # test one env
#   bash scripts/test-docker-builds.sh --no-base        # skip base image rebuild
#   SKIP_ENVS="chat-env finrl-env" bash scripts/test-docker-builds.sh
#
# Requirements: docker (with buildx)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="openenv-base:test"
HEALTH_TIMEOUT=120   # seconds to wait for /health to respond
HEALTH_INTERVAL=3    # seconds between polls
START_PORT=18000     # base port; each env gets its own to allow parallelism

# ---------------------------------------------------------------------------
# Environment matrix: name | context | dockerfile
# Format: "name:context:dockerfile"
# Keep this list in sync with the matrix in .github/workflows/docker-build.yml
# ---------------------------------------------------------------------------
declare -a ENVS=(
    "echo-env:envs/echo_env:envs/echo_env/server/Dockerfile"
    "chat-env:envs/chat_env:envs/chat_env/server/Dockerfile"
    "coding-env:.:envs/coding_env/server/Dockerfile"
    "connect4-env:envs/connect4_env:envs/connect4_env/server/Dockerfile"
    "chess-env:envs/chess_env:envs/chess_env/server/Dockerfile"
    "tbench2-env:envs/tbench2_env:envs/tbench2_env/server/Dockerfile"
    "textarena-env:envs/textarena_env:envs/textarena_env/server/Dockerfile"
    "maze-env:envs/maze_env:envs/maze_env/server/Dockerfile"
    "snake-env:envs/snake_env:envs/snake_env/server/Dockerfile"
    "browsergym-env:envs/browsergym_env:envs/browsergym_env/server/Dockerfile"
    "git-env:envs/git_env:envs/git_env/server/Dockerfile"
    "atari-env:envs/atari_env:envs/atari_env/server/Dockerfile"
    "sumo-rl-env:envs/sumo_rl_env:envs/sumo_rl_env/server/Dockerfile"
#   "finrl-env:envs/finrl_env:envs/finrl_env/server/Dockerfile"          # heavy deps, long build
#   "dipg-safety-env:envs/dipg_safety_env:envs/dipg_safety_env/server/Dockerfile"  # needs special runtime setup
    "unity-env:envs/unity_env:envs/unity_env/server/Dockerfile"
    "openapp-env:envs/openapp_env:envs/openapp_env/server/Dockerfile"
    "openspiel-env:envs/openspiel_env:envs/openspiel_env/server/Dockerfile"
)

# Envs that need special runtime deps or take very long — skip health check,
# only verify the build succeeds.
# Example:
# BUILD_ONLY_ENVS1="sumo-rl-env finrl-env"
BUILD_ONLY_ENVS=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${NC}[$(date +%H:%M:%S)] $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠ $*${NC}"; }
fail() { echo -e "${RED}[$(date +%H:%M:%S)] ✗ $*${NC}"; }

cleanup_container() {
    local cid="$1"
    docker rm -f "$cid" &>/dev/null || true
}

wait_for_health() {
    local url="$1" timeout="$2" interval="$3"
    local elapsed=0
    while (( elapsed < timeout )); do
        if curl -sf "$url" &>/dev/null; then
            return 0
        fi
        sleep "$interval"
        (( elapsed += interval ))
    done
    return 1
}

is_build_only() {
    local name="$1"
    for skip in $BUILD_ONLY_ENVS; do
        [[ "$skip" == "$name" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
BUILD_BASE=true
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --no-base) BUILD_BASE=false ;;
        --*) warn "Unknown flag: $arg" ;;
        *)   FILTER="$arg" ;;
    esac
done

SKIP_ENVS="${SKIP_ENVS:-}"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Build base image
# ---------------------------------------------------------------------------
if $BUILD_BASE; then
    log "Building base image: $BASE_IMAGE"
    docker build \
        -t "$BASE_IMAGE" \
        -f src/openenv/core/containers/images/Dockerfile \
        . \
        --quiet \
    && ok "Base image built" \
    || { fail "Base image build failed"; exit 1; }
else
    log "Skipping base image build (--no-base)"
fi

# ---------------------------------------------------------------------------
# Build and test each env
# ---------------------------------------------------------------------------
PASSED=()
FAILED=()
SKIPPED=()
port=$START_PORT

for entry in "${ENVS[@]}"; do
    IFS=':' read -r name context dockerfile <<< "$entry"

    # Filter to a single env if requested
    if [[ -n "$FILTER" && "$FILTER" != "$name" ]]; then
        continue
    fi

    # Skip envs listed in SKIP_ENVS
    if echo "$SKIP_ENVS" | grep -qw "$name"; then
        warn "Skipping $name (in SKIP_ENVS)"
        SKIPPED+=("$name")
        continue
    fi

    # Skip if dockerfile doesn't exist
    if [[ ! -f "$dockerfile" ]]; then
        warn "Skipping $name ($dockerfile not found)"
        SKIPPED+=("$name")
        continue
    fi

    image="openenv-test-${name}"
    log "─────────────────────────────────────────"
    log "Building $name"
    log "  context:    $context"
    log "  dockerfile: $dockerfile"

    if ! docker build \
            -t "$image" \
            -f "$dockerfile" \
            --build-arg "BASE_IMAGE=$BASE_IMAGE" \
            "$context" \
            2>&1 | sed 's/^/  /'; then
        fail "Build failed: $name"
        FAILED+=("$name (build)")
        continue
    fi
    ok "Build succeeded: $name"

    # Build-only envs: don't run a container
    if is_build_only "$name"; then
        ok "Build-only check passed: $name"
        PASSED+=("$name")
        (( port++ ))
        continue
    fi

    # Start container
    log "Starting container for $name on port $port"
    if ! cid=$(docker run -d --rm -p "${port}:8000" "$image" 2>&1) || [[ -z "$cid" ]]; then
        fail "Failed to start container: $name"
        FAILED+=("$name (start)")
        (( port++ )) || true
        continue
    fi

    # Poll health endpoint
    health_url="http://localhost:${port}/health"
    log "Waiting for $health_url (up to ${HEALTH_TIMEOUT}s)"
    if wait_for_health "$health_url" "$HEALTH_TIMEOUT" "$HEALTH_INTERVAL"; then
        ok "Health check passed: $name → $health_url"
        PASSED+=("$name")
    else
        fail "Health check timed out: $name"
        log "Container logs:"
        docker logs "$cid" 2>&1 | tail -30 | sed 's/^/  /'
        FAILED+=("$name (health)")
    fi

    cleanup_container "$cid"
    (( port++ ))
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════"
echo " Results"
echo "═══════════════════════════════════════════"
echo ""
if (( ${#PASSED[@]} > 0 )); then
    ok "Passed (${#PASSED[@]}): ${PASSED[*]}"
fi
if (( ${#SKIPPED[@]} > 0 )); then
    warn "Skipped (${#SKIPPED[@]}): ${SKIPPED[*]}"
fi
if (( ${#FAILED[@]} > 0 )); then
    fail "Failed (${#FAILED[@]}): ${FAILED[*]}"
    echo ""
    exit 1
fi
echo ""
ok "All checks passed."
