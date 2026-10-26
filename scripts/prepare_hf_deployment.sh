#!/usr/bin/env bash

# OpenEnv Hugging Face Deployment Script
# - deploys one or all environments under envs/
# - pins OpenEnv git refs in environment pyproject.toml to a release version
# - uploads each staged env to Hugging Face Spaces
# - updates a versioned HF collection via scripts/manage_hf_collection.py

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/prepare_hf_deployment.sh [options]

Environment selection:
  --all                            Deploy all deployable envs under envs/ (default);
                                   skips envs containing SKIP_HF_DEPLOYMENT
  --env <name>                     Deploy a single env (repeatable); explicit
                                   selection ignores SKIP_HF_DEPLOYMENT

Deployment options:
  --base-sha <sha|tag>             openenv-base image ref suffix (default: latest)
  --hf-namespace <user|org>        HF namespace to deploy to (default: HF_NAMESPACE or openenv)
  --repo-id <owner/repo>           Exact HF Space repo to update (single-env only)
  --space-suffix <suffix>          Suffix appended to each space name
                                   (default: -<openenv-version>, e.g. -0.2.1)
  --private                        Create/update spaces as private (default)
  --public                         Create/update spaces as public
  --dry-run                        Prepare and print actions without uploading
  --staging-dir <path>             Staging root directory (default: hf-staging)
  --hub-tag <tag>                  Hub tag for generated README metadata (default: openenv)

Version pinning:
  --openenv-version <ref>          Pin OpenEnv git refs to this tag/ref
                                   (default: latest vX.Y.Z release tag, else main)

Collection update:
  --collection-namespace <owner>   Collection owner namespace (default: openenv)
  --collection-slug <slug>         Explicit collection slug override
  --skip-collection                Skip collection update step

Compatibility positional args:
  scripts/prepare_hf_deployment.sh <env_name> [base_image_sha]
EOF
}

log() {
    echo "[hf-deploy] $*"
}

warn() {
    echo "[hf-deploy][warn] $*" >&2
}

error() {
    echo "[hf-deploy][error] $*" >&2
    exit 1
}

# Cross-platform sed in-place editing:
# BSD sed (macOS) requires -i '', GNU sed (Linux) requires -i
sed_inplace() {
    local expression="$1"
    local target_file="$2"
    if sed --version >/dev/null 2>&1; then
        sed -i "$expression" "$target_file"
    else
        sed -i '' "$expression" "$target_file"
    fi
}

ensure_trailing_newline() {
    local target_file="$1"
    [ -f "$target_file" ] || return 0

    # If the last byte is not a newline, append one so future appends don't
    # accidentally land on the same line (common with Dockerfiles missing EOF \n).
    if [ -n "$(tail -c 1 "$target_file" 2>/dev/null || true)" ]; then
        printf "\n" >> "$target_file"
    fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

SPACE_SUFFIX_EXPLICIT=false
if [ "${SPACE_SUFFIX+x}" = "x" ]; then
    SPACE_SUFFIX_EXPLICIT=true
fi

BASE_IMAGE_SHA=""
BASE_IMAGE_REF=""
HF_NAMESPACE="${HF_NAMESPACE:-}"
SPACE_REPO_OVERRIDE="${SPACE_REPO_OVERRIDE:-}"
SPACE_SUFFIX="${SPACE_SUFFIX:-}"
STAGING_DIR="hf-staging"
HUB_TAG="openenv"
# Latest stable release tag reachable from HEAD (vX.Y.Z -> X.Y.Z, skipping
# pre-releases like vX.Y.Zrc1); the pyproject version on main is an untagged .devN.
DEFAULT_OPENENV_VERSION=$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude 'v*[!0-9.]*' 2>/dev/null || true)
DEFAULT_OPENENV_VERSION="${DEFAULT_OPENENV_VERSION#v}"
if [ -z "$DEFAULT_OPENENV_VERSION" ]; then
    DEFAULT_OPENENV_VERSION="main"
fi
OPENENV_VERSION="${OPENENV_VERSION:-$DEFAULT_OPENENV_VERSION}"
OPENENV_GIT_REF="${OPENENV_GIT_REF:-}"
COLLECTION_NAMESPACE="${COLLECTION_NAMESPACE:-openenv}"
COLLECTION_SLUG="${COLLECTION_SLUG:-}"
PRIVATE=true
DRY_RUN=false
DEPLOY_ALL=true
SKIP_COLLECTION=false

SELECTED_ENVS=()
DEPLOYED_SPACES=()
FAILED_ENVS=()
SKIPPED_ENVS=()
TOKEN_ARGS=()

normalized_version_suffix() {
    local raw_version="$1"
    local normalized=""
    raw_version="${raw_version#v}"
    normalized=$(printf "%s" "$raw_version" | tr -cs '[:alnum:].' '-')
    normalized="${normalized#-}"
    normalized="${normalized%-}"

    if [ -z "$normalized" ]; then
        error "Could not derive a space suffix from version '$raw_version'"
    fi

    printf "%s" "$normalized"
}

resolve_openenv_git_ref() {
    local requested_ref="$1"
    local repo_url="https://github.com/huggingface/OpenEnv.git"
    local candidate=""
    local resolved=""

    if [ -z "$requested_ref" ]; then
        printf "main"
        return
    fi

    case "$requested_ref" in
        main|master)
            printf "%s" "$requested_ref"
            return
            ;;
    esac

    if printf "%s" "$requested_ref" | grep -Eq '^[0-9a-f]{7,40}$'; then
        printf "%s" "$requested_ref"
        return
    fi

    for candidate in "$requested_ref" "v${requested_ref#v}"; do
        # Full ref names: a bare pattern also tail-matches branches like testpypi/0.8.0.
        resolved=$(git ls-remote "$repo_url" "refs/tags/$candidate" "refs/heads/$candidate" 2>/dev/null || true)
        if [ -n "$resolved" ]; then
            printf "%s" "$candidate"
            return
        fi
    done

    warn "OpenEnv ref '$requested_ref' not found on GitHub; using 'main' for dependency rewrites."
    printf "main"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            DEPLOY_ALL=true
            shift
            ;;
        --env)
            DEPLOY_ALL=false
            SELECTED_ENVS+=("$2")
            shift 2
            ;;
        --base-sha|--base-image-sha)
            BASE_IMAGE_SHA="$2"
            shift 2
            ;;
        --hf-namespace|--namespace|--hf-user|--hf-username)
            HF_NAMESPACE="$2"
            shift 2
            ;;
        --repo-id|--space-repo)
            SPACE_REPO_OVERRIDE="$2"
            shift 2
            ;;
        --space-suffix|--suffix)
            SPACE_SUFFIX="$2"
            SPACE_SUFFIX_EXPLICIT=true
            shift 2
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --public)
            PRIVATE=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --staging-dir)
            STAGING_DIR="$2"
            shift 2
            ;;
        --hub-tag)
            HUB_TAG="$2"
            shift 2
            ;;
        --openenv-version|--core-version|--openenv-ref)
            OPENENV_VERSION="$2"
            shift 2
            ;;
        --collection-namespace)
            COLLECTION_NAMESPACE="$2"
            shift 2
            ;;
        --collection-slug)
            COLLECTION_SLUG="$2"
            shift 2
            ;;
        --skip-collection)
            SKIP_COLLECTION=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            error "Unknown option: $1"
            ;;
        *)
            # Backward-compatible positional parsing:
            # <env_name> [base_image_sha]
            DEPLOY_ALL=false
            if [ ${#SELECTED_ENVS[@]} -eq 0 ]; then
                SELECTED_ENVS+=("$1")
            elif [ -z "$BASE_IMAGE_SHA" ]; then
                BASE_IMAGE_SHA="$1"
            else
                error "Unexpected positional argument: $1"
            fi
            shift
            ;;
    esac
done

if [ "$SPACE_SUFFIX_EXPLICIT" = false ]; then
    SPACE_SUFFIX="-$(normalized_version_suffix "$OPENENV_VERSION")"
fi

OPENENV_GIT_REF=$(resolve_openenv_git_ref "$OPENENV_VERSION")

if [ -z "$HF_NAMESPACE" ]; then
    # Non-fatal if user is not logged in locally.
    if command -v hf >/dev/null 2>&1; then
        HF_NAMESPACE=$(hf auth whoami 2>/dev/null | head -n 1 | tr -d '\n' || true)
        if printf "%s" "$HF_NAMESPACE" | grep -q '^user='; then
            HF_NAMESPACE=$(printf "%s" "$HF_NAMESPACE" | sed -E 's/^user=([^[:space:]]+).*/\1/')
        fi
    fi
    if [ -z "$HF_NAMESPACE" ]; then
        HF_NAMESPACE="openenv"
    fi
fi

if [ -n "$BASE_IMAGE_SHA" ]; then
    BASE_IMAGE_REF="ghcr.io/huggingface/openenv-base:$BASE_IMAGE_SHA"
else
    BASE_IMAGE_REF="ghcr.io/huggingface/openenv-base:latest"
fi

if ! command -v hf >/dev/null 2>&1; then
    if [ "$DRY_RUN" = true ]; then
        warn "'hf' CLI not found; dry-run will continue without upload."
    else
        error "'hf' CLI is required for deployment. Install with: curl -LsSf https://hf.co/cli/install.sh | bash"
    fi
fi

is_deployable_env() {
    local env_name="$1"
    [ -d "envs/$env_name" ] &&
        { [ -f "envs/$env_name/server/Dockerfile" ] || [ -f "envs/$env_name/Dockerfile" ]; } &&
        [ -f "envs/$env_name/README.md" ]
}

resolve_env_dockerfile() {
    local env_name="$1"
    if [ -f "envs/$env_name/server/Dockerfile" ]; then
        printf "%s" "envs/$env_name/server/Dockerfile"
        return 0
    fi
    if [ -f "envs/$env_name/Dockerfile" ]; then
        printf "%s" "envs/$env_name/Dockerfile"
        return 0
    fi
    return 1
}

discover_all_envs() {
    local env_name=""
    while IFS= read -r env_name; do
        if [ -f "envs/$env_name/SKIP_HF_DEPLOYMENT" ]; then
            SKIPPED_ENVS+=("$env_name")
            warn "Skipping '$env_name' (SKIP_HF_DEPLOYMENT opts out of --all; use --env for explicit deployment)"
        elif is_deployable_env "$env_name"; then
            SELECTED_ENVS+=("$env_name")
        else
            SKIPPED_ENVS+=("$env_name")
            warn "Skipping '$env_name' (missing Dockerfile or README.md)"
        fi
    done < <(
        find envs -mindepth 1 -maxdepth 1 -type d \
            ! -name '.*' \
            ! -name '__*' \
            -exec basename {} \; \
            | sort
    )
}

if [ "$DEPLOY_ALL" = true ]; then
    discover_all_envs
fi

if [ -n "$SPACE_REPO_OVERRIDE" ]; then
    if [ ${#SELECTED_ENVS[@]} -ne 1 ]; then
        error "--repo-id requires exactly one selected environment"
    fi
    if ! printf "%s" "$SPACE_REPO_OVERRIDE" | grep -Eq '^[^/]+/[^/]+$'; then
        error "Invalid --repo-id '$SPACE_REPO_OVERRIDE' (expected owner/repo)"
    fi
fi

if [ ${#SELECTED_ENVS[@]} -eq 0 ]; then
    error "No deployable environments selected."
fi

setup_auth() {
    if [ "$DRY_RUN" = true ]; then
        return
    fi

    if [ -n "${HF_TOKEN:-}" ]; then
        TOKEN_ARGS=(--token "$HF_TOKEN")
        log "Using HF_TOKEN for non-interactive auth."
        return
    fi

    if ! hf auth whoami >/dev/null 2>&1; then
        error "Not authenticated with Hugging Face. Run 'hf auth login' or set HF_TOKEN."
    fi
}

pin_openenv_refs_in_pyproject() {
    local file_path="$1"

    [ -f "$file_path" ] || return 0

    # Pin git-based OpenEnv dependencies and rewrite loose package constraints
    # so release-candidate deployments test the requested OpenEnv ref instead
    # of silently resolving an older PyPI package.
    sed_inplace \
        "/^[[:space:]]*\"/ s|git+https://github.com/huggingface/OpenEnv.git@main|git+https://github.com/huggingface/OpenEnv.git@$OPENENV_GIT_REF|g" \
        "$file_path"
    sed_inplace \
        "/^[[:space:]]*\"/ s|git+https://github.com/huggingface/OpenEnv.git\"|git+https://github.com/huggingface/OpenEnv.git@$OPENENV_GIT_REF\"|g" \
        "$file_path"
    sed_inplace \
        "/^[[:space:]]*\"/ s|\"openenv[<>=~! @][^\"]*\"|\"openenv @ git+https://github.com/huggingface/OpenEnv.git@$OPENENV_GIT_REF\"|g" \
        "$file_path"
}

strip_stage_artifacts() {
    local stage_dir="$1"

    # Remove local/dev artifacts that should never be pushed to Spaces.
    find "$stage_dir" -type d \
        \( -name '.venv' -o -name 'venv' -o -name '.git' -o -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' -o -name 'node_modules' \) \
        -prune -exec rm -rf {} +

    find "$stage_dir" -type f \
        \( -name '.DS_Store' -o -name '*.pyc' -o -name '*.pyo' -o -name 'uv.lock' \) \
        -delete
}

create_environment_dockerfile() {
    local env_name="$1"
    local stage_dir="$2"
    local space_repo="${3:-}"
    local dockerfile_path=""
    local prepare_script="envs/$env_name/server/prepare_hf.sh"
    local tmp_dockerfile="$stage_dir/Dockerfile.tmp"

    dockerfile_path=$(resolve_env_dockerfile "$env_name") || {
        error "Could not find Dockerfile for $env_name"
    }

    cp "$dockerfile_path" "$stage_dir/Dockerfile"

    if [ -f "$prepare_script" ]; then
        chmod +x "$prepare_script"
        "$prepare_script" "$stage_dir/Dockerfile" "$BASE_IMAGE_REF"
    else
        # Handle common base-image patterns.
        sed_inplace "s|^ARG BASE_IMAGE=.*$|ARG BASE_IMAGE=$BASE_IMAGE_REF|g" "$stage_dir/Dockerfile"
        sed_inplace "s|FROM \${BASE_IMAGE}|FROM $BASE_IMAGE_REF|g" "$stage_dir/Dockerfile"
        sed_inplace "s|FROM openenv-base:latest|FROM $BASE_IMAGE_REF|g" "$stage_dir/Dockerfile"
        sed_inplace "s|FROM envtorch-base:latest|FROM $BASE_IMAGE_REF|g" "$stage_dir/Dockerfile"
    fi

    sed_inplace \
        "s|git+https://github.com/huggingface/OpenEnv.git@main|git+https://github.com/huggingface/OpenEnv.git@$OPENENV_GIT_REF|g" \
        "$stage_dir/Dockerfile"
    sed_inplace \
        "s|git+https://github.com/huggingface/OpenEnv.git\"|git+https://github.com/huggingface/OpenEnv.git@$OPENENV_GIT_REF\"|g" \
        "$stage_dir/Dockerfile"
    sed_inplace \
        "s|git+https://github.com/huggingface/OpenEnv.git$|git+https://github.com/huggingface/OpenEnv.git@$OPENENV_GIT_REF|g" \
        "$stage_dir/Dockerfile"
    sed_inplace \
        "s|\"openenv[<>=~! @][^\"]*\"|\"openenv @ git+https://github.com/huggingface/OpenEnv.git@$OPENENV_GIT_REF\"|g" \
        "$stage_dir/Dockerfile"

    # Some base images include older uv versions that fail on a subset of env
    # pyproject layouts. Insert a deterministic uv install before the first sync.
    if grep -q "RUN --mount=type=cache,target=/root/.cache/uv" "$stage_dir/Dockerfile"; then
        awk '
            BEGIN { inserted = 0 }
            {
                if (!inserted && $0 ~ /RUN --mount=type=cache,target=\/root\/\.cache\/uv/) {
                    print "RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \\"
                    print "    install -m 0755 /root/.local/bin/uv /usr/local/bin/uv && \\"
                    print "    install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx"
                    print ""
                    inserted = 1
                }
                print
            }
        ' "$stage_dir/Dockerfile" > "$tmp_dockerfile"
        mv "$tmp_dockerfile" "$stage_dir/Dockerfile"

        # Some environment lockfiles are intentionally loose and may be regenerated
        # between the two uv sync passes. Avoid hard-failing on --frozen in Spaces.
        sed_inplace "s|uv sync --frozen|uv sync|g" "$stage_dir/Dockerfile"
    fi

    # Legacy Dockerfiles that copy src/core often rely on imports from both
    # `core.*` and `openenv.*`. Copy both packages under /app/src, then expose
    # only the shared parent directory to avoid shadowing stdlib modules such
    # as `types` with files under /app/src/core.
    if grep -q "COPY src/core/" "$stage_dir/Dockerfile"; then
        awk '
            /COPY src\/core\// && !inserted {
                print
                print "COPY src/openenv/ /app/src/openenv/"
                inserted = 1
                next
            }
            { print }
        ' "$stage_dir/Dockerfile" > "$tmp_dockerfile"
        mv "$tmp_dockerfile" "$stage_dir/Dockerfile"

        if grep -q '^ENV PYTHONPATH=' "$stage_dir/Dockerfile"; then
            sed_inplace \
                '/^ENV PYTHONPATH=/ { /\/app\/src/! s|^ENV PYTHONPATH=|ENV PYTHONPATH=/app/src:|; }' \
                "$stage_dir/Dockerfile"
        else
            ensure_trailing_newline "$stage_dir/Dockerfile"
            echo "ENV PYTHONPATH=/app/src:\${PYTHONPATH}" >> "$stage_dir/Dockerfile"
        fi
    fi

    if grep -q '^ENV ENABLE_WEB_INTERFACE=' "$stage_dir/Dockerfile"; then
        sed_inplace \
            's/^ENV ENABLE_WEB_INTERFACE=.*/ENV ENABLE_WEB_INTERFACE=true/' \
            "$stage_dir/Dockerfile"
    else
        ensure_trailing_newline "$stage_dir/Dockerfile"
        echo "" >> "$stage_dir/Dockerfile"
        echo "ENV ENABLE_WEB_INTERFACE=true" >> "$stage_dir/Dockerfile"
    fi

    if [ "$env_name" = "textarena_env" ] && [ -n "$space_repo" ]; then
        local repo_name="${space_repo##*/}"
        local textarena_env_id=""
        case "$repo_name" in
            sudoku)
                textarena_env_id="Sudoku-v0"
                ;;
            wordle)
                textarena_env_id="Wordle-v0"
                ;;
        esac

        if [ -n "$textarena_env_id" ]; then
            if grep -q '^ENV TEXTARENA_ENV_ID=' "$stage_dir/Dockerfile"; then
                sed_inplace \
                    "s/^ENV TEXTARENA_ENV_ID=.*/ENV TEXTARENA_ENV_ID=$textarena_env_id/" \
                    "$stage_dir/Dockerfile"
            else
                ensure_trailing_newline "$stage_dir/Dockerfile"
                echo "ENV TEXTARENA_ENV_ID=$textarena_env_id" >> "$stage_dir/Dockerfile"
            fi
        fi
    fi
}

ensure_readme_front_matter_tags() {
    local readme_file="$1"

    [ -f "$readme_file" ] || return 0

    if ! head -n 1 "$readme_file" | grep -q '^---$'; then
        return 0
    fi

    local closing_line
    closing_line=$(grep -n '^---$' "$readme_file" | sed -n '2p' | cut -d: -f1)
    if [ -z "$closing_line" ]; then
        return 0
    fi

    local front_matter
    front_matter=$(sed -n "1,${closing_line}p" "$readme_file")

    local required_tags=()
    required_tags+=("openenv")
    if [ "$HUB_TAG" != "openenv" ]; then
        required_tags+=("$HUB_TAG")
    fi
    required_tags+=("openenv-$OPENENV_VERSION")

    local missing_tags=()
    local required_tag
    for required_tag in "${required_tags[@]}"; do
        if ! printf "%s\n" "$front_matter" | awk -v wanted="$required_tag" '
            BEGIN { found = 0 }
            /^[[:space:]]*-[[:space:]]*/ {
                candidate = $0
                sub(/^[[:space:]]*-[[:space:]]*/, "", candidate)
                gsub(/[[:space:]]+$/, "", candidate)
                if (candidate == wanted) {
                    found = 1
                }
            }
            END { exit found ? 0 : 1 }
        '; then
            missing_tags+=("$required_tag")
        fi
    done

    if [ ${#missing_tags[@]} -eq 0 ]; then
        return 0
    fi

    local missing_csv
    missing_csv=$(IFS=,; echo "${missing_tags[*]}")
    local tmp_file="${readme_file}.tmp"

    awk \
        -v close_line="$closing_line" \
        -v missing_csv="$missing_csv" \
        '
        BEGIN {
            missing_count = split(missing_csv, missing_tags, ",")
            inserted = 0
            saw_tags_key = 0
        }
        {
            if (NR < close_line && $0 ~ /^[[:space:]]*tags:[[:space:]]*$/ && inserted == 0) {
                saw_tags_key = 1
                print $0
                for (i = 1; i <= missing_count; i++) {
                    if (length(missing_tags[i]) > 0) {
                        print "  - " missing_tags[i]
                    }
                }
                inserted = 1
                next
            }

            if (NR == close_line && saw_tags_key == 0 && inserted == 0) {
                print "tags:"
                for (i = 1; i <= missing_count; i++) {
                    if (length(missing_tags[i]) > 0) {
                        print "  - " missing_tags[i]
                    }
                }
                inserted = 1
            }

            print $0
        }
        ' "$readme_file" > "$tmp_file"

    mv "$tmp_file" "$readme_file"
}

create_readme() {
    local env_name="$1"
    local stage_dir="$2"
    local space_repo="$3"
    local readme_source="envs/$env_name/README.md"
    local output_readme="$stage_dir/README.md"
    # Client class: the first class in client.py that subclasses EnvClient or MCPToolClient.
    local env_class
    env_class=$(awk '/^class /{name=$2; sub(/[(:].*/,"",name); hdr=1} hdr && /(EnvClient|MCPToolClient)/{print name; exit} hdr && /:[[:space:]]*$/{hdr=0}' "envs/$env_name/client.py" 2>/dev/null)
    # Space host: https://<owner>-<name>.hf.space, lowercased, with "/", "_" and "." replaced by "-".
    local space_url
    space_url="https://$(printf "%s" "$space_repo" | tr '[:upper:]' '[:lower:]' | tr '/_.' '---').hf.space"

    if head -n 1 "$readme_source" | grep -q '^---$'; then
        # The env's own text comes first: the web interface reads its first paragraph.
        cat "$readme_source" > "$output_readme"
        cat >> "$output_readme" << README_EOF

## Hugging Face Space Deployment

This Space is built from OpenEnv environment \`$env_name\`.

- Space URL: \`https://huggingface.co/spaces/$space_repo\`
- OpenEnv pinned ref: \`$OPENENV_VERSION\`
- Hub tag: \`$HUB_TAG\`
README_EOF
        # The snippet needs anonymous access, so private Spaces skip it.
        if [ -n "$env_class" ] && [ "$PRIVATE" = false ]; then
            cat >> "$output_readme" << README_EOF

### Connecting from Code

\`\`\`bash
pip install git+https://huggingface.co/spaces/$space_repo
\`\`\`

\`\`\`python
from $env_name import $env_class

with $env_class(base_url="$space_url").sync() as env:
    result = env.reset()
\`\`\`
README_EOF
        fi
    else
        cat > "$output_readme" << README_EOF
---
title: ${env_name} Environment
sdk: docker
app_port: 8000
base_path: /web
tags:
  - $HUB_TAG
  - openenv-$OPENENV_VERSION
---

# ${env_name} Environment

Space URL: \`https://huggingface.co/spaces/$space_repo\`

OpenEnv pinned ref: \`$OPENENV_VERSION\`

README_EOF
        cat "$readme_source" >> "$output_readme"
    fi

    # Ensure metadata color values pass Hub validation regardless of source README defaults.
    if grep -q '^[[:space:]]*colorFrom:' "$output_readme"; then
        sed_inplace "s|^[[:space:]]*colorFrom:.*$|colorFrom: blue|g" "$output_readme"
    fi
    if grep -q '^[[:space:]]*colorTo:' "$output_readme"; then
        sed_inplace "s|^[[:space:]]*colorTo:.*$|colorTo: green|g" "$output_readme"
    fi

    # Global collection sync relies on this tag, so enforce it for every
    # generated README even when source front matter omits tags.
    ensure_readme_front_matter_tags "$output_readme"
}

prepare_stage() {
    local env_name="$1"
    local stage_dir="$2"
    local space_repo="$3"

    rm -rf "$stage_dir"
    mkdir -p "$stage_dir/envs"

    # Include full src for Dockerfiles that reference src/ directly.
    cp -R src "$stage_dir/"
    # Back-compat for legacy Dockerfiles that still copy src/core.
    # We provide both:
    #  - core/* imports at /app/src/core/*
    #  - openenv/* imports at /app/src/core/openenv/*
    if [ -d "$stage_dir/src/openenv" ]; then
        mkdir -p "$stage_dir/src/core"
        if [ -d "$stage_dir/src/openenv/core" ]; then
            cp -R "$stage_dir/src/openenv/core/." "$stage_dir/src/core/"
        fi
        mkdir -p "$stage_dir/src/core/openenv"
        cp -R "$stage_dir/src/openenv/." "$stage_dir/src/core/openenv/"
    fi
    cp -R "envs/$env_name" "$stage_dir/envs/"

    # Include root metadata used by some Dockerfiles.
    [ -f pyproject.toml ] && cp pyproject.toml "$stage_dir/"
    # Do not copy root uv.lock by default. For env-scoped projects, a mismatched
    # lock file can make `uv sync --frozen` fail inside Docker builds.

    # Also copy env content to stage root for Dockerfiles that use "COPY . /app/env".
    cp -R "envs/$env_name/." "$stage_dir/"

    strip_stage_artifacts "$stage_dir"

    # Pin OpenEnv refs in both possible pyproject locations.
    pin_openenv_refs_in_pyproject "$stage_dir/pyproject.toml"
    pin_openenv_refs_in_pyproject "$stage_dir/envs/$env_name/pyproject.toml"

    create_environment_dockerfile "$env_name" "$stage_dir" "$space_repo"
    create_readme "$env_name" "$stage_dir" "$space_repo"
}

resolve_space_repo() {
    local env_name="$1"
    if [ -n "$SPACE_REPO_OVERRIDE" ]; then
        printf "%s" "$SPACE_REPO_OVERRIDE"
    else
        printf "%s/%s%s" "$HF_NAMESPACE" "$env_name" "$SPACE_SUFFIX"
    fi
}

deploy_env() {
    local env_name="$1"
    local space_repo
    space_repo=$(resolve_space_repo "$env_name")
    local stage_dir="$STAGING_DIR/$space_repo"

    if ! is_deployable_env "$env_name"; then
        warn "Skipping '$env_name' (missing Dockerfile or README.md)"
        SKIPPED_ENVS+=("$env_name")
        return 0
    fi

    log "Preparing $env_name (OpenEnv ref: $OPENENV_VERSION, base image: $BASE_IMAGE_REF)"
    prepare_stage "$env_name" "$stage_dir" "$space_repo" || return 1

    if [ "$DRY_RUN" = true ]; then
        log "[dry-run] Would create/update space: $space_repo"
        log "[dry-run] Would upload folder: $stage_dir"
        if [ "$PRIVATE" = true ]; then
            log "[dry-run] Would set space visibility to private: $space_repo"
        else
            log "[dry-run] Would set space visibility to public: $space_repo"
        fi
        DEPLOYED_SPACES+=("$space_repo")
        return 0
    fi

    local repo_create_args=(--repo-type space --space-sdk docker --exist-ok)
    local upload_args=(--repo-type=space)

    if [ "$PRIVATE" = true ]; then
        repo_create_args+=(--private)
    fi
    if [ ${#TOKEN_ARGS[@]} -gt 0 ]; then
        repo_create_args+=("${TOKEN_ARGS[@]}")
        upload_args+=("${TOKEN_ARGS[@]}")
    fi

    hf repo create "$space_repo" "${repo_create_args[@]}" || return 1

    hf upload \
        "${upload_args[@]}" \
        "$space_repo" \
        "$stage_dir" \
        . || return 1

    # Ensure visibility is applied even when repo already existed.
    local privacy_flag="--private"
    if [ "$PRIVATE" = false ]; then
        privacy_flag="--no-private"
    fi
    if [ ${#TOKEN_ARGS[@]} -gt 0 ]; then
        hf repo settings --repo-type space "$privacy_flag" "${TOKEN_ARGS[@]}" "$space_repo" || return 1
    else
        hf repo settings --repo-type space "$privacy_flag" "$space_repo" || return 1
    fi

    DEPLOYED_SPACES+=("$space_repo")
    log "Uploaded https://huggingface.co/spaces/$space_repo"
}

update_collection() {
    if [ "$SKIP_COLLECTION" = true ]; then
        log "Skipping collection update (--skip-collection)."
        return 0
    fi

    if [ ${#DEPLOYED_SPACES[@]} -eq 0 ]; then
        log "No deployed spaces available for collection update."
        return 0
    fi

    local python_bin=""
    local candidate=""
    for candidate in "$REPO_ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
        if [ -z "$candidate" ]; then
            continue
        fi
        if [ ! -x "$candidate" ]; then
            continue
        fi
        if "$candidate" -c "import huggingface_hub" >/dev/null 2>&1; then
            python_bin="$candidate"
            break
        fi
    done

    if [ -z "$python_bin" ]; then
        warn "No Python interpreter with 'huggingface_hub' found; cannot update collection automatically."
        return 0
    fi

    local cmd=(
        "$python_bin"
        scripts/manage_hf_collection.py
        --version "$OPENENV_VERSION"
        --collection-namespace "$COLLECTION_NAMESPACE"
        --skip-global-collection
    )
    if [ -n "$COLLECTION_SLUG" ]; then
        cmd+=(--collection-slug "$COLLECTION_SLUG")
    fi
    if [ "$DRY_RUN" = true ]; then
        cmd+=(--dry-run)
    fi
    for space_id in "${DEPLOYED_SPACES[@]}"; do
        cmd+=(--space-id "$space_id")
    done

    log "Updating collection for version $OPENENV_VERSION"
    "${cmd[@]}"
}

log "Namespace: $HF_NAMESPACE"
log "Space suffix: $SPACE_SUFFIX"
log "OpenEnv pinned ref: $OPENENV_VERSION"
log "OpenEnv git ref for dependency rewrites: $OPENENV_GIT_REF"
log "Base image ref: $BASE_IMAGE_REF"
log "Selected env count: ${#SELECTED_ENVS[@]}"

setup_auth

for env_name in "${SELECTED_ENVS[@]}"; do
    if deploy_env "$env_name"; then
        :
    else
        warn "Deployment failed for '$env_name'"
        FAILED_ENVS+=("$env_name")
    fi
done

update_collection

log "Deployment summary:"
log "  deployed: ${#DEPLOYED_SPACES[@]}"
log "  skipped:  ${#SKIPPED_ENVS[@]}"
log "  failed:   ${#FAILED_ENVS[@]}"

if [ ${#SKIPPED_ENVS[@]} -gt 0 ]; then
    log "Skipped envs: ${SKIPPED_ENVS[*]}"
fi
if [ ${#FAILED_ENVS[@]} -gt 0 ]; then
    log "Failed envs: ${FAILED_ENVS[*]}"
    exit 1
fi
