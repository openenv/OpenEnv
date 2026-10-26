#!/bin/bash
# Custom HF deployment script for openspiel_env
# OpenSpiel uses a different base image with C++ compilation

set -e

DOCKERFILE_PATH="$1"
BASE_IMAGE_REF="$2"

echo "OpenSpiel: Using custom Dockerfile preparation"

# Cross-platform sed in-place editing
sed_inplace() {
    if sed --version >/dev/null 2>&1; then
        # GNU sed (Linux)
        sed -i "$@"
    else
        # BSD sed (macOS)
        sed -i '' "$@"
    fi
}

# Replace ARG with hardcoded FROM using the special OpenSpiel base
sed_inplace 's|ARG OPENSPIEL_BASE_IMAGE=.*|FROM ghcr.io/huggingface/openenv-openspiel-base:latest|g' "$DOCKERFILE_PATH"
sed_inplace '/^FROM \${OPENSPIEL_BASE_IMAGE}/d' "$DOCKERFILE_PATH"

echo "OpenSpiel: Modified Dockerfile to use GHCR OpenSpiel base image"
echo "OpenSpiel builds can take 10-15 minutes due to C++ compilation"
