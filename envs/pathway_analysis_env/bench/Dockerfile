# Pathway Analysis Env - Benchmark image
#
# Purpose: provide a reproducible "gold-standard" RNA-seq toolchain for benchmarks
# (STAR/featureCounts or Salmon), without bloating the normal serving image.
#
# Build:
#   docker build -t openenv-pathway-bench:latest -f envs/pathway_analysis_env/bench/Dockerfile .
#
# Run (example):
#   docker run --rm -it -v "$PWD":/work openenv-pathway-bench:latest bash

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

WORKDIR /work

# System deps for downloading, building indexes, and basic tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    bzip2 \
    gzip \
    pigz \
    unzip \
    xz-utils \
    zstd \
    git \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Bioconda ``salmon`` links against system libstdc++ and needs a real generated locale.
RUN sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install micromamba (for bioconda packages).
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN curl -LsSf https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && install -m 0755 bin/micromamba /usr/local/bin/micromamba \
    && rm -rf bin

# Create a dedicated conda env with RNA-seq tools.
# Note: versions are intentionally not pinned here; for long-term reproducibility,
# pin in an environment.yml and lock if needed.
RUN micromamba create -y -n rnaseq -c conda-forge -c bioconda \
    python=3.11 \
    pandas \
    numpy \
    sra-tools \
    fastqc \
    multiqc \
    samtools \
    "subread>=2.0.2" \
    star \
    salmon \
    && micromamba clean -a -y

ENV PATH="/opt/micromamba/envs/rnaseq/bin:${PATH}"

# Make matplotlib cache writable inside container (gseapy pulls matplotlib indirectly).
ENV MPLCONFIGDIR=/work/envs/pathway_analysis_env/outputs/.mplcache

# Default to bash for interactive benchmarking.
CMD ["bash"]

