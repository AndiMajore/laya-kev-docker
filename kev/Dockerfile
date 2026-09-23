# syntax=docker/dockerfile:1
# kev (github.com/jaredpalmer/kev) on an NVIDIA GPU: the TypeSafe /v1/systemone server with
# a Qwen3.5-based checkpoint. A separate image from laya's because kev pins torch<2.9.
# Build context is the repository root (see compose.kev.yaml), for docker/entrypoint.py.
# Single stage on purpose: the environment is ~7 GB of wheels and a build stage would
# only copy it, doubling the disk a builder (e.g. a CI runner) needs.
FROM python:3.12-slim-bookworm

ARG KEV_REF=557598fced1dada75dfbf36ed144dce309ac6ceb
ARG TORCH_INDEX=cu128
LABEL org.opencontainers.image.title="kev-gpu" \
      org.opencontainers.image.description="kev TypeSafe server with CUDA (${TORCH_INDEX})" \
      org.opencontainers.image.revision="${KEV_REF}"

ENV PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/data/hf \
    TRITON_CACHE_DIR=/data/hf/triton \
    KEV_PORT=8001 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Triton (used by flash-linear-attention) compiles its CUDA launcher with a C compiler
# the first time a kernel runs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home laya \
    && mkdir -p /data/hf \
    && chown -R laya:laya /data/hf \
    && python -m venv /opt/venv

# torch 2.8 is what kev's uv.lock pins; the CUDA build comes from the PyTorch index.
ARG KEV_TORCH_VERSION=2.8.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${KEV_TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

# kev is not on PyPI (the `kev` name there is another project), so install a pinned
# commit from GitHub. The archive needs no git in the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "kev[serve] @ https://github.com/jaredpalmer/kev/archive/${KEV_REF}.tar.gz"

# Gated DeltaNet kernels for the Qwen3.5 hybrid backbones; without them transformers falls
# back to slow reference code (kev's README and Modal image do the same).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-linear-attention

# causal_conv1d: the short-convolution kernel of the Gated DeltaNet layers; without it
# transformers warns and uses its reference PyTorch implementation. PyPI only has the
# source (building it needs nvcc), so install the matching prebuilt wheel from the
# upstream release: CUDA 12, this torch minor, cxx11 ABI, CPython 3.12, x86_64.
ARG CAUSAL_CONV1D_VERSION=1.7.0
RUN --mount=type=cache,target=/root/.cache/pip \
    v="${CAUSAL_CONV1D_VERSION}"; t="${KEV_TORCH_VERSION%.*}"; \
    pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${v}/causal_conv1d-${v}%2Bcu12torch${t}cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    && pip check

COPY docker/entrypoint.py /opt/laya/entrypoint.py
COPY kev/kev_serve.py /opt/kev/kev_serve.py

USER laya
WORKDIR /home/laya
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('KEV_PORT', '8001'), timeout=4)"

ENTRYPOINT ["python", "/opt/laya/entrypoint.py"]
CMD ["python", "/opt/kev/kev_serve.py"]
