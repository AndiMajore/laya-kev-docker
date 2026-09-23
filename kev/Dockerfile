# syntax=docker/dockerfile:1
# kev (github.com/jaredpalmer/kev) on an NVIDIA GPU: the TypeSafe /v1/systemone server with
# a Qwen3.5-based checkpoint. A separate image from laya's because kev pins torch<2.9.
# Build context is the repository root (see compose.yaml), for docker/entrypoint.py.
FROM python:3.12-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch 2.8 is what kev's uv.lock pins; the CUDA build comes from the PyTorch index.
ARG TORCH_INDEX=cu128
ARG KEV_TORCH_VERSION=2.8.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${KEV_TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

# kev is not on PyPI (the `kev` name there is another project), so install a pinned
# commit from GitHub. The archive needs no git in the image.
ARG KEV_REF=557598fced1dada75dfbf36ed144dce309ac6ceb
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "kev[serve] @ https://github.com/jaredpalmer/kev/archive/${KEV_REF}.tar.gz"

# Gated DeltaNet kernels for the Qwen3.5 hybrid backbones; without them transformers falls
# back to slow reference code (kev's README and Modal image do the same).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-linear-attention && pip check

FROM python:3.12-slim-bookworm AS runtime

ARG KEV_REF=557598fced1dada75dfbf36ed144dce309ac6ceb
ARG TORCH_INDEX=cu128
LABEL org.opencontainers.image.title="kev-gpu" \
      org.opencontainers.image.description="kev TypeSafe server with CUDA (${TORCH_INDEX})" \
      org.opencontainers.image.revision="${KEV_REF}"

ENV PATH="/opt/venv/bin:$PATH" \
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
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home laya \
    && mkdir -p /data/hf \
    && chown -R laya:laya /data/hf

COPY --from=build /opt/venv /opt/venv
COPY docker/entrypoint.py /opt/laya/entrypoint.py
COPY kev/kev_serve.py /opt/kev/kev_serve.py

USER laya
WORKDIR /home/laya
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('KEV_PORT', '8001'), timeout=4)"

ENTRYPOINT ["python", "/opt/laya/entrypoint.py"]
CMD ["python", "/opt/kev/kev_serve.py"]
