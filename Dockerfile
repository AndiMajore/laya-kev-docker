# syntax=docker/dockerfile:1
# laya-serve on an NVIDIA GPU. CUDA comes from the PyTorch wheels (they bundle the CUDA
# runtime, cuDNN and NCCL), so the base stays slim and only the host driver is needed.
FROM python:3.12-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch first, from the CUDA index, so installing laya keeps this build instead of
# pulling the default wheel from PyPI.
ARG TORCH_INDEX=cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

ARG LAYA_VERSION=0.3.9
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "laya[serve]==${LAYA_VERSION}" && pip check

FROM python:3.12-slim-bookworm AS runtime

ARG LAYA_VERSION=0.3.9
ARG TORCH_INDEX=cu128
LABEL org.opencontainers.image.title="laya-gpu" \
      org.opencontainers.image.description="laya-serve with CUDA (${TORCH_INDEX})" \
      org.opencontainers.image.version="${LAYA_VERSION}"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/data/hf \
    LAYA_DEVICE=cuda \
    LAYA_PORT=8000 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home laya \
    && mkdir -p /data/hf \
    && chown -R laya:laya /data/hf

COPY --from=build /opt/venv /opt/venv
COPY docker/entrypoint.py /opt/laya/entrypoint.py

USER laya
WORKDIR /home/laya
EXPOSE 8000

# Preloading downloads and builds the checkpoints on first boot, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('LAYA_PORT', '8000'), timeout=4)"

# The entrypoint moves *_FILE secrets (HF_TOKEN, LAYA_API_KEY) into the environment.
ENTRYPOINT ["python", "/opt/laya/entrypoint.py"]
CMD ["laya-serve"]
