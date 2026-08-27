# CUDA 12.8 runtime — first CUDA version with native Blackwell (sm_120) support,
# required for RTX 50-series GPUs. cuDNN runtime included for torch.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1

# System deps: Python 3.11 + build tooling + curl (for uv bootstrap)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        python3-pip curl ca-certificates git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# uv for fast dependency resolution/install
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# ---- Dependency layer (cached separately from source) ----
# Install PyTorch FIRST from the cu128 index (Blackwell/sm_120 support), pinning
# to a stable 2.7.x release. This avoids the nightly `weights_only` model-load
# bug seen with dev torch builds.
RUN uv pip install --system \
        torch==2.7.* \
        --index-url https://download.pytorch.org/whl/cu128

# Copy the manifest and install the remaining Python deps (darts, fastapi, ...).
# torch is already satisfied so uv won't replace it.
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Quick sanity check that the GPU is usable from torch at build time. This will
# only exercise the driver when --gpus is passed at build; otherwise it just
# confirms torch imports and reports the (absent) device.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# ---- Application layer ----
COPY . .

# Default command: serve the FastAPI API. Override via compose for training/UI.
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
