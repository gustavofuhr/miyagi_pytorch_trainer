FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    # Make wandb work smoothly in containers
    WANDB_DIR=/workspace/wandb \
    WANDB_CACHE_DIR=/workspace/.cache/wandb

# git is intentionally excluded — its presence causes wandb to attempt git
# metadata collection which fails inside a container (no .git directory).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv python3-dev \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /workspace

# Install dependencies first (separate layer for caching)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

# Copy trainer source code
COPY . .

CMD ["python", "miyagi_trainer/train.py", "--help"]
