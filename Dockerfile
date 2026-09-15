FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r /tmp/requirements-docker.txt && \
    rm /tmp/requirements-docker.txt

COPY main.py worker.py tasks.py README.md launch.sh ./
COPY configs ./configs
COPY repo ./repo

RUN mkdir -p /app/outputs /app/temp
