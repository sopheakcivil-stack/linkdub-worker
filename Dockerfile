FROM python:3.11.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/linkdub/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 linkdub

WORKDIR /app
COPY requirements.txt ./
RUN pip install --requirement requirements.txt
COPY linkdub ./linkdub
COPY run_worker.py ./

USER linkdub
ENTRYPOINT ["python", "run_worker.py"]

