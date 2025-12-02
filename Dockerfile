FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY pyproject.toml setup.cfg README.md LICENSE ./

ENV POSTHOG_URL=https://app.posthog.com \
    DATA_DIR=/data \
    PYTHONPATH=/app/src

RUN mkdir -p /data

ENTRYPOINT ["python", "-m", "mem0_analytics.daemon"]
CMD []
