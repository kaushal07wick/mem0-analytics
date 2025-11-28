# =========
# 1️⃣ Base Image
# =========
FROM python:3.12-slim AS base

# Disable Python’s buffering and bytecode caching
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# =========
# 2️⃣ System Dependencies
# =========
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =========
# 3️⃣ Working Directory
# =========
WORKDIR /app

# =========
# 4️⃣ Copy Files
# =========
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src ./src
COPY pyproject.toml setup.cfg README.md LICENSE ./

# =========
# 5️⃣ Environment Variables
# =========
# These can also be overridden via docker run -e VAR=value
ENV POSTHOG_URL=https://app.posthog.com \
    DATA_DIR=/data \
    PYTHONPATH=/app/src

# =========
# 6️⃣ Runtime Directory
# =========
RUN mkdir -p /data

# =========
# 7️⃣ Command Options
# =========
# You can change this to mem0-analytics-track if you prefer
ENTRYPOINT ["python", "-m", "analytics.daemon"]
CMD []
