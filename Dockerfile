# MALAR engine + control plane
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY malar ./malar
# Install core + ml extras (CPU wheels by default; GPU comes from the host runtime)
RUN pip install --upgrade pip && pip install ".[dev]"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "malar.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
