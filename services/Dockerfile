# Single image for both the API and the ingestion worker; they differ only by
# the command they run. Keeps the build cache warm and the images identical.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# tesseract-ocr is the OCR engine for the scanned-document path. It is a real,
# offline engine: no credentials, and no network call at run time. That second
# property matters beyond convenience -- it is what keeps the air-gapped
# deployment story intact. It is only invoked when OCR_PROVIDER=tesseract.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services /app/services
COPY database /app/database
COPY scripts /app/scripts
COPY web /app/web
COPY eval /app/eval

# Run as a non-root user. `data/` is bind-mounted from the host, so the blob root
# is created and chowned here rather than relying on the mount's ownership.
RUN useradd --create-home --uid 10001 brain \
 && mkdir -p /app/data/blobs \
 && chown -R brain:brain /app
USER brain

EXPOSE 8000
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
