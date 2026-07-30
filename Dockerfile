FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY *.pdf *.txt ./data/
RUN mkdir -p /app/storage/followups

# Build the one-time local index at image build time. Re-run the command after changing data.
RUN python scripts/ingest.py --source /app/data --output /app/storage/index.json --chroma-path /app/storage/chroma --download-model

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
