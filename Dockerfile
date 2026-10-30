FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

COPY api ./api
COPY contracts ./contracts
COPY models ./models

COPY data/processed/incidents_features.parquet ./data/processed/
COPY data/processed/sales_orders_features.parquet ./data/processed/
COPY data/processed/sales_pickup_features.parquet ./data/processed/

RUN mkdir -p /app/data/uploads /app/data/app/analysis /app/data/app/models \
    && chmod -R 777 /app/data /app

WORKDIR /app/api

CMD ["sh", "-c", "uvicorn opsinsight.main:app --host 0.0.0.0 --port ${X_ZOHO_CATALYST_LISTEN_PORT:-9000}"]
