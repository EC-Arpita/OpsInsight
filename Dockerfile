# OpsInsight API as a container (Zoho Catalyst AppSail custom runtime, or any Docker host)
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY contracts ./contracts
COPY models ./models

WORKDIR /app/api
# Catalyst passes the port in X_ZOHO_CATALYST_LISTEN_PORT; elsewhere it falls back to 9000
CMD ["sh", "-c", "uvicorn opsinsight.main:app --host 0.0.0.0 --port ${X_ZOHO_CATALYST_LISTEN_PORT:-9000}"]
