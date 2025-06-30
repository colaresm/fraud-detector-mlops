FROM python:3.11 AS builder

RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/
WORKDIR /app

COPY api/requirements.txt .

RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

COPY main.py .
COPY .env .env
COPY api /app/api

EXPOSE 3000

CMD ["python", "main.py"]
