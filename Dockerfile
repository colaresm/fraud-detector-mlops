FROM python:3.11-alpine AS builder

RUN apk add --no-cache build-base

WORKDIR /app

COPY api/requirements.txt .

RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-alpine

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

COPY api/ .

EXPOSE 3000

CMD ["python", "main.py"]
