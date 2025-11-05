# сборка
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Загрузка модели (только при сборке)
ENV HF_HOME=/root/.cache/huggingface
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
m = 'jinaai/jina-embeddings-v3'; \
AutoTokenizer.from_pretrained(m, trust_remote_code=True); \
AutoModel.from_pretrained(m, trust_remote_code=True); \
print('Модель закэширована')"

# финальный образ
FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libpq5 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем ВСЁ окружение из builder
COPY --from=builder /usr/local /usr/local

# Копируем кэш модели
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Копируем код
COPY embed.py pydantic_models.py app.py ./

ENV HF_HOME=/root/.cache/huggingface
ENV LOG_LEVEL=INFO

EXPOSE 8000

# Порт и запуск(можно с несколькими воркерами, если ресурсы позволяют)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]