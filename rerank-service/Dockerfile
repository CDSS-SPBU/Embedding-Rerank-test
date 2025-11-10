# Сборка
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Загрузка модели CrossEncoder
ENV HF_HOME=/root/.cache/huggingface
RUN python -c "\
from sentence_transformers import CrossEncoder; \
m = 'cross-encoder/ms-marco-MiniLM-L-12-v2'; \
CrossEncoder(m); \
print('CrossEncoder модель закэширована')"

# Финальный образ
FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем Python пакеты из builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Добавляем .local/bin в PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV HF_HOME=/root/.cache/huggingface
ENV LOG_LEVEL=INFO

# Копируем код
COPY *.py ./

EXPOSE 8001

# Здоровье при запуске
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]