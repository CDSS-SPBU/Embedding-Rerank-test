import logging
import os
from fastapi import FastAPI, HTTPException
from embed import JinaEmbedder
from pydantic_models import EmbedRequest
import json
import asyncpg
from contextlib import asynccontextmanager


# Логгирование
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedding-service")

# Глобальные переменные
embedder = None
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedder, db_pool
    
    logger.info("Запуск приложения...")
    
    # Инициализация модели
    try:
        embedder = JinaEmbedder(max_workers=3)
        logger.info("Модель загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise
    
    # Инициализация БД
    try:
        db_pool = await asyncpg.create_pool(
            host=os.getenv("PG_HOST", "postgres"),
            port=int(os.getenv("PG_PORT", 5432)),
            database=os.getenv("PG_DB", "rag"),
            user=os.getenv("PG_USER", "embedd_user"),
            password=os.getenv("PG_PASSWORD", "embedd_password"),
            min_size=1,
            max_size=10
        )
        logger.info("Пул БД инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации БД: {e}")
        raise
    
    yield  # Здесь приложение работает
    
    # Shutdown
    logger.info("Завершение приложения...")
    
    if db_pool:
        await db_pool.close()
        logger.info("Пул БД закрыт")
    
    logger.info("Приложение завершено")

app = FastAPI(title="Jina Embedding Service", lifespan=lifespan)




@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": embedder is not None,
        "db_connected": db_pool is not None
    }

@app.post("/embed")
async def embed_chunks(req: EmbedRequest):
    if not embedder:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    if len(req.texts) == 0:
        raise HTTPException(status_code=400, detail="texts пустой")
    
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Слишком большой батч. Максимум 100 текстов")

    try:
        logger.info(f"Обработка батча из {len(req.texts)} текстов")
        
        # Асинхронная генерация эмбеддингов
        embeddings = await embedder.encode_async(
            texts=req.texts,
            task=req.task,
            dimensions=req.dimensions
        )
    except Exception as e:
        logger.exception("Ошибка при генерации эмбеддингов")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        # Подготовка данных для БД
        meta_list = req.metadata or [{} for _ in req.texts]
        
        # Преобразуем embeddings в строковое представление для pgvector
        texts = req.texts
        metadatas = [json.dumps(meta, ensure_ascii=False) for meta in meta_list]
        embedding_strings = [
            '[' + ','.join(map(str, emb.tolist())) + ']' 
            for emb in embeddings
        ]

        async with db_pool.acquire() as connection:
            query = """
                INSERT INTO chunks (content, metadata, embedding) 
                SELECT content, metadata::jsonb, embedding::vector
                FROM unnest($1::text[], $2::text[], $3::text[]) 
                AS t(content, metadata, embedding)
                RETURNING id
            """
            
            rows = await connection.fetch(query, texts, metadatas, embedding_strings)
            ids = [row['id'] for row in rows]

        logger.info(f"Вставлено {len(ids)} чанков")
        return {
            "inserted_ids": ids, 
            "count": len(ids),
            "batch_size": len(req.texts),
            "embedding_dimensions": embeddings.shape[1]
        }

    except Exception as e:
        logger.exception("Ошибка в /embed")
        raise HTTPException(status_code=500, detail=str(e))