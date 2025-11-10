import logging
import os
from fastapi import FastAPI, HTTPException
from embed import JinaEmbedder
from pydantic_models import EmbedRequest, EmbedResponsePassage, EmbedResponseQuery
import json
import asyncpg
from contextlib import asynccontextmanager
from config import Config

conf = Config()

# Логгирование
logging.basicConfig(
    level=getattr(logging, conf.LOG_LEVEL),
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
            host=conf.DB_HOST,
            port=int(conf.DB_PORT),
            database=conf.DB_NAME,
            user=conf.DB_USER,
            password=conf.DB_PASSWORD,
            min_size=conf.DB_POOL_MIN_SIZE,
            max_size=conf.DB_POOL_MAX_SIZE
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

@app.post("/embed", summary="Генерация эмбеддингов")
async def embed_chunks(req: EmbedRequest) -> EmbedResponsePassage | EmbedResponseQuery:
    if not embedder:
        raise HTTPException(status_code=503, detail="Модель не загружена")

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
    
    if req.task == "retrieval.query":
        return EmbedResponseQuery(
            embedding=embeddings.tolist(),
            count=len(embeddings),
            batch_size=len(req.texts),
            embedding_dimensions=embeddings.shape[1]
        )

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