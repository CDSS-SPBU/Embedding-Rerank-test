from rerank import RerankModel
# import json
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException
from pydantic_models import RerankRequest, RerankResponse, RerankedItem
from config import Config
import asyncio

conf = Config()

# Логгирование
logging.basicConfig(
    level=getattr(logging, conf.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rerank-service")

# Глобальные переменные
reranker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global reranker
    
    logger.info("Запуск приложения...")
    
    # Инициализация модели
    try:
        reranker = RerankModel()
        logger.info("Модель загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise
    
    
    yield
    
    if reranker:
        reranker.thread_pool.shutdown(wait=True)
    logger.info("Приложение завершено")

app = FastAPI(title="Rerank Service", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": reranker is not None
    }

@app.post("/rerank", response_model=RerankResponse, summary="Переранжирование текстов")
async def rerank_chunks(req: RerankRequest):
    if not reranker:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        reranked_chunks = await reranker.rerank_async(
            query=req.query, 
            passages=req.passages
        )
        
        results = []
        for rank, (text, score) in enumerate(reranked_chunks, 1):
            results.append(RerankedItem(
                text=text,
                score=score,
                rank=rank
            ))
        
        return RerankResponse(
            reranked_results=results,
            original_query=req.query,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Превышено время ожидания обработки")
    except Exception as e:
        logger.exception("Ошибка при переранжировании")
        raise HTTPException(status_code=500, detail=str(e))
