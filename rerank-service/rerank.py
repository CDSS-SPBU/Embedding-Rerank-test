from sentence_transformers import CrossEncoder
import logging
import torch
import numpy as np
from config import Config
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException

conf = Config()

logging.basicConfig(
    level=getattr(logging, conf.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RerankModel:
    """Переранжирование пар текстов на основе кросс-энкодера"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", max_workers: int = conf.MAX_WORKERS):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            self.model = CrossEncoder(model_name, max_length=512, device=self.device)

            logger.info("Используется %s, max_workers=%d", self.device, max_workers)
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели: {e}")
        
    async def rerank_async(self, query: str, passages: list[str], timeout: float = conf.RERANK_TIMEOUT) -> list[tuple[str, float]]:
        """Асинхронное переранжирование с timeout"""
        logger.info("Переранжирование %d пар текстов", len(passages))
        
        loop = asyncio.get_event_loop()
        
        try:
            # Запускаем в thread pool чтобы не блокировать event loop
            scores = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool, 
                    lambda: self.model.predict([[query, passage] for passage in passages])
                ),
                timeout=timeout
            )
            
            # Нормализация scores
            normalized_scores = self._normalize_scores(scores)
            
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            logger.info("Переранжирование завершено")
            
            return [
                (passages[i], float(normalized_scores[i]))
                for i in sorted_indices
                if float(normalized_scores[i]) != 0
            ]
            
        except asyncio.TimeoutError:
            logger.error("Timeout при переранжировании %d пар за %.1f сек", len(passages), timeout)
            raise HTTPException(status_code=504, detail="Timeout при переранжировании")
        except Exception as e:
            logger.error("Ошибка при переранжировании: %s", e)
            raise

    def _normalize_scores(self, scores):
        """Быстрая синхронная нормализация"""
        p25 = np.percentile(scores, 25)
        p75 = np.percentile(scores, 75)

        if p75 > p25:
            normalized_scores = (scores - p25) / (p75 - p25)
            normalized_scores = np.clip(normalized_scores, 0, 1)
        else:
            normalized_scores = np.ones_like(scores) * 0.5
            
        return normalized_scores


    def rerank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Переранжировать пары текстов на основе кросс-энкодера"""
        logger.info("Переранжирование %d пар текстов", len(passages))
        try:
            scores = self.model.predict([[query, passage] for passage in passages])

            # Используем перцентили для более robust нормализации
            p25 = np.percentile(scores, 25)  # 25-й перцентиль как "плохой" уровень
            p75 = np.percentile(scores, 75)  # 75-й перцентиль как "хороший" уровень

            if p75 > p25:
                normalized_scores = (scores - p25) / (p75 - p25)
                normalized_scores = np.clip(normalized_scores, 0, 1)
            else:
                # Если все оценки одинаковые
                normalized_scores = np.ones_like(scores) * 0.5

            sorted_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            logger.info("Переранжирование завершено")
            return [
                (passages[i], float(normalized_scores[i]))
                for i in sorted_indices
                if float(normalized_scores[i]) != 0
            ]
        except Exception as e:
            logger.error("Ошибка при переранжировании: %s", e)
            raise
