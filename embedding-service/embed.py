import logging
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from config import Config

conf = Config()

logging.basicConfig(
    level=getattr(logging, conf.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JinaEmbedder:
    """Генерация эмбеддингов через модель jina-embeddings-v3"""

    def __init__(self, model_name="jinaai/jina-embeddings-v3", max_workers=3):
        logger.info("Загрузка модели %s", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Используется устройство: %s", self.device)

        # Пул потоков для асинхронности
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)


        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error("Ошибка при загрузке модели: %s", e)
            raise
        logger.info("Модель успешно загружена")

    def encode(
        self, texts: list, task="retrieval.passage", max_length=8192, dimensions=1024
    ) -> np.ndarray:
        """Получение эмбеддингов для списка текстов

        Args:
            texts (list): список текстов размера n
            task (str, optional): "retrieval.query" для запроса. для чанка "retrieval.passage".
            max_length (int, optional): . Defaults to 8192.
            dimensions (int, optional): размерность вектора эмбеддинга.
                Defaults to 1024(другие валидные 768, 512, 384, 256, 128, 64).

        Returns:
            numpy.ndarray: эмбеддинг размерности (1, dimensions) для запроса и (n, dimensions) для чанков
        """
        if not texts:
            raise ValueError("Список текстов не должен быть пустым")

        logger.debug(
            "Кодирование %d текстов, задача=%s, max_length=%d, dim=%d",
            len(texts),
            task,
            max_length,
            dimensions,
        )

        try:
            prefixed = [f"<{task}>{text}" for text in texts]
            batch = self.tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch)
                last_hidden = outputs.last_hidden_state

            # Mean pooling
            attention_mask = batch["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_emb / sum_mask

            # Matryoshka truncation
            if dimensions != 1024:
                if dimensions > 1024:
                    raise ValueError("Размерность не может быть больше 1024")
                mean_pooled = mean_pooled[:, :dimensions]
                logger.debug("Применено усечение Matryoshka до %dD", dimensions)

            # Normalize
            embeddings = F.normalize(mean_pooled, p=2, dim=1)
            result = embeddings.cpu().numpy().astype(np.float32)

            del batch, outputs, last_hidden
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(" Сгенерировано %d эмбеддингов", len(result))
            return result
        except Exception as e:
            logger.error("Ошибка при обработке текстов: %s", e)
            raise

    async def encode_async(self, texts: list, task="retrieval.passage", max_length=8192, dimensions=1024):
        """Асинхронная версия encode"""
        loop = asyncio.get_event_loop()
        
        try:
        # Добавляем timeout чтобы не вешать сервис
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool, 
                    lambda: self.encode(texts, task, max_length, dimensions)
                ),
                timeout=conf.EMBED_TIMEOUT
        )
        except asyncio.TimeoutError:
            logger.error("Timeout при генерации эмбеддингов для %d текстов", len(texts))
            raise HTTPException(status_code=504, detail="Timeout при генерации эмбеддингов")