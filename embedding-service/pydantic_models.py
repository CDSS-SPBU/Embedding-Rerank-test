from pydantic import BaseModel
from pydantic import Field, field_validator
from typing import List, Dict, Any, Optional
from config import Config

conf = Config()


class EmbedRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Список чанков каждый около 1500 символов",
        examples=[["Текст 1", "Текст 2"]]
    )  # Ограничение батча
    metadata: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Метаданные для каждого текста. Должен совпадать по длине с texts",
        examples=[[{"source": "doc1", "page": 2}, {"source": "doc2", "page": 4}]]
    )
    task: str = Field(
        default="retrieval.passage",
        pattern="^(retrieval\\.passage|retrieval\\.query)$",
        description="Тип задачи: retrieval.passage для чанков, retrieval.query для запроса",
        examples=["retrieval.passage", "retrieval.query"],
    )
    dimensions: int = Field(
        default=1024,
        ge=64,
        le=1024,
        description="Размерность вектора эмбеддинга, не менять default просто так",
        examples=[64, 128, 256, 384, 512, 768, 1024],
    )

    @field_validator("texts")
    @classmethod
    def validate_text_length(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("текст должен быть списком")
        if len(v) > conf.MAX_BATCH_SIZE:
            raise ValueError(
                f"Список текстов не может быть больше {conf.MAX_BATCH_SIZE}"
            )
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"texts[{i}] must be a string")
            stripped = text.strip()
            if len(stripped) == 0:
                raise ValueError("Текст не может быть пустым")
            if len(text) > conf.MAX_TEXT_LENGTH:
                raise ValueError("Текст слишком длинный")
        return v

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v):
        valid_dims = [64, 128, 256, 384, 512, 768, 1024]
        if v not in valid_dims:
            raise ValueError(f"Размерность должна быть из {valid_dims}")
        return v


class EmbedResponsePassage(BaseModel):
    """Схема ответа для эндпоинта /embed, task=retrieval.passage"""
    inserted_ids: List[int] = Field(..., description="ID вставленных записей в БД", examples=[[1, 2, 3]])
    count: int = Field(..., description="Количество обработанных чанков", examples=[3])
    batch_size: int = Field(..., description="Размер исходного батча", examples=[3])
    embedding_dimensions: int = Field(..., description="Размерность эмбеддингов", examples=[1024])
    status: str = Field(default="success", description="Статус операции", examples=["success"])


class EmbedResponseQuery(BaseModel):
    """Схема ответа для эндпоинта /embed, task=retrieval.query"""
    embedding: List[List[float]] = Field(..., description="Эмбеддинг запроса", examples=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    count: int = Field(..., description="Количество обработанных чанков", examples=[1])
    batch_size: int = Field(..., description="Размер исходного батча", examples=[1])
    embedding_dimensions: int = Field(..., description="Размерность эмбеддингов", examples=[1024])
    status: str = Field(default="success", description="Статус операции", examples=["success"])
