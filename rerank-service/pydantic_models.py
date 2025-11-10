from pydantic import BaseModel
from pydantic import Field, model_validator
from typing import List, Dict, Any, Optional
from config import Config

conf = Config()


class RerankRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=conf.MAX_LENGTH_QUERY, examples=["Текст запроса"]
    )
    passages: List[str] = Field(
        ...,
        min_length=1,
        max_length=conf.MAX_NUMBER_PASSAGE,
        examples=[["Текст 1", "Текст 2"]],
    )
    metadata: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Метаданные для каждого текста. Должен совпадать по длине с texts",
        examples=[[{"source": "doc1", "page": 2}, {"source": "doc2", "page": 4}]],
    )
    max_tokens: int = 512  # максимальное количество токенов

    def _estimate_tokens(self, text: str) -> int:
        """Грубая оценка токенов для русского текста"""
        return len(text) // 2

    @model_validator(mode="after")
    def validate_token_length(self):
        query_tokens = self._estimate_tokens(self.query)

        for i, passage in enumerate(self.passages):
            passage_tokens = self._estimate_tokens(passage)
            total_tokens = query_tokens + passage_tokens

            if total_tokens > self.max_tokens:
                raise ValueError(
                    f"Предполагаемое количество токенов для пары {i} превышает {self.max_tokens}: "
                    f"query(~{query_tokens}) + passage(~{passage_tokens}) = ~{total_tokens} токенов"
                )
        return self


class RerankedItem(BaseModel):
    text: str = Field(..., description="Текст")
    score: float = Field(..., description="Скор для ранжирования")
    rank: int = Field(..., description="Позиция в ранжированном списке")


class RerankResponse(BaseModel):
    reranked_results: List[RerankedItem]
    original_query: str = Field(..., description="Исходный запрос")
