from pydantic import BaseModel
from pydantic import Field, field_validator
from typing import List, Dict, Any, Optional

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)  # Ограничение батча
    metadata: Optional[List[Dict[str, Any]]] = None
    task: str = Field(default="retrieval.passage", pattern="^(retrieval\\.passage|retrieval\\.query)$")
    dimensions: int = Field(default=1024, ge=64, le=1024)

    @field_validator('texts')
    @classmethod
    def validate_text_length(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise ValueError('текст должен быть списком')
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f'texts[{i}] must be a string')
            stripped = text.strip()
            if len(stripped) == 0:
                raise ValueError('Текст не может быть пустым')
            if len(text) > 100000:
                raise ValueError('Текст слишком длинный')
        return v

    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v):
        valid_dims = [64, 128, 256, 384, 512, 768, 1024]
        if v not in valid_dims:
            raise ValueError(f'Размерность должна быть из {valid_dims}')
        return v