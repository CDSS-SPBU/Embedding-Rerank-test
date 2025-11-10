import os

class Config:
    MAX_WORKERS = 2
    MAX_LENGTH_QUERY = 500
    MAX_NUMBER_PASSAGE = 20

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "30.0"))
    RERANK_MAX_WORKERS = int(os.getenv("RERANK_MAX_WORKERS", "2"))