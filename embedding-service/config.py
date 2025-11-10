import os

class Config:
    # для модели
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "12"))
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10_000"))
    EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "30.0"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # для бд
    DB_HOST = os.getenv("DB_HOST", "postgres")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_USER = os.getenv("DB_USER", "embedd_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "embedd_password")
    DB_NAME = os.getenv("DB_NAME", "rag")
    DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", 1))
    DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", 10))

