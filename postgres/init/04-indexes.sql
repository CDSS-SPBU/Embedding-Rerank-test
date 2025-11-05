-- IVFFlat индекс для быстрого поиска (подходит для до ~1M векторов)
CREATE INDEX IF NOT EXISTS hnsw_chunks_embedding ON chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);