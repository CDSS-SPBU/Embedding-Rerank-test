from fastapi import FastAPI, WebSocket
from transformers import pipeline, TextIteratorStreamer
from threading import Thread
import asyncio
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import asyncpg

# Глобальное подключение к БД (можно инициализировать при старте)
DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/rag"


# Клиент для HTTP-запросов
async_client = httpx.AsyncClient()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static/"), name="static")

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()
    await async_client.aclose()


def mock_llm(query: str):
    return "Hello, I am a mock LLM. You said: " + query

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_input = await websocket.receive_text()
            print(f"Получено: {user_input}")

            # === Делаем запрос к эмбеддинг-серверу ===
            embedding_payload = {
                "texts": [user_input],
                "metadata": [{}],
                "task": "retrieval.query",
                "dimensions": 1024
            }

            try:
                response = await async_client.post(
                    "http://localhost:8000/embed",
                    json=embedding_payload,
                    timeout=30.0
                )
                embedding_result = response.json()
                print(f"Embedding response: {embedding_result}")

            

            except Exception as e:
                print(f"Ошибка при запросе к эмбеддингу: {e}")
                await websocket.send_text(f"[Ошибка эмбеддинга: {e}]")

            query_embedding = embedding_result["embedding"][0]  # список из 1024 float

            # === 2. Ищем похожие записи в PostgreSQL с pgvector ===
            async with db_pool.acquire() as conn:
                # Важно: преобразуем список Python -> строку в формате pgvector: '[x1,x2,...]'
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

                # Запрос: косинусное сходство (1 - cosine_distance), сортировка по сходству
                rows = await conn.fetch(
                    """
                    SELECT content
                    FROM chunks
                    ORDER BY embedding <=> $1::vector
                    LIMIT 10;
                    """,
                    embedding_str
                )

            response = mock_llm(rows)

            await websocket.send_text(response)

    except Exception as e:
        print(f"Ошибка WebSocket: {e}")
        await websocket.close()