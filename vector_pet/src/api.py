from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List, Optional
import json
import os

app = FastAPI(title="Семантический поиск по словарю Ожегова")

# Загружаем модель и подключаемся к Qdrant (переменные окружения или значения по умолчанию)
QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION = os.getenv('QDRANT_COLLECTION', 'ogegov')

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    word: str
    score: float
    full_text: str

@app.post('/search', response_model=List[SearchResult])
def search(request: SearchRequest):
    emb = model.encode(request.query)
    results = client.search(
        collection_name=COLLECTION,
        query_vector=emb.tolist(),
        limit=request.top_k,
        with_payload=True
    )
    return [SearchResult(
        word=hit.payload['word'],
        score=hit.score,
        full_text=hit.payload['full_text']
    ) for hit in results]

@app.get('/health')
def health():
    return {'status': 'ok'}

# Эндпоинт для оценки (опционально)
class EvalRequest(BaseModel):
    test_file: str  # путь внутри контейнера, например /data/test_queries.json

@app.post('/evaluate')
def evaluate(request: EvalRequest):
    if not os.path.exists(request.test_file):
        raise HTTPException(status_code=404, detail="Файл не найден")
    with open(request.test_file, 'r', encoding='utf-8') as f:
        tests = json.load(f)
    recall_at_k = {1:0, 3:0, 5:0}
    mrr = 0.0
    total = len(tests)
    if total == 0:
        return {'error': 'Нет тестовых запросов'}
    for test in tests:
        query = test['query']
        expected = test['expected']
        emb = model.encode(query)
        hits = client.search(COLLECTION, query_vector=emb.tolist(), limit=5, with_payload=True)
        rank = None
        for i, hit in enumerate(hits, 1):
            if hit.payload['word'].lower() == expected.lower():
                rank = i
                break
        if rank:
            mrr += 1.0 / rank
            for k in recall_at_k.keys():
                if rank <= k:
                    recall_at_k[k] += 1
    mrr /= total
    for k in recall_at_k:
        recall_at_k[k] /= total
    return {
        'recall@1': recall_at_k[1],
        'recall@3': recall_at_k[3],
        'recall@5': recall_at_k[5],
        'MRR': mrr
    }
