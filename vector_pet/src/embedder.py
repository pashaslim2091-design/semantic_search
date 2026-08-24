from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import json
import os

def index_articles(articles_path: str,
                   collection_name: str = 'ogegov',
                   qdrant_host: str = 'qdrant',
                   qdrant_port: int = 6333,
                   batch_size: int = 100):
    """
    Индексирует статьи в Qdrant, разбивая на батчи для экономии памяти.
    Если коллекция не существует — создаёт её, иначе обновляет/добавляет точки (upsert).
    """
    if not os.path.exists(articles_path):
        raise FileNotFoundError(f"Файл {articles_path} не найден")

    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Проверяем коллекцию — создаём, если отсутствует
    if not client.collection_exists(collection_name):
        # Получим размерность векторов по первому элементу
        sample_text = articles[0]['full_text']
        sample_emb = model.encode(sample_text)
        vector_size = sample_emb.shape[0]
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Создана новая коллекция '{collection_name}' (размерность {vector_size})")
    else:
        print(f"ℹ️ Коллекция '{collection_name}' уже существует — будет выполнено обновление (upsert)")

    total = len(articles)
    print(f"📚 Начинаем индексацию {total} статей, батч-размер: {batch_size}")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_articles = articles[start:end]
        texts = [a['full_text'] for a in batch_articles]

        # Генерируем эмбеддинги для батча (без прогресс-бара, чтобы не засорять логи)
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

        points = []
        for idx, (art, emb) in enumerate(zip(batch_articles, embeddings)):
            point_id = start + idx   # id = глобальный индекс статьи
            points.append(PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload={'word': art['word'], 'full_text': art['full_text']}
            ))

        # Upsert — обновляет существующие или добавляет новые точки
        client.upsert(collection_name=collection_name, points=points)
        print(f"   Обработаны статьи {start+1}-{end} из {total}")

    print(f"✅ Индексация завершена. Всего точек в коллекции: {total}")
