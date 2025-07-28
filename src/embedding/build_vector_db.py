import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def load_chunks(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def build_chroma_db(chunks, persist_dir="data/embeddings/contract_legal"):
    # ChromaDB 세팅 - PersistentClient 사용
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("contract_legal_docs")

    # 임베딩 모델 로드
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 벡터DB에 저장
    for chunk in tqdm(chunks, desc="Embedding and storing chunks"):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        metadata = chunk["metadata"]
        embedding = model.encode(text).tolist()
        collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[chunk_id],
            embeddings=[embedding]
        )
    print(f"ChromaDB built at {persist_dir}")

if __name__ == "__main__":
    chunks = load_chunks("data/processed/contract_legal_chunks.json")
    build_chroma_db(chunks)