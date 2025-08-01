import json
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# 1. 데이터 로드
with open("data/processed/knowledge_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# 2. 임베딩 모델 준비
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 3. ChromaDB 준비
os.makedirs("data/embeddings/knowledge_qa", exist_ok=True)
client = chromadb.PersistentClient(path="data/embeddings/knowledge_qa")
collection = client.get_or_create_collection("knowledge_qa")

# 4. 데이터 추가
for idx, qa in tqdm(enumerate(qa_data), total=len(qa_data), desc="Embedding and storing"):
    context = qa["context"]
    question = qa["question"]
    answer = qa["answer"]
    instruction = qa.get("instruction", "")
    metadata = qa.get("metadata", {}).copy()
    # 메타데이터에 question, answer, instruction도 추가
    metadata.update({
        "question": question,
        "answer": answer,
        "instruction": instruction
    })
    # context 임베딩
    embedding = model.encode([context])[0]
        collection.add(
        ids=[str(idx)],
        embeddings=[embedding],
        documents=[context],
        metadatas=[metadata]
    )

if __name__ == "__main__":
    start = time.time()
    print(f"총 소요 시간: {time.time() - start:.2f}초")