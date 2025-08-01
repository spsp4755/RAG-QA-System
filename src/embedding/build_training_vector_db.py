import json
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def build_training_vector_db():
    """Training 데이터로 벡터 DB 구축"""
    print("🏗️ Training 데이터로 벡터 DB 구축 시작...")
    
    # 1. Training 데이터 로드
    with open("data/processed/training_qa.json", "r", encoding="utf-8") as f:
        training_qa = json.load(f)
    
    print(f"📚 Training 데이터 로드 완료: {len(training_qa)}개")
    
    # 2. 임베딩 모델 준비
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 3. ChromaDB 준비 (Training 전용)
    persist_dir = "data/embeddings/training_db"
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    
    # 기존 컬렉션 삭제 (재구축)
    try:
        client.delete_collection("training_knowledge_qa")
        print("🗑️ 기존 컬렉션 삭제 완료")
    except:
        pass
    
    collection = client.create_collection("training_knowledge_qa")
    print("📦 새 컬렉션 생성 완료")
    
    # 4. Training 데이터 추가
    start_time = time.time()
    for idx, qa in tqdm(enumerate(training_qa), total=len(training_qa), desc="Training 데이터 임베딩"):
        context = qa["context"]
        question = qa["question"]
        answer = qa["answer"]
        instruction = qa.get("instruction", "")
        metadata = qa.get("metadata", {}).copy()
        
        # 메타데이터에 question, answer, instruction 추가
        metadata.update({
            "question": question,
            "answer": answer,
            "instruction": instruction,
            "data_type": "training"  # Training 데이터임을 표시
        })
        
        # context 임베딩
        embedding = model.encode([context])[0]
        collection.add(
            ids=[f"training_{idx}"],
            embeddings=[embedding],
            documents=[context],
            metadatas=[metadata]
        )
    
    total_time = time.time() - start_time
    print(f"✅ Training 벡터 DB 구축 완료!")
    print(f"📊 총 문서 수: {collection.count()}")
    print(f"⏱️ 소요 시간: {total_time:.2f}초")
    
    return persist_dir

if __name__ == "__main__":
    build_training_vector_db() 