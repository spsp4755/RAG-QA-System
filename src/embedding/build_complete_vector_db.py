import json
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

def build_complete_vector_db():
    """전체 DB 방식으로 벡터 DB 구축 (Training + Validation sentences + Training input/output)"""
    print("🏗️ 전체 DB 방식으로 벡터 DB 구축 시작...")

    # 1. 전체 데이터 로드 (Training + Validation)
    with open("data/processed/knowledge_qa.json", "r", encoding="utf-8") as f:
        complete_qa = json.load(f)

    print(f"📚 전체 데이터 로드 완료: {len(complete_qa)}개")

    # 2. 임베딩 모델 준비
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 3. ChromaDB 준비 (전체 DB용)
    persist_dir = "data/embeddings/complete_db"
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # 기존 컬렉션 삭제 (재구축)
    try:
        client.delete_collection("complete_knowledge_qa")
        print("🗑️ 기존 컬렉션 삭제 완료")
    except:
        pass

    collection = client.create_collection("complete_knowledge_qa")
    print("📦 새 컬렉션 생성 완료")

    # 4. 데이터 추가
    start_time = time.time()
    doc_count = 0

    for idx, qa in tqdm(enumerate(complete_qa), total=len(complete_qa), desc="전체 데이터 임베딩"):
        # 컨텍스트 (sentences)
        context = qa["context"]
        if context.strip():  # 빈 컨텍스트가 아닌 경우만
            metadata = qa.get("metadata", {}).copy()
            metadata.update({
                "question": qa["question"],
                "answer": qa["answer"],
                "instruction": qa.get("instruction", ""),
                "data_type": "context",  # 컨텍스트 데이터임을 표시
                "source": "complete_db"
            })

            # context 임베딩
            embedding = model.encode([context])[0]
            collection.add(
                ids=[f"context_{doc_count}"],
                embeddings=[embedding.tolist()],
                documents=[context],
                metadatas=[metadata]
            )
            doc_count += 1

        # Training 데이터의 경우 input/output도 추가 (Validation은 제외)
        # Validation 데이터는 sentences만 포함하고 input/output은 제외
        # 이는 데이터 소스를 통해 구분 (Training/Validation 폴더 구조)
        question = qa["question"]
        answer = qa["answer"]
        
        # Training 데이터의 input/output을 별도 문서로 추가
        qa_text = f"질문: {question}\n답변: {answer}"
        if qa_text.strip():
            metadata = qa.get("metadata", {}).copy()
            metadata.update({
                "question": question,
                "answer": answer,
                "instruction": qa.get("instruction", ""),
                "data_type": "qa_pair",  # Q&A 쌍 데이터임을 표시
                "source": "complete_db"
            })

            # Q&A 쌍 임베딩
            embedding = model.encode([qa_text])[0]
            collection.add(
                ids=[f"qa_pair_{doc_count}"],
                embeddings=[embedding.tolist()],
                documents=[qa_text],
                metadatas=[metadata]
            )
            doc_count += 1

    total_time = time.time() - start_time
    print(f"✅ 전체 DB 벡터 DB 구축 완료!")
    print(f"📊 총 문서 수: {collection.count()}")
    print(f"⏱️ 소요 시간: {total_time:.2f}초")

    return persist_dir

if __name__ == "__main__":
    build_complete_vector_db() 