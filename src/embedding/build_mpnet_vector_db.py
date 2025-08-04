import json
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

def build_mpnet_vector_db():
    """고성능 다국어 MPNet 모델로 벡터 DB 구축"""
    print("🏗️ 고성능 다국어 MPNet 모델로 벡터 DB 구축 시작...")

    # 1. Training 데이터와 Validation 데이터 모두 로드
    with open("data/processed/training_qa.json", "r", encoding="utf-8") as f:
        training_qa = json.load(f)
    
    with open("data/processed/validation_qa.json", "r", encoding="utf-8") as f:
        validation_qa = json.load(f)

    print(f"📚 Training 데이터 로드 완료: {len(training_qa)}개")
    print(f"📚 Validation 데이터 로드 완료: {len(validation_qa)}개")

    # 2. 고성능 다국어 MPNet 임베딩 모델 준비
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print("🔤 고성능 다국어 MPNet 모델 로드 완료")

    # 3. ChromaDB 준비 (MPNet용)
    persist_dir = "data/embeddings/mpnet_db"
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # 기존 컬렉션 삭제 (재구축)
    try:
        client.delete_collection("mpnet_knowledge_qa")
        print("🗑️ 기존 컬렉션 삭제 완료")
    except:
        pass

    collection = client.create_collection("mpnet_knowledge_qa")
    print("📦 새 컬렉션 생성 완료")

    # 4. 데이터 추가
    start_time = time.time()
    doc_count = 0

    # Training 데이터 처리 (context + input/output 모두 포함)
    for idx, qa in tqdm(enumerate(training_qa), total=len(training_qa), desc="Training 데이터 임베딩"):
        # 컨텍스트 (sentences)
        context = qa["context"]
        if context.strip():  # 빈 컨텍스트가 아닌 경우만
            metadata = qa.get("metadata", {}).copy()
            metadata.update({
                "question": qa["question"],
                "answer": qa["answer"],
                "instruction": qa.get("instruction", ""),
                "data_type": "context",  # 컨텍스트 데이터임을 표시
                "source": "mpnet_db",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "data_split": "training"  # Training 데이터임을 표시
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

        # Training 데이터의 input/output을 별도 문서로 추가
        question = qa["question"]
        answer = qa["answer"]
        
        qa_text = f"질문: {question}\n답변: {answer}"
        if qa_text.strip():
            metadata = qa.get("metadata", {}).copy()
            metadata.update({
                "question": question,
                "answer": answer,
                "instruction": qa.get("instruction", ""),
                "data_type": "qa_pair",  # Q&A 쌍 데이터임을 표시
                "source": "mpnet_db",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "data_split": "training"  # Training 데이터임을 표시
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

    # Validation 데이터 처리 (context만 포함, input/output은 제외)
    for idx, qa in tqdm(enumerate(validation_qa), total=len(validation_qa), desc="Validation 데이터 임베딩"):
        # 컨텍스트 (sentences)만 포함
        context = qa["context"]
        if context.strip():  # 빈 컨텍스트가 아닌 경우만
            metadata = qa.get("metadata", {}).copy()
            metadata.update({
                "question": qa["question"],
                "answer": qa["answer"],
                "instruction": qa.get("instruction", ""),
                "data_type": "context",  # 컨텍스트 데이터임을 표시
                "source": "mpnet_db",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "data_split": "validation"  # Validation 데이터임을 표시
            })

            # context 임베딩
            embedding = model.encode([context])[0]
            collection.add(
                ids=[f"validation_context_{doc_count}"],
                embeddings=[embedding.tolist()],
                documents=[context],
                metadatas=[metadata]
            )
            doc_count += 1

    total_time = time.time() - start_time
    print(f"✅ 고성능 다국어 MPNet 벡터 DB 구축 완료!")
    print(f"📊 총 문서 수: {collection.count()}")
    print(f"⏱️ 소요 시간: {total_time:.2f}초")
    print(f"🔤 사용 모델: sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    return persist_dir

if __name__ == "__main__":
    build_mpnet_vector_db() 