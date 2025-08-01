import json
import os
import time
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util
import chromadb

class RAGEvaluator:
    """RAG 시스템 성능 평가기"""
    
    def __init__(self, training_db_path: str = "data/embeddings/training_db"):
        self.training_db_path = training_db_path
        self.client = chromadb.PersistentClient(path=training_db_path)
        self.collection = self.client.get_collection("training_knowledge_qa")
        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
    def load_validation_data(self) -> List[Dict]:
        """Validation 데이터 로드 (평가용 질문-답변 쌍)"""
        with open("data/processed/validation_qa.json", "r", encoding="utf-8") as f:
            validation_data = json.load(f)
        return validation_data
    
    def search_similar_docs(self, query: str, n_results: int = 3) -> List[Dict]:
        """Training DB에서 유사한 문서 검색"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            result = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results.append(result)
            
        return formatted_results
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 의미적 유사도 계산"""
        emb1 = self.embedding_model.encode(text1)
        emb2 = self.embedding_model.encode(text2)
        similarity = util.cos_sim(emb1, emb2)[0][0].item()
        return similarity
    
    def evaluate_single_qa(self, qa_pair: Dict, generated_answer: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """단일 Q&A 쌍 평가"""
        question = qa_pair["question"]
        ground_truth_answer = qa_pair["answer"]
        
        # 1. 의미적 유사도 (Semantic Similarity)
        semantic_similarity = self.calculate_semantic_similarity(
            generated_answer, ground_truth_answer
        )
        
        # 2. 답변 길이 비율
        length_ratio = len(generated_answer) / len(ground_truth_answer) if len(ground_truth_answer) > 0 else 0
        
        # 3. 키워드 포함 여부 (간단한 버전)
        gt_words = set(ground_truth_answer.lower().split())
        gen_words = set(generated_answer.lower().split())
        keyword_overlap = len(gt_words.intersection(gen_words)) / len(gt_words) if len(gt_words) > 0 else 0
        
        # 4. 검색 품질 평가 (검색된 문서가 질문과 관련있는지)
        search_relevance = 0
        if retrieved_docs:
            # 검색된 문서들과 질문의 유사도 평균
            doc_similarities = []
            for doc in retrieved_docs:
                doc_sim = self.calculate_semantic_similarity(question, doc['text'])
                doc_similarities.append(doc_sim)
            search_relevance = np.mean(doc_similarities)
        
        return {
            "question": question,
            "ground_truth": ground_truth_answer,
            "generated_answer": generated_answer,
            "retrieved_docs_count": len(retrieved_docs),
            "semantic_similarity": semantic_similarity,
            "length_ratio": length_ratio,
            "keyword_overlap": keyword_overlap,
            "search_relevance": search_relevance,
            "evaluation_score": (semantic_similarity + keyword_overlap + search_relevance) / 3  # 종합 점수
        }
    
    def evaluate_rag_system(self, llm_system, sample_size: int = 50) -> Dict[str, Any]:
        """RAG 시스템 전체 평가"""
        print("🧪 RAG 시스템 평가 시작...")
        print("📋 평가 방식:")
        print("  1. Training 데이터로 벡터 DB 구축 (완료)")
        print("  2. Validation 데이터의 질문으로 검색")
        print("  3. 검색된 문서 + 질문 → LLM 답변 생성")
        print("  4. 생성된 답변 vs Validation 정답 비교")
        
        # Validation 데이터 로드
        validation_data = self.load_validation_data()
        print(f"📊 Validation 데이터: {len(validation_data)}개")
        
        # 샘플링 (전체 평가는 시간이 오래 걸림)
        if sample_size and sample_size < len(validation_data):
            import random
            random.seed(42)  # 재현 가능성을 위해
            validation_sample = random.sample(validation_data, sample_size)
        else:
            validation_sample = validation_data
        
        print(f"🔍 평가할 샘플: {len(validation_sample)}개")
        
        # 평가 실행
        evaluation_results = []
        total_time = 0
        
        for i, qa_pair in enumerate(validation_sample):
            print(f"평가 진행 중... ({i+1}/{len(validation_sample)})")
            
            start_time = time.time()
            
            # 1. Validation 질문으로 Training DB에서 검색
            retrieved_docs = self.search_similar_docs(qa_pair["question"], n_results=3)
            
            # 2. 검색된 문서 + 질문으로 LLM 답변 생성
            try:
                generated_answer = llm_system.generate_answer(qa_pair["question"], retrieved_docs)
            except Exception as e:
                print(f"답변 생성 실패: {e}")
                generated_answer = "답변 생성 실패"
            
            generation_time = time.time() - start_time
            total_time += generation_time
            
            # 3. 평가
            eval_result = self.evaluate_single_qa(qa_pair, generated_answer, retrieved_docs)
            eval_result["generation_time"] = generation_time
            evaluation_results.append(eval_result)
        
        # 전체 통계 계산
        semantic_similarities = [r["semantic_similarity"] for r in evaluation_results]
        length_ratios = [r["length_ratio"] for r in evaluation_results]
        keyword_overlaps = [r["keyword_overlap"] for r in evaluation_results]
        search_relevances = [r["search_relevance"] for r in evaluation_results]
        evaluation_scores = [r["evaluation_score"] for r in evaluation_results]
        generation_times = [r["generation_time"] for r in evaluation_results]
        
        # 성공률 계산 (답변 생성 실패 제외)
        successful_generations = [r for r in evaluation_results if r["generated_answer"] != "답변 생성 실패"]
        success_rate = len(successful_generations) / len(evaluation_results)
        
        overall_results = {
            "total_samples": len(evaluation_results),
            "successful_generations": len(successful_generations),
            "success_rate": success_rate,
            "avg_semantic_similarity": np.mean(semantic_similarities),
            "avg_length_ratio": np.mean(length_ratios),
            "avg_keyword_overlap": np.mean(keyword_overlaps),
            "avg_search_relevance": np.mean(search_relevances),
            "avg_evaluation_score": np.mean(evaluation_scores),
            "avg_generation_time": np.mean(generation_times),
            "total_evaluation_time": total_time,
            "detailed_results": evaluation_results
        }
        
        return overall_results
    
    def save_evaluation_results(self, results: Dict, model_name: str):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiments/evaluation_results_{model_name}_{timestamp}.json"
        
        os.makedirs("experiments", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 평가 결과 저장: {filename}")
        return filename
    
    def print_evaluation_summary(self, results: Dict, model_name: str):
        """평가 결과 요약 출력"""
        print(f"\n📊 {model_name} 평가 결과 요약")
        print("=" * 50)
        print(f"총 평가 샘플: {results['total_samples']}개")
        print(f"성공적 답변 생성: {results['successful_generations']}개")
        print(f"성공률: {results['success_rate']:.2%}")
        print(f"평균 의미적 유사도: {results['avg_semantic_similarity']:.3f}")
        print(f"평균 키워드 중복도: {results['avg_keyword_overlap']:.3f}")
        print(f"평균 검색 관련성: {results['avg_search_relevance']:.3f}")
        print(f"평균 종합 점수: {results['avg_evaluation_score']:.3f}")
        print(f"평균 생성 시간: {results['avg_generation_time']:.2f}초")
        print(f"총 평가 시간: {results['total_evaluation_time']:.2f}초")
        print("=" * 50)

def main():
    """평가 실행 예시"""
    evaluator = RAGEvaluator()
    
    # LLM 시스템 임포트
    from src.generation.llm_system import LLMSystem
    
    # 여러 모델 테스트
    models_to_test = [
        "EleutherAI/polyglot-ko-1.3b",
        "skt/kogpt2-base-v2"
    ]
    
    for model_name in models_to_test:
        print(f"\n🤖 {model_name} 모델 평가 시작...")
        
        try:
            llm_system = LLMSystem(model_name)
            results = evaluator.evaluate_rag_system(llm_system, sample_size=20)
            evaluator.save_evaluation_results(results, model_name.replace("/", "_"))
            evaluator.print_evaluation_summary(results, model_name)
        except Exception as e:
            print(f"❌ {model_name} 모델 평가 실패: {e}")

if __name__ == "__main__":
    main() 