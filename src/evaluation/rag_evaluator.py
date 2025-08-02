import json
import os
import time
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import chromadb

# 표준 평가 지표들을 위한 라이브러리
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # NLTK 데이터 다운로드
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
except ImportError as e:
    print(f"⚠️ 일부 평가 라이브러리 설치 필요: {e}")
    print("pip install nltk rouge-score")

class RAGEvaluator:
    """RAG 시스템 성능 평가기 (표준 지표만 사용)"""

    def __init__(self, db_path: str = "data/embeddings/complete_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("complete_knowledge_qa")
        
        # 표준 평가 지표 초기화
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def load_validation_data(self) -> List[Dict]:
        """Validation 데이터 로드 (평가용 질문-답변 쌍)"""
        with open("data/processed/validation_qa.json", "r", encoding="utf-8") as f:
            validation_data = json.load(f)
        return validation_data

    def search_similar_docs(self, query: str, n_results: int = 3) -> List[Dict]:
        """Training DB에서 유사한 문서 검색"""
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        query_embedding = embedding_model.encode(query).tolist()

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

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """BLEU 점수 계산"""
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smooth)
        except:
            return 0.0

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def calculate_meteor_score(self, reference: str, candidate: str) -> float:
        """METEOR 점수 계산"""
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            return meteor_score([reference_tokens], candidate_tokens)
        except:
            return 0.0

    def evaluate_single_qa(self, qa_pair: Dict, generated_answer: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """단일 Q&A 쌍 평가 (표준 지표만 사용)"""
        question = qa_pair["question"]
        ground_truth_answer = qa_pair["answer"]

        # 표준 평가 지표 계산
        bleu_score = self.calculate_bleu_score(ground_truth_answer, generated_answer)
        rouge_scores = self.calculate_rouge_scores(ground_truth_answer, generated_answer)
        meteor_score_val = self.calculate_meteor_score(ground_truth_answer, generated_answer)

        # 종합 점수 (표준 지표들의 평균)
        comprehensive_score = np.mean([
            bleu_score,
            rouge_scores['rouge1'],
            rouge_scores['rouge2'],
            rouge_scores['rougeL'],
            meteor_score_val
        ])

        return {
            "question": question,
            "ground_truth": ground_truth_answer,
            "generated_answer": generated_answer,
            "retrieved_docs_count": len(retrieved_docs),
            
            # 표준 평가 지표
            "bleu_score": bleu_score,
            "rouge1_score": rouge_scores['rouge1'],
            "rouge2_score": rouge_scores['rouge2'],
            "rougeL_score": rouge_scores['rougeL'],
            "meteor_score": meteor_score_val,
            "comprehensive_score": comprehensive_score
        }

    def evaluate_rag_system(self, llm_system, sample_size: int = 50) -> Dict[str, Any]:
        """RAG 시스템 전체 평가 (표준 지표만 사용)"""
        print("🧪 RAG 시스템 평가 시작...")
        print("📋 평가 방식:")
        print("  1. Training 데이터로 벡터 DB 구축 (완료)")
        print("  2. Validation 데이터의 질문으로 검색")
        print("  3. 검색된 문서 + 질문 → LLM 답변 생성")
        print("  4. 생성된 답변 vs Validation 정답 비교")
        print("  5. 표준 평가 지표 (BLEU, ROUGE, METEOR) 계산")

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
            
            # 남은 시간 계산
            if i > 0:
                avg_time_per_sample = total_time / i
                remaining_samples = len(validation_sample) - i
                estimated_remaining_time = avg_time_per_sample * remaining_samples
                print(f"   예상 남은 시간: {estimated_remaining_time/60:.1f}분")
            
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

            # 3. 평가 (표준 지표만 사용)
            eval_result = self.evaluate_single_qa(qa_pair, generated_answer, retrieved_docs)
            eval_result["generation_time"] = generation_time
            evaluation_results.append(eval_result)

        # 전체 통계 계산
        bleu_scores = [r["bleu_score"] for r in evaluation_results]
        rouge1_scores = [r["rouge1_score"] for r in evaluation_results]
        rouge2_scores = [r["rouge2_score"] for r in evaluation_results]
        rougeL_scores = [r["rougeL_score"] for r in evaluation_results]
        meteor_scores = [r["meteor_score"] for r in evaluation_results]
        comprehensive_scores = [r["comprehensive_score"] for r in evaluation_results]
        generation_times = [r["generation_time"] for r in evaluation_results]

        # 성공률 계산 (답변 생성 실패 제외)
        successful_generations = [r for r in evaluation_results if r["generated_answer"] != "답변 생성 실패"]
        success_rate = len(successful_generations) / len(evaluation_results)

        overall_results = {
            "total_samples": len(evaluation_results),
            "successful_generations": len(successful_generations),
            "success_rate": success_rate,
            
            # 표준 평가 지표
            "avg_bleu_score": np.mean(bleu_scores),
            "avg_rouge1_score": np.mean(rouge1_scores),
            "avg_rouge2_score": np.mean(rouge2_scores),
            "avg_rougeL_score": np.mean(rougeL_scores),
            "avg_meteor_score": np.mean(meteor_scores),
            "avg_comprehensive_score": np.mean(comprehensive_scores),
            
            # 성능 지표
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
        """평가 결과 요약 출력 (표준 지표만)"""
        print(f"\n📊 {model_name} 평가 결과 요약")
        print("=" * 50)
        print(f"총 평가 샘플: {results['total_samples']}개")
        print(f"성공적 답변 생성: {results['successful_generations']}개")
        print(f"성공률: {results['success_rate']:.2%}")
        print()
        print("📈 표준 평가 지표:")
        print(f"  BLEU 점수: {results['avg_bleu_score']:.3f}")
        print(f"  ROUGE-1: {results['avg_rouge1_score']:.3f}")
        print(f"  ROUGE-2: {results['avg_rouge2_score']:.3f}")
        print(f"  ROUGE-L: {results['avg_rougeL_score']:.3f}")
        print(f"  METEOR: {results['avg_meteor_score']:.3f}")
        print(f"  종합 점수: {results['avg_comprehensive_score']:.3f}")
        print()
        print(f"⏱️ 성능:")
        print(f"  평균 생성 시간: {results['avg_generation_time']:.2f}초")
        print(f"  총 평가 시간: {results['total_evaluation_time']:.2f}초")
        print("=" * 50)

def main():
    """평가 실행 예시"""
    import sys
    import os
    
    # 프로젝트 루트를 Python 경로에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    evaluator = RAGEvaluator()

    # LLM 시스템 임포트
    from src.generation.llm_system import LLMSystem

    # 여러 모델 테스트
    models_to_test = [
        "EleutherAI/polyglot-ko-1.3b",
        "skt/kogpt2-base-v2",
        "beomi/gemma-ko-2b"
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