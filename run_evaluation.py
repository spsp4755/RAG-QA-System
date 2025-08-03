#!/usr/bin/env python3
"""
RAG 시스템 성능 평가 스크립트 (한국어 SBERT 벡터DB 비교 평가)
"""

import sys
import os
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_evaluation():
    """RAG 시스템 성능 평가 실행 (한국어 SBERT 벡터DB - beomi/gemma-ko-2b)"""
    print("🧪 한국어 SBERT 벡터DB 성능 평가 시작...")
    print("🤖 beomi/gemma-ko-2b 모델 평가")
    
    try:
        from src.evaluation.rag_evaluator import RAGEvaluator
        from src.generation.llm_system import LLMSystem
        
        # 한국어 SBERT DB 평가기 초기화
        print("📊 한국어 SBERT DB 평가기 초기화 중...")
        evaluator_korean_sbert = RAGEvaluator(
            db_path="data/embeddings/korean_sbert_db", 
            collection_name="korean_sbert_knowledge_qa"
        )
        
        # beomi/gemma-ko-2b 모델만 평가
        model_name = "beomi/gemma-ko-2b"
        
        print(f"\n🤖 {model_name} 모델 평가 시작...")
        
        try:
            # LLM 시스템 초기화
            print(f"📥 {model_name} 모델 로딩 중...")
            llm_system = LLMSystem(model_name)
            
            # 한국어 SBERT DB 평가 실행
            print(f"🔍 {model_name} 모델로 korean_sbert_db 평가 실행 중...")
            results = evaluator_korean_sbert.evaluate_rag_system(llm_system, sample_size=20)
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_clean = model_name.replace("/", "_")
            filename = evaluator_korean_sbert.save_evaluation_results(results, f"korean_sbert_db_{model_name_clean}_{timestamp}")
            
            # 결과 요약 출력
            print(f"\n📊 {model_name} 모델 평가 결과:")
            evaluator_korean_sbert.print_evaluation_summary(results, f"korean_sbert_db_{model_name_clean}")
            
            print("\n" + "="*60)
            print(f"✅ {model_name} 모델 평가 완료!")
            print(f"📁 결과 파일: {filename}")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"❌ {model_name} 모델 평가 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"❌ 평가 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_evaluation() 