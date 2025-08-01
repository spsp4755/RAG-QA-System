#!/usr/bin/env python3
"""
RAG 시스템 성능 평가 스크립트
"""

import sys
import os
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_evaluation():
    """RAG 시스템 성능 평가 실행"""
    print("🧪 RAG 시스템 성능 평가 시작...")
    
    try:
        from src.evaluation.rag_evaluator import RAGEvaluator
        from src.generation.llm_system import LLMSystem
        
        # 평가기 초기화
        print("📊 평가기 초기화 중...")
        evaluator = RAGEvaluator()
        
        # LLM 시스템 초기화
        print("🤖 LLM 시스템 초기화 중...")
        llm_system = LLMSystem("EleutherAI/polyglot-ko-1.3b")
        
        # 평가 실행 (샘플 크기: 20개)
        print("🔍 평가 실행 중...")
        results = evaluator.evaluate_rag_system(llm_system, sample_size=20)
        
        # 결과 저장
        model_name = "EleutherAI_polyglot_ko_1_3b"
        filename = evaluator.save_evaluation_results(results, model_name)
        
        # 결과 요약 출력
        evaluator.print_evaluation_summary(results, model_name)
        
        print(f"\n✅ 평가 완료! 결과 저장: {filename}")
        
        return results
        
    except Exception as e:
        print(f"❌ 평가 실행 실패: {e}")
        return None

if __name__ == "__main__":
    run_evaluation() 