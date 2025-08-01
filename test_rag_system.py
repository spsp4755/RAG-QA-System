#!/usr/bin/env python3
"""
RAG 시스템 간단 테스트 스크립트
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rag_system():
    """RAG 시스템 테스트"""
    print("🧪 RAG 시스템 테스트 시작...")
    
    try:
        from src.rag_system import RAGSystem
        
        # RAG 시스템 초기화 (training_db 사용)
        print("🚀 RAG 시스템 초기화 중...")
        rag_system = RAGSystem(
            persist_dir="data/embeddings/training_db",
            use_llm=True
        )
        
        # 시스템 정보 출력
        info = rag_system.get_system_info()
        print(f"📊 총 문서 수: {info['total_documents']}")
        print(f"🔍 검색 시스템: {info['search_system']}")
        print(f"🤖 LLM 시스템: {info['llm_system']}")
        
        # 테스트 질문들
        test_queries = [
            "특허권의 보호기간은 얼마나 되나요?",
            "상표권 침해의 구체적인 행위는 무엇인가요?",
            "저작권의 발생 시점은 언제인가요?"
        ]
        
        print("\n" + "="*50)
        print("🔍 테스트 질문들:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 질문: {query}")
            print("-" * 30)
            
            try:
                result = rag_system.answer_question(query, n_results=3, save_result=False)
                
                print(f"🤖 답변: {result['answer'][:200]}...")
                print(f"📄 검색된 문서 수: {len(result['retrieved_docs'])}")
                print(f"⏱️ 검색 시간: {result['search_time']:.2f}초")
                print(f"⏱️ 생성 시간: {result['generation_time']:.2f}초")
                print(f"⏱️ 총 시간: {result['total_time']:.2f}초")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
        print("\n" + "="*50)
        print("✅ RAG 시스템 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_rag_system() 