import os
import json
from typing import List, Dict, Optional
from datetime import datetime

from src.retrieval.search_system import SearchSystem
from src.generation.llm_system import SimpleLLMSystem, LLMSystem


class RAGSystem:
    """완전한 RAG (Retrieval-Augmented Generation) 시스템"""
    
    def __init__(self, 
                 persist_dir: str = "data/embeddings/contract_legal",
                 use_llm: bool = False,
                 llm_model: str = "microsoft/DialoGPT-medium"):
        """
        RAG 시스템 초기화
        
        Args:
            persist_dir: ChromaDB 저장 경로
            use_llm: 실제 LLM 사용 여부 (False면 SimpleLLMSystem 사용)
            llm_model: 사용할 LLM 모델명
        """
        print("🚀 RAG 시스템 초기화 중...")
        
        # 검색 시스템 초기화
        self.search_system = SearchSystem(persist_dir)
        
        # LLM 시스템 초기화
        if use_llm:
            self.llm_system = LLMSystem(llm_model)
        else:
            self.llm_system = SimpleLLMSystem()
        
        print("✅ RAG 시스템 초기화 완료")
    
    def answer_question(self, 
                       query: str, 
                       n_results: int = 5,
                       filter_dict: Optional[Dict] = None,
                       save_result: bool = True) -> Dict:
        """
        질문에 대한 답변 생성
        
        Args:
            query: 사용자 질문
            n_results: 검색할 문서 수
            filter_dict: 메타데이터 필터
            save_result: 결과 저장 여부
            
        Returns:
            답변 결과 딕셔너리
        """
        start_time = datetime.now()
        
        # 1. 문서 검색 (Retrieval)
        print(f"🔍 질문 검색 중: '{query}'")
        retrieved_docs = self.search_system.search(query, n_results, filter_dict)
        
        if not retrieved_docs:
            return {
                'query': query,
                'answer': '죄송합니다. 관련된 문서를 찾을 수 없습니다.',
                'retrieved_docs': [],
                'search_time': (datetime.now() - start_time).total_seconds(),
                'generation_time': 0,
                'total_time': (datetime.now() - start_time).total_seconds()
            }
        
        search_time = datetime.now() - start_time
        print(f"📄 {len(retrieved_docs)}개 문서 검색 완료 ({search_time.total_seconds():.2f}초)")
        
        # 2. 답변 생성 (Generation)
        generation_start = datetime.now()
        print("🤖 답변 생성 중...")
        answer = self.llm_system.generate_answer(query, retrieved_docs)
        generation_time = datetime.now() - generation_start
        
        total_time = datetime.now() - start_time
        
        # 3. 결과 구성
        result = {
            'query': query,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'search_time': search_time.total_seconds(),
            'generation_time': generation_time.total_seconds(),
            'total_time': total_time.total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 4. 결과 저장 (선택사항)
        if save_result:
            self._save_result(result)
        
        print(f"✅ 답변 생성 완료 (총 {total_time.total_seconds():.2f}초)")
        
        return result
    
    def _save_result(self, result: Dict):
        """결과를 파일에 저장"""
        os.makedirs("experiments/results", exist_ok=True)
        
        filename = f"experiments/results/rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 결과 저장됨: {filename}")
    
    def get_system_info(self) -> Dict:
        """시스템 정보 반환"""
        stats = self.search_system.get_collection_stats()
        
        return {
            'search_system': 'ChromaDB + SentenceTransformer',
            'llm_system': type(self.llm_system).__name__,
            'total_documents': stats['total_documents'],
            'persist_directory': stats['persist_directory']
        }


def interactive_qa():
    """대화형 질의응답 시스템"""
    print("🎯 RAG 기반 법률 문서 QA 시스템")
    print("=" * 50)
    
    # RAG 시스템 초기화
    rag_system = RAGSystem(use_llm=False)  # 간단한 시스템으로 시작
    
    # 시스템 정보 출력
    info = rag_system.get_system_info()
    print(f"📊 총 문서 수: {info['total_documents']}")
    print(f"🔍 검색 시스템: {info['search_system']}")
    print(f"🤖 LLM 시스템: {info['llm_system']}")
    print()
    
    print("💡 질문 예시:")
    print("  - 계약서의 기본 조항은 무엇인가요?")
    print("  - 임대차 계약에서 임차인의 의무는?")
    print("  - 계약 해지 조건에 대해 알려주세요")
    print("  - 법적 책임과 의무는 어떻게 되나요?")
    print()
    
    while True:
        try:
            query = input("❓ 질문을 입력하세요 (종료: 'quit' 또는 'exit'): ").strip()
            
            if query.lower() in ['quit', 'exit', '종료']:
                print("👋 시스템을 종료합니다.")
                break
            
            if not query:
                print("⚠️  질문을 입력해주세요.")
                continue
            
            print("\n" + "="*50)
            
            # 답변 생성
            result = rag_system.answer_question(query, n_results=3)
            
            # 결과 출력
            print(f"🤖 답변:")
            print(result['answer'])
            print()
            
            print(f"📄 참고 문서 ({len(result['retrieved_docs'])}개):")
            for i, doc in enumerate(result['retrieved_docs'], 1):
                print(f"  {i}. 거리: {doc['distance']:.4f}")
                print(f"     문서: {doc['text'][:100]}...")
                print(f"     메타데이터: {doc['metadata']}")
                print()
            
            print(f"⏱️  검색 시간: {result['search_time']:.2f}초")
            print(f"⏱️  생성 시간: {result['generation_time']:.2f}초")
            print(f"⏱️  총 시간: {result['total_time']:.2f}초")
            print("="*50)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            print("다시 시도해주세요.")


def test_rag_system():
    """RAG 시스템 테스트"""
    print("🧪 RAG 시스템 테스트 시작...")
    
    # RAG 시스템 초기화
    rag_system = RAGSystem(use_llm=False)
    
    # 테스트 질문들
    test_queries = [
        "계약서의 기본 조항",
        "임대차 계약 조건",
        "법적 책임과 의무"
    ]
    
    for query in test_queries:
        print(f"\n🔍 테스트 질문: '{query}'")
        result = rag_system.answer_question(query, n_results=2, save_result=False)
        
        print(f"답변: {result['answer'][:200]}...")
        print(f"검색된 문서 수: {len(result['retrieved_docs'])}")
        print(f"총 소요 시간: {result['total_time']:.2f}초")


if __name__ == "__main__":
    # 테스트 모드 또는 대화형 모드 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_rag_system()
    else:
        interactive_qa() 