import json
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class SearchSystem:
    """ChromaDB 기반 검색 시스템"""
    
    def __init__(self, persist_dir: str = "data/embeddings/training_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("training_knowledge_qa")
        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        쿼리로 관련 문서 검색
        
        Args:
            query: 검색할 질문
            n_results: 반환할 결과 수
            filter_dict: 메타데이터 필터 (예: {"doc_type": "계약서"})
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # ChromaDB에서 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        # 결과 포맷팅
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
    
    def search_by_metadata(self, filter_dict: Dict, n_results: int = 10) -> List[Dict]:
        """
        메타데이터로 문서 검색
        
        Args:
            filter_dict: 메타데이터 필터
            n_results: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        results = self.collection.get(
            where=filter_dict,
            limit=n_results
        )
        
        formatted_results = []
        for i in range(len(results['documents'])):
            result = {
                'text': results['documents'][i],
                'metadata': results['metadatas'][i],
                'id': results['ids'][i]
            }
            formatted_results.append(result)
            
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보 반환"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'persist_directory': self.persist_dir
        }


def test_search_system():
    """검색 시스템 테스트"""
    print("🔍 검색 시스템 테스트 시작...")
    
    # 검색 시스템 초기화
    search_system = SearchSystem()
    
    # 컬렉션 통계 확인
    stats = search_system.get_collection_stats()
    print(f"📊 총 문서 수: {stats['total_documents']}")
    
    # 테스트 쿼리들
    test_queries = [
        "계약서의 기본 조항",
        "임대차 계약 조건",
        "법적 책임과 의무",
        "계약 해지 조건"
    ]
    
    for query in test_queries:
        print(f"\n🔎 쿼리: '{query}'")
        results = search_system.search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. 거리: {result['distance']:.4f}")
            print(f"     문서: {result['text'][:100]}...")
            print(f"     메타데이터: {result['metadata']}")
            print()


if __name__ == "__main__":
    test_search_system() 