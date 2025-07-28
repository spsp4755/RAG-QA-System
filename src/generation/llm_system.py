import os
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMSystem:
    """로컬 LLM 기반 답변 생성 시스템"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        LLM 시스템 초기화
        
        Args:
            model_name: 사용할 모델명 (CPU 호환 모델 권장)
        """
        self.model_name = model_name
        self.device = "cpu"  # MacBook CPU 사용
        
        print(f"🤖 LLM 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU 호환
            low_cpu_mem_usage=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ LLM 모델 로딩 완료")
    
    def generate_answer(self, query: str, context_docs: List[Dict], max_length: int = 512) -> str:
        """
        컨텍스트 기반 답변 생성
        
        Args:
            query: 사용자 질문
            context_docs: 검색된 관련 문서들
            max_length: 최대 생성 길이
            
        Returns:
            생성된 답변
        """
        # 컨텍스트 구성
        context_text = self._build_context(context_docs)
        
        # 프롬프트 구성
        prompt = self._build_prompt(query, context_text)
        
        # 토큰화
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        # 답변 생성
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거하고 답변만 추출
        answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def _build_context(self, context_docs: List[Dict]) -> str:
        """검색된 문서들을 컨텍스트로 구성"""
        context_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            doc_text = doc['text']
            metadata = doc['metadata']
            
            context_part = f"[문서 {i}] {doc_text}"
            if metadata.get('title'):
                context_part = f"[문서 {i} - {metadata['title']}] {doc_text}"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """RAG 프롬프트 구성"""
        prompt = f"""다음은 법률 문서 관련 질문에 답변하기 위한 컨텍스트입니다.

컨텍스트:
{context}

질문: {query}

위의 컨텍스트를 바탕으로 질문에 답변해주세요. 답변은 한국어로 작성하고, 컨텍스트에 없는 내용은 추측하지 마세요.

답변:"""
        
        return prompt


class SimpleLLMSystem:
    """간단한 템플릿 기반 답변 시스템 (LLM 없이)"""
    
    def __init__(self):
        pass
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        템플릿 기반 답변 생성 (LLM 없이)
        
        Args:
            query: 사용자 질문
            context_docs: 검색된 관련 문서들
            
        Returns:
            생성된 답변
        """
        if not context_docs:
            return "죄송합니다. 관련된 문서를 찾을 수 없습니다."
        
        # 가장 관련성 높은 문서 선택
        best_doc = context_docs[0]
        
        # 간단한 템플릿 기반 답변
        answer = f"""질문: {query}

관련 문서에서 찾은 내용:
{best_doc['text']}

이 문서는 {best_doc['metadata'].get('doc_type', '법률 문서')}에 속하며, 
제목은 "{best_doc['metadata'].get('title', '제목 없음')}"입니다.

더 자세한 정보가 필요하시면 다른 질문을 해주세요."""
        
        return answer


def test_llm_system():
    """LLM 시스템 테스트"""
    print("🤖 LLM 시스템 테스트 시작...")
    
    # 간단한 시스템으로 테스트 (빠른 테스트용)
    llm_system = SimpleLLMSystem()
    
    # 테스트용 컨텍스트
    test_context = [
        {
            'text': '계약서에는 당사자의 기본 정보, 계약 목적, 계약 기간, 계약 조건 등이 명시되어야 합니다.',
            'metadata': {
                'doc_type': '계약서',
                'title': '계약서 기본 조항'
            }
        }
    ]
    
    test_query = "계약서에는 어떤 내용이 포함되어야 하나요?"
    
    answer = llm_system.generate_answer(test_query, test_context)
    print(f"질문: {test_query}")
    print(f"답변: {answer}")


if __name__ == "__main__":
    test_llm_system() 