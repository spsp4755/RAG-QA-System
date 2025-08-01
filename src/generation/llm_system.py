import os
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import util


class LLMSystem:
    """로컬 LLM 기반 답변 생성 시스템"""
    
    def __init__(self, model_name: str = "EleutherAI/polyglot-ko-1.3b"):
        """
        LLM 시스템 초기화
        
        Args:
            model_name: 사용할 모델명 (CPU 호환 모델 권장)
        """
        self.model_name = model_name
        self.device = "cpu"  # MacBook CPU 사용
        
        print(f"🤖 LLM 모델 로딩 중: {model_name}")
        try:
            print(f"📥 토크나이저 다운로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"📥 모델 다운로드 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU 호환
                low_cpu_mem_usage=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 임베딩 모델 로드 (문장 추출용)
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                
            print("✅ LLM 모델 로딩 완료")
        except Exception as e:
            print(f"❌ LLM 모델 로딩 실패: {e}")
            print("⚠️ SimpleLLMSystem으로 대체합니다.")
            raise e
    
    def generate_answer(self, query: str, context_docs: List[Dict], max_new_tokens: int = 256) -> str:
        """
        문장 추출 기반 컨텍스트 생성 + 간결 프롬프트
        """
        if not context_docs:
            return "죄송합니다. 관련된 문서를 찾을 수 없습니다."
        try:
            context_text = self._build_context(context_docs, query)
            prompt = self._build_prompt(query, context_text)
            print(f"프롬프트 길이: {len(prompt)} 문자")
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            print(f"입력 토큰 수: {inputs.shape[1]}")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,  # 반복 방지
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):].strip()
            print(f"생성된 답변 길이: {len(answer)} 문자")
            if not answer:
                return "죄송합니다. 답변을 생성할 수 없습니다. 다시 시도해주세요."
            return answer
        except Exception as e:
            print(f"LLM 생성 오류: {e}")
            return f"오류가 발생했습니다: {str(e)}"

    def _build_context(self, context_docs: List[Dict], question: str) -> str:
        # 문서별로 질문과 가장 유사한 문장 1-2개만 추출 (더 짧게)
        context_parts = []
        for i, doc in enumerate(context_docs[:1], 1):  # 최대 1개 문서만 사용
            doc_text = doc['text']
            # 문장 분리 (간단하게 마침표 기준)
            sentences = [s.strip() for s in doc_text.split('.') if s.strip()]
            if not sentences:
                continue
            # 임베딩 모델 재사용
            q_emb = self.embedding_model.encode(question)
            s_embs = self.embedding_model.encode(sentences)
            sims = util.cos_sim(q_emb, s_embs)[0]
            top_indices = sims.argsort(descending=True)[:2]  # 최대 2개 문장만
            key_sents = [sentences[idx] for idx in top_indices]
            context_parts.append(" ".join(key_sents))
        return " ".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        # 개선된 프롬프트 - 더 구체적이고 지시적
        return f"""### 지시사항:
다음 질문에 대해 참고 문서의 내용을 바탕으로 정확하고 구체적으로 답변해주세요.
답변은 질문의 핵심에 집중하고, 참고 문서의 내용을 충실히 반영해야 합니다.
불필요한 정보나 관련 없는 내용은 포함하지 마세요.

### 질문:
{query}

### 참고 문서:
{context}

### 답변:
"""


class SimpleLLMSystem:
    """개선된 규칙 기반 답변 시스템 (LLM 없이)"""
    
    def __init__(self):
        pass
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        문장 추출 기반 답변 생성 (LLM 대신 사용)
        """
        if not context_docs:
            return "죄송합니다. 관련된 문서를 찾을 수 없습니다."
        
        try:
            # 임베딩 모델 로드
            from sentence_transformers import SentenceTransformer, util
            embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            # 질문과 가장 유사한 문장들 추출
            relevant_sentences = []
            
            for doc in context_docs[:2]:  # 최대 2개 문서
                doc_text = doc['text']
                sentences = [s.strip() for s in doc_text.split('.') if s.strip() and len(s.strip()) > 10]
                
                if not sentences:
                    continue
                
                # 질문과 문장들의 유사도 계산
                q_emb = embedding_model.encode(query)
                s_embs = embedding_model.encode(sentences)
                sims = util.cos_sim(q_emb, s_embs)[0]
                
                # 가장 유사한 문장 2개 선택
                top_indices = sims.argsort(descending=True)[:2]
                for idx in top_indices:
                    if sims[idx] > 0.3:  # 유사도 임계값
                        relevant_sentences.append(sentences[idx])
            
            if not relevant_sentences:
                return "죄송합니다. 질문과 관련된 구체적인 정보를 찾을 수 없습니다."
            
            # 답변 구성
            answer = f"질문: {query}\n\n"
            answer += "관련 정보:\n"
            for i, sent in enumerate(relevant_sentences[:3], 1):
                answer += f"{i}. {sent}\n"
            
            return answer
            
        except Exception as e:
            print(f"SimpleLLMSystem 오류: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def _generate_period_answer(self, query: str, docs: List[Dict]) -> str:
        """기간 관련 질문 답변"""
        answer_parts = [f"질문: {query}\n\n"]
        
        for i, doc in enumerate(docs, 1):
            doc_text = doc['text']
            metadata = doc['metadata']
            
            # 기간 관련 정보 추출
            if any(keyword in doc_text for keyword in ['기간', '일', '개월', '년', '주']):
                answer_parts.append(f"📅 문서 {i}에서 찾은 기간 정보:")
                answer_parts.append(f"{doc_text}")
                answer_parts.append(f"출처: {metadata.get('doc_type', '법률 문서')}")
                answer_parts.append("")
        
        if len(answer_parts) == 1:  # 기간 정보가 없는 경우
            answer_parts.append("제공된 문서에서 구체적인 기간 정보를 찾을 수 없습니다.")
            answer_parts.append("더 구체적인 질문을 해주시거나 다른 문서를 검색해보세요.")
        
        return "\n".join(answer_parts)
    
    def _generate_condition_answer(self, query: str, docs: List[Dict]) -> str:
        """조건 관련 질문 답변"""
        answer_parts = [f"질문: {query}\n\n"]
        
        for i, doc in enumerate(docs, 1):
            doc_text = doc['text']
            metadata = doc['metadata']
            
            answer_parts.append(f"📋 문서 {i}에서 찾은 조건:")
            answer_parts.append(f"{doc_text}")
            answer_parts.append(f"출처: {metadata.get('doc_type', '법률 문서')}")
            answer_parts.append("")
        
        return "\n".join(answer_parts)
    
    def _generate_obligation_answer(self, query: str, docs: List[Dict]) -> str:
        """의무/책임 관련 질문 답변"""
        answer_parts = [f"질문: {query}\n\n"]
        
        for i, doc in enumerate(docs, 1):
            doc_text = doc['text']
            metadata = doc['metadata']
            
            answer_parts.append(f"⚖️ 문서 {i}에서 찾은 의무/책임:")
            answer_parts.append(f"{doc_text}")
            answer_parts.append(f"출처: {metadata.get('doc_type', '법률 문서')}")
            answer_parts.append("")
        
        return "\n".join(answer_parts)
    
    def _generate_general_answer(self, query: str, docs: List[Dict]) -> str:
        """일반적인 질문 답변"""
        answer_parts = [f"질문: {query}\n\n"]
        
        for i, doc in enumerate(docs, 1):
            doc_text = doc['text']
            metadata = doc['metadata']
            
            answer_parts.append(f"📄 문서 {i}에서 찾은 정보:")
            answer_parts.append(f"{doc_text}")
            answer_parts.append(f"출처: {metadata.get('doc_type', '법률 문서')}")
            answer_parts.append("")
        
        return "\n".join(answer_parts)


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