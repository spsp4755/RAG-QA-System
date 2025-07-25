# RAG 기반 로컬 문서 QA 시스템

🔍 **Local 문서 기반의 RAG QA 시스템 구축 및 성능 최적화**

회사 내부 지식 문서, 논문, 오픈 데이터셋을 기반으로 LLM이 답변하는 검색+생성 시스템입니다.

## 🎯 프로젝트 목표

- **RAG 파이프라인 구축**: 문서 검색 및 답변 생성 시스템
- **성능 최적화**: 다양한 embedding 모델, chunk 전략, reranker 비교
- **실험 관리**: 체계적인 실험 로그 및 결과 분석
- **모듈화**: 재사용 가능한 컴포넌트 구조

## 🏗️ 프로젝트 구조

```
RAG_QA_System/
├── data/                   # 데이터셋 및 문서
│   ├── raw/               # 원본 문서
│   ├── processed/         # 전처리된 문서
│   └── embeddings/        # 벡터 DB 저장소
├── src/                   # 소스 코드
│   ├── data_processing/   # 데이터 전처리
│   ├── embedding/         # 임베딩 모델
│   ├── retrieval/         # 검색 시스템
│   ├── generation/        # 답변 생성
│   ├── evaluation/        # 성능 평가
│   └── utils/            # 유틸리티 함수
├── experiments/           # 실험 설정 및 결과
│   ├── configs/          # 실험 설정 파일
│   ├── logs/             # 실험 로그
│   └── results/          # 실험 결과
├── notebooks/            # Jupyter 노트북
├── tests/               # 테스트 코드
├── docs/                # 문서화
└── requirements.txt     # 의존성 패키지
```

## 🚀 주요 기능

### 1. 문서 처리
- 다양한 형식 지원 (PDF, TXT, DOCX, Markdown)
- 청킹 전략 (문단 단위, 고정 크기, 의미 기반)
- 전처리 및 정규화

### 2. 임베딩 시스템
- 다중 임베딩 모델 지원
  - BAAI/bge-small-en
  - intfloat/e5-small
  - ko-sbert
  - kr-simcse
- 벡터 DB (ChromaDB, FAISS)

### 3. 검색 시스템
- 유사도 기반 검색
- Reranker 지원
- 하이브리드 검색 (키워드 + 의미)

### 4. 답변 생성
- 다중 LLM 지원
  - Mistral-7B-Instruct
  - Zephyr-7B
  - Phi-2
  - Llama-2
- 프롬프트 엔지니어링
- 컨텍스트 관리

### 5. 성능 평가
- 정확도 (F1, Exact Match)
- 답변 유사도 (BLEU, ROUGE)
- 사용자 만족도 평가

## 📊 실험 계획

### Phase 1: 기본 RAG 파이프라인
- [ ] 문서 처리 및 청킹
- [ ] 기본 임베딩 시스템 구축
- [ ] 단순 검색 + 생성

### Phase 2: 성능 최적화
- [ ] 임베딩 모델 비교 실험
- [ ] 청킹 전략 비교
- [ ] Reranker 효과 분석

### Phase 3: 고급 기능
- [ ] 멀티모달 지원
- [ ] 실시간 업데이트
- [ ] Agent 기반 확장

## 🛠️ 기술 스택

- **LLM**: Mistral-7B, Zephyr-7B, Phi-2, Llama-2
- **Embedding**: BGE, E5, Ko-SBERT, KR-SimCSE
- **Vector DB**: ChromaDB, FAISS
- **Framework**: LangChain, LlamaIndex
- **UI**: Gradio, Streamlit
- **Monitoring**: MLflow, Weights & Biases

## 📈 성능 지표

- **검색 정확도**: Precision@K, Recall@K
- **답변 품질**: F1 Score, BLEU Score
- **응답 시간**: 검색 시간, 생성 시간
- **사용자 만족도**: 설문 조사 결과

## 🚀 시작하기

```bash
# 1. 저장소 클론
git clone <repository-url>
cd RAG_QA_System

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 기본 설정
python src/setup.py

# 5. 실험 실행
python experiments/run_basic_rag.py
```

## 📝 실험 로그

각 실험의 상세한 로그와 결과는 `experiments/` 디렉토리에서 확인할 수 있습니다.

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 Issue를 생성해주세요. 