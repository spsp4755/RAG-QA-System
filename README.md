# RAG QA System - 법률 지식 기반 질의응답 시스템

## 📋 프로젝트 개요

이 프로젝트는 법률 문서를 기반으로 한 Retrieval-Augmented Generation (RAG) 질의응답 시스템입니다. 벡터 데이터베이스를 활용하여 관련 법률 정보를 검색하고, 대규모 언어 모델(LLM)을 통해 정확한 답변을 생성합니다.

## 🏗️ 시스템 아키텍처

```
📁 RAG QA System
├── 📁 data/
│   ├── 📁 raw/           # 원본 법률 문서
│   ├── 📁 processed/     # 전처리된 데이터
│   └── 📁 embeddings/    # 벡터 데이터베이스
├── 📁 src/
│   ├── 📁 embedding/     # 임베딩 모델 및 벡터 DB 구축
│   ├── 📁 rag/          # RAG 시스템 핵심 로직
│   └── 📁 evaluation/   # 성능 평가 도구
├── 📁 experiments/      # 실험 결과 및 로그
└── 📁 tests/           # 테스트 코드
```

## 🔧 기술 스택

### 임베딩 모델
- **모델**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **벡터 DB**: ChromaDB
- **언어**: 한국어 다국어 지원

### LLM 모델 (실험 대상)
1. `beomi/gemma-ko-2b`
2. `EleutherAI/polyglot-ko-1.3b`
3. `skt/kogpt2-base-v2`

## 📊 실험 결과

### 평가 환경
- **평가 데이터**: Validation 데이터 20개 샘플
- **평가 지표**: BLEU, ROUGE-1/2/L, METEOR, 종합 점수
- **공통 임베딩 모델**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### 모델별 성능 비교

| 모델 | 성공률 | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | 종합점수 | 생성시간 |
|------|--------|------|---------|---------|---------|--------|----------|----------|
| **beomi/gemma-ko-2b** | 100% | 0.0052 | 0.1393 | 0.0250 | 0.1393 | 0.0224 | **0.0662** | 148.0초 |
| **EleutherAI/polyglot-ko-1.3b** | 100% | 0.0033 | 0.0543 | 0.0083 | 0.0543 | 0.0288 | 0.0298 | 45.2초 |
| **skt/kogpt2-base-v2** | 100% | 0.0026 | 0.0366 | 0.0000 | 0.0366 | 0.0297 | 0.0211 | 12.3초 |

### 성능 분석

#### 🥇 최고 성능: beomi/gemma-ko-2b
- **장점**: 가장 높은 종합 점수, ROUGE 점수 우수
- **단점**: 생성 시간이 매우 김 (148초)
- **적용**: 프로덕션 환경 (성능 우선)

#### 🥈 균형잡힌 성능: EleutherAI/polyglot-ko-1.3b
- **장점**: 적절한 성능과 속도 균형
- **단점**: 중간 수준의 성능
- **적용**: 개발/테스트 환경

#### 🥉 빠른 속도: skt/kogpt2-base-v2
- **장점**: 가장 빠른 생성 속도 (12.3초)
- **단점**: 가장 낮은 성능
- **적용**: 빠른 프로토타이핑

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# 벡터 데이터베이스 구축
python src/embedding/build_complete_vector_db.py
```

### 3. 시스템 실행
```bash
# RAG QA 시스템 실행
python src/rag/rag_qa_system.py

# 웹 인터페이스 실행
streamlit run streamlit_app.py
```

### 4. 성능 평가
```bash
# 시스템 성능 평가
python src/evaluation/rag_evaluator.py
```

## 📁 주요 파일 구조

```
📁 src/
├── 📄 embedding/
│   ├── build_complete_vector_db.py    # 벡터 DB 구축
│   └── build_training_vector_db.py    # 학습용 벡터 DB 구축
├── 📄 rag/
│   ├── rag_qa_system.py              # RAG 시스템 메인
│   └── context_retriever.py          # 컨텍스트 검색기
└── 📄 evaluation/
    └── rag_evaluator.py              # 성능 평가기

📁 experiments/
├── 📄 evaluation_results_*.json      # 실험 결과
└── 📄 logs/                          # 실험 로그

📁 data/
├── 📄 embeddings/complete_db/        # ChromaDB 저장소
└── 📄 processed/knowledge_qa.json    # 전처리된 데이터
```

## 🔍 주요 기능

### 1. 벡터 검색
- 의미적 유사도 기반 문서 검색
- 다국어 지원 임베딩 모델
- 실시간 검색 및 랭킹

### 2. 답변 생성
- 검색된 컨텍스트 기반 답변 생성
- 다양한 LLM 모델 지원
- 프롬프트 엔지니어링 최적화

### 3. 성능 평가
- 표준 NLP 평가 지표 사용
- BLEU, ROUGE, METEOR 점수 계산
- 상세한 실험 결과 분석

## 📈 개선 방향

### 1. 모델 개선
- 더 큰 한국어 특화 모델 테스트
- 프롬프트 엔지니어링 강화
- 답변 후처리 로직 추가

### 2. 시스템 최적화
- 검색 품질 향상
- 생성 속도 개선
- 메모리 사용량 최적화

### 3. 평가 체계
- 더 정교한 평가 지표 도입
- 인간 평가와의 상관관계 분석
- 도메인별 성능 분석
