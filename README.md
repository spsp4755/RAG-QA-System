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
│       ├── 📁 complete_db/        # 다국어 임베딩 벡터 DB
│       ├── 📁 korean_sbert_db/    # 한국어 SBERT 벡터 DB
│       └── 📁 training_db/        # 학습용 벡터 DB
├── 📁 src/
│   ├── 📁 embedding/     # 임베딩 모델 및 벡터 DB 구축
│   ├── 📁 rag/          # RAG 시스템 핵심 로직
│   └── 📁 evaluation/   # 성능 평가 도구
├── 📁 experiments/      # 실험 결과 및 로그
└── 📁 tests/           # 테스트 코드
```

## 🔧 기술 스택

### 임베딩 모델 (실험 대상)
1. **다국어 모델**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
2. **한국어 특화 모델**: `jhgan/ko-sroberta-multitask`

### 벡터 DB
- **ChromaDB**: 지속적 저장소 기반 벡터 데이터베이스
- **언어**: 한국어 다국어 지원

### LLM 모델 (실험 대상)
1. `beomi/gemma-ko-2b`
2. `EleutherAI/polyglot-ko-1.3b`
3. `skt/kogpt2-base-v2`

## 📊 실험 결과

### 평가 환경
- **평가 데이터**: Validation 데이터 20개 샘플
- **평가 지표**: BLEU, ROUGE-1/2/L, METEOR, 종합 점수
- **평가 방식**: 동일한 LLM 모델로 다른 벡터DB 비교

## 🔍 벡터DB 성능 비교 실험

### 실험 1: 다국어 임베딩 모델 (complete_db)

#### 모델별 성능 비교

| 모델 | 성공률 | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | 종합점수 | 생성시간 |
|------|--------|------|---------|---------|---------|--------|----------|----------|
| **beomi/gemma-ko-2b** | 100% | 0.0052 | 0.1393 | 0.0250 | 0.1393 | 0.0224 | **0.0662** | 148.0초 |
| **EleutherAI/polyglot-ko-1.3b** | 100% | 0.0033 | 0.0543 | 0.0083 | 0.0543 | 0.0288 | 0.0298 | 45.2초 |
| **skt/kogpt2-base-v2** | 100% | 0.0026 | 0.0366 | 0.0000 | 0.0366 | 0.0297 | 0.0211 | 12.3초 |

### 실험 2: 한국어 SBERT 임베딩 모델 (korean_sbert_db)

#### 모델별 성능 비교

| 모델 | 성공률 | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | 종합점수 | 생성시간 |
|------|--------|------|---------|---------|---------|--------|----------|----------|
| **beomi/gemma-ko-2b** | 100% | 0.0036 | 0.0214 | 0.0167 | 0.0214 | 0.0227 | **0.0172** | 114.0초 |
| **EleutherAI/polyglot-ko-1.3b** | 100% | 0.0017 | 0.0125 | 0.0000 | 0.0125 | 0.0104 | 0.0074 | 41.0초 |
| **skt/kogpt2-base-v2** | 100% | 0.0022 | 0.0254 | 0.0000 | 0.0254 | 0.0207 | 0.0147 | 11.7초 |

## 📈 벡터DB 비교 분석

### 성능 변화 요약

| 모델 | 벡터DB | 종합점수 | ROUGE-1 | 생성시간 | 성능 변화 |
|------|--------|----------|---------|----------|-----------|
| **beomi/gemma-ko-2b** | complete_db | 0.0662 | 0.1393 | 148.0초 | **기준** |
| **beomi/gemma-ko-2b** | korean_sbert_db | 0.0172 | 0.0214 | 114.0초 | **74% 감소** |
| **EleutherAI/polyglot-ko-1.3b** | complete_db | 0.0298 | 0.0543 | 45.2초 | **기준** |
| **EleutherAI/polyglot-ko-1.3b** | korean_sbert_db | 0.0074 | 0.0125 | 41.0초 | **75% 감소** |
| **skt/kogpt2-base-v2** | complete_db | 0.0211 | 0.0366 | 12.3초 | **기준** |
| **skt/kogpt2-base-v2** | korean_sbert_db | 0.0147 | 0.0254 | 11.7초 | **30% 감소** |

### 주요 발견사항

#### ✅ **긍정적 측면**
- **생성 속도 개선**: 모든 모델에서 생성 시간이 단축됨 (5-23% 개선)
- **ROUGE-2 점수**: `beomi/gemma-ko-2b`에서 상대적으로 덜 감소

#### ❌ **부정적 측면**
- **검색 품질 저하**: ROUGE-1 점수가 크게 감소 (77-85% 감소)
- **전체 성능 저하**: 모든 모델에서 종합 점수가 감소

### 성능 저하 원인 분석

1. **임베딩 모델 호환성 문제**:
   - `jhgan/ko-sroberta-multitask`가 법률 도메인에 최적화되지 않음
   - 다국어 모델이 오히려 더 범용적인 검색 성능 제공

2. **도메인 특화 문제**:
   - 한국어 SBERT가 일반 한국어에 특화되어 법률 전문 용어 처리 능력 부족

3. **벡터DB 구성 차이**:
   - 한국어 SBERT DB의 문서 구성이 검색 품질에 부정적 영향

## 🏆 최종 권장사항

### 🥇 **현재 최적 설정**
- **벡터DB**: `complete_db` (다국어 임베딩 모델)
- **LLM 모델**: `beomi/gemma-ko-2b` (최고 성능)
- **종합 점수**: 0.0662

### 🎯 **개선 방향**
1. **법률 도메인 특화 임베딩 모델 탐색**
2. **벡터DB 구성 최적화**
3. **하이브리드 검색 시스템 구현**

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
# 다국어 임베딩 벡터 DB 구축 (권장)
python src/embedding/build_complete_vector_db.py

# 한국어 SBERT 벡터 DB 구축 (실험용)
python src/embedding/build_korean_sbert_vector_db.py
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
python run_evaluation.py
```

## 📁 주요 파일 구조

```
📁 src/
├── 📄 embedding/
│   ├── build_complete_vector_db.py        # 다국어 임베딩 벡터 DB 구축
│   ├── build_korean_sbert_vector_db.py    # 한국어 SBERT 벡터 DB 구축
│   └── build_training_vector_db.py        # 학습용 벡터 DB 구축
├── 📄 rag/
│   ├── rag_qa_system.py                  # RAG 시스템 메인
│   └── context_retriever.py              # 컨텍스트 검색기
└── 📄 evaluation/
    ├── rag_evaluator.py                  # 성능 평가기
    └── korean_sbert_rag_evaluator.py     # 한국어 SBERT 평가기

📁 experiments/
├── 📄 evaluation_results_*.json          # 실험 결과
└── 📄 logs/                              # 실험 로그

📁 data/
├── 📄 embeddings/complete_db/            # 다국어 임베딩 ChromaDB
├── 📄 embeddings/korean_sbert_db/        # 한국어 SBERT ChromaDB
└── 📄 processed/knowledge_qa.json        # 전처리된 데이터
```

## 🔍 주요 기능

### 1. 벡터 검색
- 의미적 유사도 기반 문서 검색
- 다국어/한국어 특화 임베딩 모델 지원
- 실시간 검색 및 랭킹

### 2. 답변 생성
- 검색된 컨텍스트 기반 답변 생성
- 다양한 LLM 모델 지원
- 프롬프트 엔지니어링 최적화

### 3. 성능 평가
- 표준 NLP 평가 지표 사용
- BLEU, ROUGE, METEOR 점수 계산
- 벡터DB 간 성능 비교 분석

## 📈 향후 개선 방향

### 1. 임베딩 모델 개선
- 법률 도메인 특화 임베딩 모델 탐색
- 고성능 다국어 모델 테스트 (`paraphrase-multilingual-mpnet-base-v2`)
- 법률 문서로 fine-tuning된 모델 개발

### 2. 벡터DB 최적화
- 문서 청킹 방식 개선
- 컨텍스트 vs Q&A 쌍 비율 조정
- 메타데이터 활용 최적화

### 3. 하이브리드 시스템
- 다중 벡터DB 조합 검색
- 앙상블 기반 결과 선택
- 가중치 기반 결합 시스템

### 4. 평가 체계 고도화
- 더 정교한 평가 지표 도입
- 인간 평가와의 상관관계 분석
- 도메인별 성능 분석
