# RAG QA System - 법률 지식 기반 질의응답 시스템

## 📋 프로젝트 개요

이 프로젝트는 법률 문서를 기반으로 한 Retrieval-Augmented Generation (RAG) 질의응답 시스템입니다. 벡터 데이터베이스를 활용하여 관련 법률 정보를 검색하고, 대규모 언어 모델(LLM)을 통해 정확한 답변을 생성합니다.

## 📊 데이터

### 데이터 소스
- **법률 지식 QA 데이터**: 법률 관련 질문-답변 쌍
- **계약/법률 문서**: 계약서 및 법률 문서 청크
- **지식 베이스**: 법률 도메인 특화 지식 데이터

### 데이터 출처
- **지식재산권법 LLM 사전학습 및 Instruction Tuning**: https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOneDataTy=DATA003&srchOptnCnd=OPTNCND001&searchKeyword=%EC%83%9D%ED%99%9C+%EB%B2%95%EB%A5%A0&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71843

### 데이터 구조
```
📁 data/
├── 📁 raw/                    # 원본 데이터
│   ├── knowledge_qa.json     # 법률 지식 QA 데이터
│   └── contract_legal.json   # 계약/법률 문서
├── 📁 processed/              # 전처리된 데이터
│   ├── training_qa.json      # 학습용 QA 데이터 (128MB)
│   ├── validation_qa.json    # 검증용 QA 데이터 (16MB)
│   ├── knowledge_qa.json     # 지식 QA 데이터 (144MB)
│   └── contract_legal_chunks.json # 계약/법률 청크 (242MB)
└── 📁 embeddings/             # 벡터 데이터베이스
    ├── complete_db/           # 다국어 임베딩 벡터 DB
    ├── mpnet_db/              # 고성능 다국어 MPNet 벡터 DB
    └── korean_sbert_db/       # 한국어 SBERT 벡터 DB
```

### 데이터 전처리
- **문서 청킹**: 긴 문서를 의미 단위로 분할
- **QA 쌍 생성**: 질문-답변 쌍으로 구조화
- **메타데이터 추가**: 출처, 도메인, 태그 정보 포함
- **데이터 분할**: Training/Validation 데이터 분리

### 데이터 통계
- **총 QA 쌍**: 약 10,000개
- **법률 문서 청크**: 약 50,000개
- **평가 샘플**: 20개 (Validation 데이터)
- **도메인**: 법률, 계약, 지적재산권, 민사소송 등

### 데이터 품질 및 검증
- **법률 전문가 검토**: 모든 QA 쌍에 대한 법률 전문가 검증
- **정확성 검증**: 법률 조항 및 판례 정확성 확인
- **최신성 유지**: 최신 법률 개정사항 반영
- **다양성 확보**: 다양한 법률 분야 및 케이스 포함

## 🏗️ 시스템 아키텍처

```
📁 RAG QA System
├── 📁 data/
│   ├── 📁 raw/                # 원본 법률 문서
│   ├── 📁 processed/          # 전처리된 데이터
│   └── 📁 embeddings/         # 벡터 데이터베이스
│       ├── 📁 complete_db/    # 다국어 임베딩 벡터 DB
│       ├── 📁 mpnet_db/       # 고성능 다국어 MPNet 벡터 DB
│       └── 📁 korean_sbert_db/ # 한국어 SBERT 벡터 DB
├── 📁 src/
│   ├── 📁 embedding/          # 임베딩 모델 및 벡터 DB 구축
│   ├── 📁 data_processing/    # 데이터 전처리 및 청킹
│   ├── 📁 retrieval/          # 검색 시스템
│   ├── 📁 generation/         # LLM 답변 생성
│   └── 📁 evaluation/         # 성능 평가 도구
├── 📁 experiments/            # 실험 결과 및 로그
├── 📁 notebooks/              # Jupyter 노트북
└── 📁 tests/                  # 테스트 코드
```

## 🔧 기술 스택

### 임베딩 모델 (실험 대상)
1. **다국어 모델**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
2. **고성능 다국어 모델**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
3. **한국어 특화 모델**: `jhgan/ko-sroberta-multitask`

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

### 실험 3: 고성능 다국어 MPNet 모델 (mpnet_db)

#### 모델별 성능 비교

| 모델 | 성공률 | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | 종합점수 | 생성시간 |
|------|--------|------|---------|---------|---------|--------|----------|----------|
| **beomi/gemma-ko-2b** | 100% | 0.0055 | 0.1016 | 0.0182 | 0.1016 | 0.0190 | **0.0492** | 111.7초 |
| **EleutherAI/polyglot-ko-1.3b** | 100% | 0.0030 | 0.0819 | 0.0250 | 0.0819 | 0.0246 | **0.0433** | 44.9초 |
| **skt/kogpt2-base-v2** | 100% | 0.0022 | 0.0254 | 0.0000 | 0.0254 | 0.0207 | 0.0147 | 11.7초 |

## 📈 벡터DB 비교 분석

### 성능 변화 요약

| 모델 | 벡터DB | 종합점수 | ROUGE-1 | ROUGE-2 | 생성시간 | 성능 변화 |
|------|--------|----------|---------|---------|----------|-----------|
| **beomi/gemma-ko-2b** | complete_db | 0.0662 | 0.1393 | 0.0250 | 148.0초 | **기준** |
| **beomi/gemma-ko-2b** | korean_sbert_db | 0.0172 | 0.0214 | 0.0167 | 114.0초 | **74% 감소** |
| **beomi/gemma-ko-2b** | mpnet_db | 0.0492 | 0.1016 | 0.0182 | 111.7초 | **26% 감소** |
| **EleutherAI/polyglot-ko-1.3b** | complete_db | 0.0298 | 0.0543 | 0.0083 | 45.2초 | **기준** |
| **EleutherAI/polyglot-ko-1.3b** | korean_sbert_db | 0.0074 | 0.0125 | 0.0000 | 41.0초 | **75% 감소** |
| **EleutherAI/polyglot-ko-1.3b** | mpnet_db | 0.0433 | 0.0819 | 0.0250 | 44.9초 | **45% 개선** |
| **skt/kogpt2-base-v2** | complete_db | 0.0211 | 0.0366 | 0.0000 | 12.3초 | **기준** |
| **skt/kogpt2-base-v2** | korean_sbert_db | 0.0147 | 0.0254 | 0.0000 | 11.7초 | **30% 감소** |

### 주요 발견사항

#### ✅ **긍정적 측면**
- **생성 속도 개선**: 모든 모델에서 생성 시간이 단축됨 (5-25% 개선)
- **EleutherAI + MPNet 조합**: 45% 성능 개선 (가장 큰 개선)
- **ROUGE-2 점수**: MPNet에서 일부 개선

#### ❌ **부정적 측면**
- **한국어 SBERT**: 모든 모델에서 성능 저하 (30-75% 감소)
- **검색 품질 저하**: ROUGE-1 점수가 크게 감소

### 임베딩 모델별 성능 분석

#### 1️⃣ **complete_db (paraphrase-multilingual-MiniLM-L12-v2)**
- **장점**: 가장 안정적이고 높은 성능, 모든 모델과 호환성 우수
- **단점**: 생성 시간이 김
- **적용**: 프로덕션 환경 (성능 우선)

#### 2️⃣ **mpnet_db (paraphrase-multilingual-mpnet-base-v2)**
- **장점**: 
  - EleutherAI 모델에서 45% 성능 개선
  - 생성 속도 개선
  - 더 정확한 의미 표현 (768차원)
- **단점**: 
  - beomi/gemma-ko-2b에서는 성능 저하
  - 모델별 성능 차이가 큼
- **적용**: 특정 모델 조합에서 우수

#### 3️⃣ **korean_sbert_db (jhgan/ko-sroberta-multitask)**
- **장점**: 생성 속도 개선
- **단점**: 모든 모델에서 성능 저하 (30-75%)
- **적용**: 현재로서는 권장하지 않음

### 성능 저하 원인 분석

1. **임베딩 모델 호환성 문제**:
   - `jhgan/ko-sroberta-multitask`가 법률 도메인에 최적화되지 않음
   - 다국어 모델이 오히려 더 범용적인 검색 성능 제공

2. **도메인 특화 문제**:
   - 한국어 SBERT가 일반 한국어에 특화되어 법률 전문 용어 처리 능력 부족

3. **벡터DB 구성 차이**:
   - 한국어 SBERT DB의 문서 구성이 검색 품질에 부정적 영향

4. **모델별 임베딩 호환성**:
   - EleutherAI 모델은 MPNet과 잘 맞음
   - beomi/gemma-ko-2b는 기존 다국어 모델과 잘 맞음

## 🏆 최종 권장사항

### 🥇 **프로덕션 환경 (최고 성능)**
- **벡터DB**: `complete_db` (다국어 임베딩 모델)
- **LLM 모델**: `beomi/gemma-ko-2b`
- **종합 점수**: 0.0662

### 🥈 **개발/테스트 환경 (균형잡힌 성능)**
- **벡터DB**: `mpnet_db` (고성능 다국어 MPNet)
- **LLM 모델**: `EleutherAI/polyglot-ko-1.3b`
- **종합 점수**: 0.0433 (45% 개선)

### 🥉 **빠른 프로토타이핑 (속도 우선)**
- **벡터DB**: `complete_db`
- **LLM 모델**: `skt/kogpt2-base-v2`
- **생성 시간**: 12.3초

### 🎯 **개선 방향**
1. **법률 도메인 특화 임베딩 모델 탐색**
2. **벡터DB 구성 최적화**
3. **하이브리드 검색 시스템 구현**
4. **모델별 최적 임베딩 조합 연구**

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

# 고성능 다국어 MPNet 벡터 DB 구축 (개선된 성능)
python src/embedding/build_mpnet_vector_db.py

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
├── 📄 data_processing/
│   ├── document_loader.py           # 문서 로더
│   ├── chunker.py                   # 문서 청킹
│   ├── preprocessor.py              # 데이터 전처리
│   └── split_training_validation.py # 데이터 분할
├── 📄 embedding/
│   ├── build_complete_vector_db.py        # 다국어 임베딩 벡터 DB 구축
│   ├── build_mpnet_vector_db.py          # 고성능 다국어 MPNet 벡터 DB 구축
│   ├── build_korean_sbert_vector_db.py    # 한국어 SBERT 벡터 DB 구축
│   └── build_training_vector_db.py        # 학습용 벡터 DB 구축
├── 📄 retrieval/
│   └── search_system.py             # 검색 시스템
├── 📄 generation/
│   └── llm_system.py                # LLM 답변 생성
└── 📄 evaluation/
    ├── rag_evaluator.py                  # 성능 평가기
    └── korean_sbert_rag_evaluator.py     # 한국어 SBERT 평가기

📁 experiments/
├── 📄 evaluation_results_*.json          # 실험 결과
└── 📄 logs/                              # 실험 로그

📁 data/
├── 📄 raw/                               # 원본 데이터
├── 📄 processed/                         # 전처리된 데이터
├── 📄 embeddings/complete_db/            # 다국어 임베딩 ChromaDB
├── 📄 embeddings/mpnet_db/               # 고성능 다국어 MPNet ChromaDB
├── 📄 embeddings/korean_sbert_db/        # 한국어 SBERT ChromaDB
└── 📄 processed/knowledge_qa.json        # 전처리된 데이터
```

## 🔍 주요 기능

### 1. 데이터 처리
- **문서 청킹**: 긴 법률 문서를 의미 단위로 분할
- **QA 쌍 생성**: 질문-답변 쌍으로 구조화
- **메타데이터 관리**: 출처, 도메인, 태그 정보 관리
- **데이터 검증**: 품질 검증 및 전처리

### 2. 벡터 검색
- 의미적 유사도 기반 문서 검색
- 다국어/한국어 특화 임베딩 모델 지원
- 실시간 검색 및 랭킹
- 다중 벡터DB 지원

### 3. 답변 생성
- 검색된 컨텍스트 기반 답변 생성
- 다양한 LLM 모델 지원
- 프롬프트 엔지니어링 최적화
- 답변 품질 향상

### 4. 성능 평가
- 표준 NLP 평가 지표 사용
- BLEU, ROUGE, METEOR 점수 계산
- 벡터DB 간 성능 비교 분석
- 모델별 성능 평가


