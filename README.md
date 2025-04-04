# Voronoi KV-Cache RAG

Voronoi cell(클러스터링) 기반 분산 Vector DB와 KV-Cache를 활용한 효율적인 검색 증강 생성(RAG) 시스템입니다.

## 개요

기존의 RAG 시스템은 검색 시간(retrieval time)으로 인한 비효율성이 존재합니다. 매번 검색과 생성을 수행해야 하므로 비용이 증가하는 문제가 있습니다.

Cache-Augmented Generation(CAG) 방식은 미리 데이터들의 KV-Cache 값을 저장해두고 활용함으로써 이 문제를 해결하고자 합니다. 하지만, 대규모 데이터셋에서는 활용이 어렵다는 한계가 있습니다.

이 프로젝트는 **Voronoi cell 기반 클러스터링**을 통해 데이터를 효율적으로 분할하고, 각 클러스터별로 **KV-Cache**를 생성하여 저장함으로써 대규모 데이터셋에서도 CAG 방식을 활용할 수 있도록 합니다.

## 핵심 기능

- **Voronoi 클러스터링**: 문서 임베딩 공간을 Voronoi 셀로 분할하여 유사한 문서들을 효율적으로 그룹화
- **분산 KV-Cache**: 각 클러스터별로 별도의 KV-Cache를 생성하여 대규모 데이터셋 지원
- **직접 KV-Cache 주입**: 프롬프트에 문서를 추가하는 대신 모델에 KV-Cache를 직접 주입하여 효율성 향상
- **클러스터 기반 라우팅**: 쿼리가 입력되면 자동으로 가장 적합한 클러스터와 KV-Cache 선택

## 프로젝트 구조

```
voronoi_kvcache_rag/
├── voronoi_cluster.py    # Voronoi cell 기반 문서 클러스터링
├── kvcache_manager.py    # KV-Cache 생성 및 관리
├── distributed_rag.py    # 분산 RAG 시스템 코어
├── dataset_utils.py      # 데이터셋 준비 유틸리티
├── main.py               # 통합 실행 스크립트
├── dataset/              # 데이터셋 저장 디렉토리
├── voronoi_clusters/     # 클러스터 정보 저장 디렉토리
└── kvcaches/             # KV-Cache 저장 디렉토리
```

## 설치 방법

### 필수 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)

### 패키지 설치

```bash
pip install -r requirements.txt
```

### 환경 변수 설정

```bash
# HuggingFace 토큰 설정 (.env 파일 생성)
echo "HF_TOKEN=your_huggingface_token" > .env
```

## 사용 방법

### 1. 데이터셋 준비

Hugging Face 데이터셋에서 샘플을 추출하여 CSV로 저장:

```bash
python main.py prepare --dataset natural_questions --size 1000
```

### 2. 문서 클러스터링

문서를 임베딩하고 Voronoi 클러스터링을 수행:

```bash
python main.py cluster --dataset dataset/natural_questions_validation_sample.csv
```

자동 클러스터 개수 결정을 비활성화하려면:

```bash
python main.py cluster --dataset dataset/natural_questions_validation_sample.csv --no_auto_k --k 20
```

### 3. KV-Cache 생성

각 클러스터별 KV-Cache를 생성:

```bash
python main.py cache
```

기존 KV-Cache를 강제로 재생성하려면:

```bash
python main.py cache --force
```

### 4. 쿼리 실행

쿼리 실행:

```bash
python main.py query --text "What is machine learning?"
```

## 설명

### Voronoi Cell 기반 클러스터링

1. 문서를 임베딩한 후 silhouette score를 사용하여 최적의 클러스터 개수를 결정합니다.
2. FAISS 클러스터링을 통해 문서를 Voronoi cell로 분할합니다.
3. 각 문서는 가장 가까운 centroid에 할당됩니다.
4. 클러스터 간 상이성과 클러스터 내 유사성을 최적화합니다.

### KV-Cache 생성 및 관리

1. 각 클러스터별로 포함된 문서들을 통합한 프롬프트를 생성합니다.
2. 언어 모델로 프롬프트를 처리하여 KV-Cache를 생성합니다.
3. 각 클러스터의 KV-Cache를 파일로 저장하여 효율적으로 관리합니다.
4. 필요한 경우에만 KV-Cache를 메모리에 로드하여 리소스를 최적화합니다.

### 쿼리 프로세스

1. 쿼리 텍스트를 임베딩하여 가장 가까운 Voronoi cell(클러스터)을 찾습니다.
2. 해당 클러스터의 KV-Cache를 로드합니다.
3. 쿼리 프롬프트에 KV-Cache를 직접 주입하여 응답을 생성합니다.