# TO DO
1. hotpotqa 통합
2. FAISS 데이터 저장 파이프라인 구현 - 방병훈
3. generation part 구현 - 방병훈
4. kvcache 추가

# 구현 방향
지금 생각으로는 build_faiss_db.py 파일을 실행하면 데이터셋으로 FAISS 인덱스를 구성하고, 
그 다음부터는 query_faiss_db.py 파일에 kvcache part와 generation part 추가해서 답변 생성하는 방식

# TMI
- 원래는 CAG 레포 pull 해서 그 위에 작업했었는데 좀 복잡해서 따로 제가 레포를 팠습니다.(근데 이것도 복잡;;)
아직 정리가 안되어있어서 정리를 빠르게 하겠슴다;;
- 우선은 NQ 데이터로 테스트하고 hotpotqa랑 squad도 통합 예정
- 우선은 build_faiss_db.py랑 query_faiss_db.py에 코드 구현했다 오류가 있어서 test.ipynb에서 오류 수정중

# 코드 설명
### build_faiss_db.py
Clustering 기반 FAISS 인덱스 구성
pickle로 데이터 저장(클러스터, 데이터 스토어 등)

### query_faiss_db.py
FAISS에서 Query와 유사한 문서 검색 후 답변 생성

# Venv Setting with conda

```bash
conda create -n kvcache python=3.9
conda activate kvcache
```

```bash
pip install -r requirements.txt
```

# .env Template
```txt
OPENAI_API_KEY=""
GOOGLE_API_KEY=""
HF_TOKEN = ""
```

# Reference
https://www.kaggle.com/datasets/frankossai/natural-questions-dataset?select=Natural-Questions-Filtered.csv
https://www.kaggle.com/datasets/jeromeblanchet/hotpotqa-question-answering-dataset
https://github.com/hhhuang/CAG