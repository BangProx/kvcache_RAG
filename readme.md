# virtual Env Setting with conda

```bash
conda create -n kvcache python=3.9
conda activate kvcache
```

```bash
pip install -r requirements.txt
```

# .env Template
OPENAI_API_KEY=""
GOOGLE_API_KEY=""
HF_TOKEN = ""

# TMI
### build_faiss_db.py
Clustering 기반 FAISS 구성
pickle로 데이터 저장(클러스터, 데이터 스토어 등)

### query_faiss_db.py
FAISS에서 Query와 유사한 문서 검색 후 답변 생성

# TO DO
1. hotpotqa 통합
2. FAISS 데이터 저장 파이프라인 구현
3. generation part 구현
4. kvcache 추가

