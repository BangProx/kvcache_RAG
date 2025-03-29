import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 저장된 파일들 불러오기
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("clusters_mapping.pkl", "rb") as f:
    clusters = pickle.load(f)

with open("meta_info.pkl", "rb") as f:
    meta_info = pickle.load(f)
best_k = meta_info["best_k"]
dim = meta_info["dim"]

# Centroid 인덱스 로드
centroid_index = faiss.read_index("centroid_index.index")

# 각 클러스터별 인덱스 로드
cluster_indexes = {}
for cluster_id in range(best_k):
    try:
        index_cluster = faiss.read_index(f"cluster_index_{cluster_id}.index")
        cluster_indexes[cluster_id] = index_cluster
    except Exception as e:
        print(f"Cluster {cluster_id} 인덱스 로드 중 오류 발생: {e}")

# SentenceTransformer 모델 로드 (쿼리 임베딩용)
model = SentenceTransformer('all-MiniLM-L6-v2')


def query_system(query, top_k=5):
    """
    query: 사용자 질의 (문자열)
    top_k: 해당 클러스터 내에서 검색할 상위 문서 수
    """
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')

    # (1) 쿼리와 centroid 간 유사도 검색
    _, centroid_idx = centroid_index.search(query_embedding, 1)
    nearest_cluster = int(centroid_idx[0, 0])

    # (2) 해당 클러스터 내에서 top_k 문서 검색
    if nearest_cluster in cluster_indexes:
        index_cluster = cluster_indexes[nearest_cluster]
        distances, I_local = index_cluster.search(query_embedding, top_k)
        # clusters 매핑을 통해 전역 문서 인덱스로 변환
        cluster_doc_idxs = clusters[nearest_cluster]
        global_doc_indices = [cluster_doc_idxs[i] for i in I_local[0]]
        return global_doc_indices, distances[0]
    else:
        print("해당 클러스터를 찾을 수 없습니다. 전체 문서 검색 필요.")
        return [], []


# 예시: 질의 실행
query = "What is the capital of France?"
results, distances = query_system(query)
print("쿼리:", query)
print("검색된 문서 인덱스 및 거리:")
for idx, d in zip(results, distances):
    print(f"문서 인덱스: {idx}, L2 거리: {d:.4f}")
    print("문서 일부:", documents[idx][:200])
    print("------")