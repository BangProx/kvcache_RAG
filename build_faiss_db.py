import pickle
import numpy as np
import pandas as pd
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset = pd.read_csv("/dataset/Natural-Questions-Filtered.csv")
documents = [
    item["document_text"]
    for item in dataset
    if item["document_text"] and item["document_text"].strip() != ""
]
num_documents = len(documents)
print("Number of documents:", num_documents)

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(
    documents, show_progress_bar=True, convert_to_numpy=True
).astype("float32")
dim = doc_embeddings.shape[1]

# 최적의 클러스터 개수를 휴리스틱( silhouette score 기반)으로 결정
candidate_ks = list(range(5, min(50, num_documents // 10 + 1)))
best_k = candidate_ks[0]
best_score = -1
for k in candidate_ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_embeddings)
    if k > 1:
        score = silhouette_score(doc_embeddings, cluster_labels)
        print(f"k={k}, silhouette score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

print("Best k based on silhouette score:", best_k)

# FAISS를 이용해 최적의 클러스터 개수(best_k)로 클러스터링
index_flat = faiss.IndexFlatL2(dim)
clustering = faiss.Clustering(dim, best_k)
clustering.niter = 20  # 클러스터링 반복 횟수
clustering.train(doc_embeddings, index_flat)
centroids = clustering.centroids  # shape: [best_k, dim]

# 각 문서를 가장 가까운 클러스터(centroid)에 할당
centroid_index = faiss.IndexFlatL2(dim)
centroid_index.add(centroids)
_, cluster_assignments = centroid_index.search(doc_embeddings, 1)
cluster_assignments = cluster_assignments.squeeze()

# 클러스터별 문서 인덱스 매핑 생성
clusters = {i: [] for i in range(best_k)}
for doc_idx, cluster_id in enumerate(cluster_assignments):
    clusters[int(cluster_id)].append(doc_idx)

# 각 클러스터별로 별도의 FAISS 인덱스 구축 (클러스터 내 빠른 검색을 위해)
cluster_indexes = {}
for cluster_id, doc_idxs in clusters.items():
    if len(doc_idxs) > 0:
        cluster_embeddings = doc_embeddings[doc_idxs]
        index_cluster = faiss.IndexFlatL2(dim)
        index_cluster.add(cluster_embeddings)
        cluster_indexes[cluster_id] = index_cluster

# FAISS centroid index 저장
faiss.write_index(centroid_index, "centroid_index.index")

# 각 클러스터별 인덱스 저장 (파일 이름에 cluster_id 포함)
for cluster_id, index_cluster in cluster_indexes.items():
    faiss.write_index(index_cluster, f"cluster_index_{cluster_id}.index")

# 클러스터 매핑 정보, 문서 목록, 메타 정보 저장 (pickle 사용)
with open("clusters_mapping.pkl", "wb") as f:
    pickle.dump(clusters, f)

with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open("meta_info.pkl", "wb") as f:
    meta_info = {"best_k": best_k, "dim": dim}
    pickle.dump(meta_info, f)

print("FAISS 인덱스와 관련 데이터가 저장되었습니다.")
