import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset = load_dataset("natural_questions", split="train[:1000]")
documents = [
    item["document_text"]
    for item in dataset
    if item["document_text"] and item["document_text"].strip() != ""
]
num_documents = len(documents)
print("Number of documents:", num_documents)

model = SentenceTransformer("all-MiniLM-L6-v2")

# 모델은 한꺼번에 임베딩을 생성하며, numpy 배열(float32)로 반환합니다.
doc_embeddings = model.encode(
    documents, show_progress_bar=True, convert_to_numpy=True
).astype("float32")
dim = doc_embeddings.shape[1]

# 후보 클러스터 수: 데이터 개수와 상황에 따라 범위를 조정합니다.
candidate_ks = list(range(5, min(50, num_documents // 10 + 1)))
best_k = candidate_ks[0]
best_score = -1
for k in candidate_ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_embeddings)
    # k==1에서는 silhouette score를 계산할 수 없으므로 k>=2인 경우에만 계산
    if k > 1:
        score = silhouette_score(doc_embeddings, cluster_labels)
        print(f"k={k}, silhouette score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

print("Best k based on silhouette score:", best_k)

index_flat = faiss.IndexFlatL2(dim)
clustering = faiss.Clustering(dim, best_k)
clustering.niter = 20  # 클러스터링 반복 횟수 (필요에 따라 조정)
clustering.train(doc_embeddings, index_flat)
# 클러스터 중심(centroids) 추출
centroids = clustering.centroids

centroid_index = faiss.IndexFlatL2(dim)
centroid_index.add(centroids)
_, cluster_assignments = centroid_index.search(doc_embeddings, 1)
cluster_assignments = cluster_assignments.squeeze()

# 클러스터별 문서 인덱스 매핑 생성
clusters = {i: [] for i in range(best_k)}
for doc_idx, cluster_id in enumerate(cluster_assignments):
    clusters[int(cluster_id)].append(doc_idx)

cluster_indexes = {}
for cluster_id, doc_idxs in clusters.items():
    if len(doc_idxs) > 0:
        cluster_embeddings = doc_embeddings[doc_idxs]
        index_cluster = faiss.IndexFlatL2(dim)
        index_cluster.add(cluster_embeddings)
        cluster_indexes[cluster_id] = (index_cluster, doc_idxs)
