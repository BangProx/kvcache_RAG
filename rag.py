from init import model, centroid_index, index_flat, cluster_indexes

# 사용자 쿼리에 대해 먼저 centroid 유사도 검색 후, 해당 클러스터 내에서 재검색하는 함수 정의
def query_system(query, top_k=5):
    """
    query: 사용자의 질의 (문자열)
    top_k: 클러스터 내 검색 시 반환할 상위 문서 수
    """
    # 쿼리 임베딩 생성
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    # (1) 쿼리와 각 centroid 간 L2 거리 계산 후 가장 가까운 클러스터 선택
    _, centroid_idx = centroid_index.search(query_embedding, 1)
    nearest_cluster = int(centroid_idx[0, 0])

    # (2) 해당 클러스터 내에서 top_k 문서 검색
    if nearest_cluster in cluster_indexes:
        index_cluster, doc_idxs = cluster_indexes[nearest_cluster]
        distances, I_local = index_cluster.search(query_embedding, top_k)
        # local 인덱스를 전역 문서 인덱스로 매핑
        global_doc_indices = [doc_idxs[i] for i in I_local[0]]
        return global_doc_indices, distances[0]
    else:
        # 예외 상황: 해당 클러스터가 없으면 전체 문서에서 검색 (fallback)
        distances_all, I_all = index_flat.search(query_embedding, top_k)
        return I_all[0].tolist(), distances_all[0].tolist()


# 8. 테스트: 예시 쿼리 실행
query = input("Enter your query: ")
results, distances = query_system(query)
print("Retrieved document indices:", results)
print("Distances:", distances)