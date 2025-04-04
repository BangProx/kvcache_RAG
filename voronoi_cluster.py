import os
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any, Optional

class VoronoiCluster:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        
        self.documents = []
        self.document_ids = []
        self.doc_embeddings = None
        self.dim = 0
        self.best_k = 0
        self.clusters = {}
        self.centroid_index = None
    
    def load_embedding_model(self):
        if self.embedding_model is None:
            print(f"임베딩 모델 로드: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        model = self.load_embedding_model()
        embeddings = model.encode(documents, show_progress_bar=True, 
                                 batch_size=batch_size, convert_to_numpy=True)
        return embeddings.astype("float32")
    
    def fit(self, documents: List[str], document_ids: Optional[List[Any]] = None, 
           find_optimal_k: bool = True, k: int = 10, batch_size: int = 32) -> Dict:
        print(f"총 {len(documents)} 문서 처리")
        self.documents = documents
        self.document_ids = document_ids if document_ids is not None else list(range(len(documents)))
        
        print("문서 임베딩 생성 중...")
        self.doc_embeddings = self.encode_documents(documents, batch_size)
        self.dim = self.doc_embeddings.shape[1]
        
        if find_optimal_k:
            print("최적 클러스터 개수 결정 중...")
            self.best_k = self._find_optimal_clusters()
        else:
            self.best_k = k
        
        print(f"Voronoi 클러스터링 수행 중 (k={self.best_k})...")
        self._perform_clustering()
        
        return {
            "num_documents": len(self.documents),
            "num_clusters": self.best_k,
            "dimension": self.dim,
            "clusters": {k: len(v) for k, v in self.clusters.items()}
        }
    
    def _find_optimal_clusters(self, min_k: int = 5, max_k: int = 50, 
                             step: int = 5) -> int:
        num_documents = len(self.documents)
        max_k = min(max_k, num_documents // 10)
        candidate_ks = list(range(min_k, max_k + 1, step))
        
        best_k = candidate_ks[0]
        best_score = -1
        
        for k in tqdm(candidate_ks):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.doc_embeddings)
            
            if k > 1:  
                score = silhouette_score(self.doc_embeddings, cluster_labels)
                print(f"k={k}, 실루엣 점수: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
        
        print(f"최적 클러스터 개수: {best_k} (실루엣 점수: {best_score:.4f})")
        return best_k
    
    def _perform_clustering(self):
        index_flat = faiss.IndexFlatL2(self.dim)
        clustering = faiss.Clustering(self.dim, self.best_k)
        clustering.niter = 20
        clustering.train(self.doc_embeddings, index_flat)
        
        centroids = faiss.vector_float_to_array(clustering.centroids)
        centroids = centroids.reshape(self.best_k, self.dim)
        
        self.centroid_index = faiss.IndexFlatL2(self.dim)
        self.centroid_index.add(centroids)
        
        _, cluster_ids = self.centroid_index.search(self.doc_embeddings, 1)
        cluster_ids = cluster_ids.flatten()
        
        self.clusters = {i: [] for i in range(self.best_k)}
        for idx, cluster_id in enumerate(cluster_ids):
            self.clusters[int(cluster_id)].append(idx)
        
        cluster_sizes = {k: len(v) for k, v in self.clusters.items()}
        print(f"클러스터 크기: 최소={min(cluster_sizes.values())}, 최대={max(cluster_sizes.values())}, 평균={sum(cluster_sizes.values())/len(cluster_sizes):.1f}")
    
    def predict_cluster(self, query: str) -> int:
        if self.centroid_index is None:
            raise ValueError("클러스터링이 수행되지 않았습니다.")
        
        query_embedding = self.encode_documents([query])
        
        _, cluster_ids = self.centroid_index.search(query_embedding, 1)
        return int(cluster_ids[0, 0])
    
    def get_cluster_documents(self, cluster_id: int) -> List[str]:
        if cluster_id not in self.clusters:
            raise ValueError(f"클러스터 ID {cluster_id}가 존재하지 않습니다.")
        
        doc_indices = self.clusters[cluster_id]
        return [self.documents[idx] for idx in doc_indices]
    
    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(output_dir, "document_ids.pkl"), "wb") as f:
            pickle.dump(self.document_ids, f)
        
        with open(os.path.join(output_dir, "clusters.pkl"), "wb") as f:
            pickle.dump(self.clusters, f)
        
        meta_info = {
            "best_k": self.best_k,
            "dim": self.dim,
            "embedding_model": self.embedding_model_name,
            "num_documents": len(self.documents)
        }
        with open(os.path.join(output_dir, "meta_info.pkl"), "wb") as f:
            pickle.dump(meta_info, f)
        
        if self.centroid_index is not None:
            faiss.write_index(self.centroid_index, os.path.join(output_dir, "centroid_index.faiss"))
        
        print(f"클러스터링 결과가 {output_dir}에 저장되었습니다.")
    
    @classmethod
    def load(cls, input_dir: str):
        instance = cls()
        
        with open(os.path.join(input_dir, "documents.pkl"), "rb") as f:
            instance.documents = pickle.load(f)
        
        with open(os.path.join(input_dir, "document_ids.pkl"), "rb") as f:
            instance.document_ids = pickle.load(f)
        
        with open(os.path.join(input_dir, "clusters.pkl"), "rb") as f:
            instance.clusters = pickle.load(f)
        
        with open(os.path.join(input_dir, "meta_info.pkl"), "rb") as f:
            meta_info = pickle.load(f)
            instance.best_k = meta_info["best_k"]
            instance.dim = meta_info["dim"]
            instance.embedding_model_name = meta_info["embedding_model"]
        
        centroid_index_path = os.path.join(input_dir, "centroid_index.faiss")
        if os.path.exists(centroid_index_path):
            instance.centroid_index = faiss.read_index(centroid_index_path)
        
        print(f"클러스터링 결과 로드 완료 ({len(instance.documents)} 문서, {instance.best_k} 클러스터)")
        return instance

if __name__ == "__main__":
    pass