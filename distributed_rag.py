import os
import torch
from typing import Dict, List, Optional, Tuple, Any
from voronoi_cluster import VoronoiCluster
from kvcache_manager import KVCacheManager

class VoronoiKVCacheRAG:
    def __init__(self, 
                 cluster_dir: str = "voronoi_clusters",
                 cache_dir: str = "kvcaches",
                 embed_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "meta-llama/Llama-3.2-1B-Instruct",
                 hf_token: Optional[str] = None,
                 quantized: bool = True):
        self.cluster_dir = cluster_dir
        self.cache_dir = cache_dir
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.hf_token = hf_token
        self.quantized = quantized
        
        self.cluster_manager = None
        self.cache_manager = None
        
        self.is_loaded = False
        self.last_cluster_id = None
    
    def load_resources(self):
        if self.is_loaded:
            return
        
        print(f"클러스터 정보 로드: {self.cluster_dir}")
        if os.path.exists(self.cluster_dir):
            self.cluster_manager = VoronoiCluster.load(self.cluster_dir)
        else:
            self.cluster_manager = VoronoiCluster(self.embed_model)
        
        self.cache_manager = KVCacheManager(
            model_name=self.llm_model,
            cache_dir=self.cache_dir,
            hf_token=self.hf_token,
            quantized=self.quantized
        )
        
        self.is_loaded = True
    
    def build_clusters(self, documents: List[str], document_ids: Optional[List[Any]] = None, 
                      find_optimal_k: bool = True, k: int = 10) -> Dict:
        self.load_resources()
        
        result = self.cluster_manager.fit(documents, document_ids, find_optimal_k, k)
        
        os.makedirs(self.cluster_dir, exist_ok=True)
        self.cluster_manager.save(self.cluster_dir)
        
        return result
    
    def build_kvcaches(self, force_rebuild: bool = False) -> Dict[int, bool]:
        self.load_resources()
        cluster_documents = {}
        for cluster_id, doc_indices in self.cluster_manager.clusters.items():
            cluster_docs = [self.cluster_manager.documents[idx] for idx in doc_indices]
            cluster_documents[cluster_id] = cluster_docs
        
        results = self.cache_manager.create_cluster_kvcaches(cluster_documents, force_rebuild)
        
        return results
    
    def query(self, query_text: str, max_new_tokens: int = 256, 
             temperature: float = 0.7) -> Dict:
        self.load_resources()
        
        cluster_id = self.cluster_manager.predict_cluster(query_text)
        self.last_cluster_id = cluster_id
        
        print(f"쿼리에 적합한 클러스터: {cluster_id}")
        
        if cluster_id in self.cluster_manager.clusters and not self.cluster_manager.clusters[cluster_id]:
            print(f"클러스터 {cluster_id}에 문서가 없습니다. 다른 클러스터를 선택합니다.")
            for alt_id, doc_indices in self.cluster_manager.clusters.items():
                if doc_indices: 
                    cluster_id = alt_id
                    self.last_cluster_id = cluster_id
                    print(f"대체 클러스터 선택됨: {cluster_id}")
                    break
        
        kv_cache = self.cache_manager.load_kvcache(cluster_id)
        
        if kv_cache is None:
            print(f"클러스터 {cluster_id}의 KV-Cache 생성 중...")
            cluster_docs = self.cluster_manager.get_cluster_documents(cluster_id)
            kv_cache = self.cache_manager.create_kvcache(cluster_id, cluster_docs)
        
        if kv_cache is None:
            print("유효한 KV-Cache를 생성할 수 없습니다. 기본 응답 반환.")
            return {
                "query": query_text,
                "response": "죄송합니다, 질문에 답변할 수 있는 정보가 충분하지 않습니다.",
                "cluster_id": cluster_id,
                "cluster_size": 0
            }
        
        response = self.cache_manager.generate_with_kvcache(
            query_text, 
            kv_cache, 
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        result = {
            "query": query_text,
            "response": response,
            "cluster_id": cluster_id,
            "cluster_size": len(self.cluster_manager.clusters.get(cluster_id, []))
        }
        
        return result
    
    def clean_up(self):
        if self.cache_manager:
            self.cache_manager.clear_kvcaches()
            self.cache_manager.unload_model()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("리소스 정리 완료")