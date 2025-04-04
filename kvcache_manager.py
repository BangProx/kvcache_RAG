import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm

class KVCacheManager:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
                 cache_dir: str = "kvcaches", 
                 hf_token: Optional[str] = None,
                 quantized: bool = True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.quantized = quantized
        
        self.model = None
        self.tokenizer = None
        
        self.kv_caches = {}
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        
        print(f"모델 로드: {self.model_name}")
        
        if self.quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=self.hf_token
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                token=self.hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=self.hf_token
            )
        
        return self.model, self.tokenizer
    
    def create_kvcache_prompt(self, documents: List[str]) -> str:
        combined_docs = "\n\n".join(documents)
        
        prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for question answering that provides accurate and helpful responses based on the given context.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context information:
------------------------------------------------
{combined_docs}
------------------------------------------------
Remember this context for future questions.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I'll remember the context you've provided and use it to answer your questions.
"""
        return prompt
    
    def create_kvcache(self, cluster_id: int, documents: List[str]) -> Optional[DynamicCache]:
        model, tokenizer = self.load_model()
        
        if not documents:
            print(f"클러스터 {cluster_id}에 문서가 없습니다. 기본 프롬프트 사용.")
            documents = ["이 클러스터에는 문서가 없습니다. 일반적인 지식에 기반하여 답변합니다."]
        
        prompt = self.create_kvcache_prompt(documents)
        
        device = model.device
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        print(f"클러스터 {cluster_id}의 KV-Cache 생성 중 ({len(documents)} 문서)...")
        kv_cache = DynamicCache()
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                use_cache=True
            )
        
        cache_path = self.get_cache_path(cluster_id)
        self.save_kvcache(outputs.past_key_values, cache_path)
        
        self.kv_caches[cluster_id] = outputs.past_key_values
        return outputs.past_key_values
    
    def get_cache_path(self, cluster_id: int) -> str:
        return os.path.join(self.cache_dir, f"cluster_kvcache_{cluster_id}.pt")
    
    def save_kvcache(self, kv_cache: DynamicCache, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(kv_cache, path)
            print(f"KV-Cache 저장 완료: {path}")
            return True
        except Exception as e:
            print(f"KV-Cache 저장 오류: {e}")
            return False
    
    def load_kvcache(self, cluster_id: int) -> Optional[DynamicCache]:
        if cluster_id in self.kv_caches:
            return self.kv_caches[cluster_id]
        
        path = self.get_cache_path(cluster_id)
        
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"KV-Cache 로드: {path}")
                try:
                    import torch.serialization
                    from transformers.cache_utils import DynamicCache
                    torch.serialization.add_safe_globals([DynamicCache])
                    
                    kv_cache = torch.load(path, map_location="cpu")
                    
                    if self.model:
                        target_device = self.model.device
                        for i in range(len(kv_cache.key_cache)):
                            kv_cache.key_cache[i] = kv_cache.key_cache[i].to(target_device)
                            kv_cache.value_cache[i] = kv_cache.value_cache[i].to(target_device)
                        
                        if not hasattr(kv_cache, 'cache_position') or kv_cache.cache_position is None:
                            kv_len = kv_cache.key_cache[0].shape[-2]
                            kv_cache.cache_position = torch.tensor([kv_len], dtype=torch.long, device=target_device)
                    
                except Exception as e1:
                    try:
                        print(f"안전 모드로 다시 시도 중...")
                        kv_cache = torch.load(path, map_location="cpu", weights_only=False)
                        
                        if self.model:
                            target_device = self.model.device
                            for i in range(len(kv_cache.key_cache)):
                                kv_cache.key_cache[i] = kv_cache.key_cache[i].to(target_device)
                                kv_cache.value_cache[i] = kv_cache.value_cache[i].to(target_device)
                            
                            if not hasattr(kv_cache, 'cache_position') or kv_cache.cache_position is None:
                                kv_len = kv_cache.key_cache[0].shape[-2]
                                kv_cache.cache_position = torch.tensor([kv_len], dtype=torch.long, device=target_device)
                        
                    except Exception as e2:
                        print(f"KV-Cache 로드 오류: {e2}")
                        return None
                
                self.kv_caches[cluster_id] = kv_cache
                return kv_cache
            else:
                print(f"KV-Cache 파일이 존재하지 않음: {path}")
                return None
        except Exception as e:
            print(f"KV-Cache 로드 오류: {e}")
            return None
    
    def create_cluster_kvcaches(self, 
                               cluster_documents: Dict[int, List[str]], 
                               force_rebuild: bool = False) -> Dict[int, bool]:
        results = {}
        
        for cluster_id, documents in tqdm(cluster_documents.items(), desc="클러스터 KV-Cache 생성"):
            cache_path = self.get_cache_path(cluster_id)
            
            if not force_rebuild and os.path.exists(cache_path):
                print(f"클러스터 {cluster_id}의 KV-Cache가 이미 존재함 (건너뜀)")
                results[cluster_id] = True
                continue
            
            if len(documents) == 0:
                print(f"클러스터 {cluster_id}에 문서가 없음 (건너뜀)")
                results[cluster_id] = False
                continue
            
            try:
                self.create_kvcache(cluster_id, documents)
                results[cluster_id] = True
            except Exception as e:
                print(f"클러스터 {cluster_id} KV-Cache 생성 오류: {e}")
                results[cluster_id] = False
        
        return results
    
    def clean_kvcache(self, kv_cache: DynamicCache, origin_len: int = None) -> DynamicCache:
        if origin_len is None:
            return kv_cache
        
        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i] = kv_cache.key_cache[i][:, :, :origin_len, :]
            kv_cache.value_cache[i] = kv_cache.value_cache[i][:, :, :origin_len, :]
        
        return kv_cache
        
    def generate_with_kvcache(self, 
                            query: str, 
                            kv_cache: DynamicCache, 
                            max_new_tokens: int = 256,
                            temperature: float = 0.7) -> str:
        model, tokenizer = self.load_model()
        
        embed_device = model.model.embed_tokens.weight.device
        
        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i] = kv_cache.key_cache[i].to(embed_device)
            kv_cache.value_cache[i] = kv_cache.value_cache[i].to(embed_device)
        
        prompt = f"""
    <|start_header_id|>user<|end_header_id|>
    {query}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
        
        origin_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = origin_ids.to(embed_device)
        
        output_ids = input_ids.clone()
        next_token = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=next_token, 
                    past_key_values=kv_cache,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                
                next_token = next_token.to(embed_device)
                
                kv_cache = outputs.past_key_values
                
                output_ids = torch.cat([output_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        generated_text = tokenizer.decode(output_ids[0, origin_ids.shape[1]:], skip_special_tokens=True)
        
        return generated_text
    
    def unload_kvcache(self, cluster_id: int):
        if cluster_id in self.kv_caches:
            del self.kv_caches[cluster_id]
    
    def clear_kvcaches(self):
        self.kv_caches.clear()
    
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("모델 언로드 및 GPU 메모리 정리 완료")