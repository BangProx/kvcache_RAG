"""
데이터 로드/전처리 구현
"""

import os
import csv
import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional, Any

def create_sample_dataset(dataset_name: str = "natural_questions", 
                         split: str = "validation",
                         sample_size: int = 1000, 
                         output_file: Optional[str] = None) -> Tuple[str, int]:
    """
nq 처리 함수
    """
    if output_file is None:
        os.makedirs("dataset", exist_ok=True)
        output_file = f"dataset/{dataset_name.replace('/', '_')}_{split}_sample.csv"
    
    print(f"데이터셋 로드: {dataset_name} ({split} 분할)...")
    
    # 스트리밍 + take로 시간 단축 및 메모리 단축
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.take(sample_size)
    
    print(f"{sample_size} 샘플 처리 중...")
    data = []

    # BeautifulSoup으로 HTML 파싱    
    def extract_text_from_html(html_content):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style", "meta", "link", "noscript"]):
                script.extract()
            
            body = soup.body if soup.body else soup
            
            text = ' '.join(body.get_text(separator=' ').split())
            
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            print(f"HTML 파싱 오류: {e}")
            import re
            return re.sub(r'<[^>]+>', ' ', html_content)
    
    for item in dataset:
        doc_id = len(data)
        
        if dataset_name == "natural_questions":
            question = ""
            if "question" in item and isinstance(item["question"], dict) and "text" in item["question"]:
                question = item["question"]["text"]
            
            doc_text = ""
            doc_title = ""
            
            if "document" in item and isinstance(item["document"], dict):
                if "title" in item["document"]:
                    doc_title = item["document"]["title"]
                
                if "html" in item["document"]:
                    html_content = item["document"]["html"]
                    doc_text = extract_text_from_html(html_content)
            
            if doc_text:
                doc_text = doc_text.replace("Jump to navigation", "")
                doc_text = doc_text.replace("Jump to search", "")
                
                import re
                doc_text = re.sub(r'\s+', ' ', doc_text).strip()
            
            full_doc = ""
            if doc_title:
                full_doc += f"제목: {doc_title}\n\n"
            if doc_text:
                full_doc += doc_text
            
            if question and full_doc:
                max_length = 1000
                truncated_doc = full_doc[:max_length] + "..." if len(full_doc) > max_length else full_doc
                
                data.append({
                    "id": doc_id,
                    "question": question,
                    "document": truncated_doc
                })
        
        elif "text" in item:
            text = item["text"]
            if isinstance(text, str) and text.strip():
                max_length = 1000
                truncated_text = text[:max_length] + "..." if len(text) > max_length else text
                
                data.append({
                    "id": doc_id,
                    "document": truncated_text
                })
        
        else:
            doc_dict = {"id": doc_id}
            
            for key, value in item.items():
                if isinstance(value, str):
                    doc_dict[key] = value
                elif isinstance(value, (int, float, bool)):
                    doc_dict[key] = value
            
            if "document" not in doc_dict and "text" in doc_dict:
                doc_dict["document"] = doc_dict["text"]
            
            if "document" in doc_dict and doc_dict["document"].strip():
                data.append(doc_dict)
    
    print(f"{len(data)} 문서 저장: {output_file}")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"샘플 데이터셋 생성 완료 ({len(data)} 문서)")
    
    if len(data) > 0:
        print("\n데이터셋 샘플:")
        print(df.head(3))
    
    return output_file, len(data)

def load_csv_dataset(file_path: str, 
                    text_column: str = "document", 
                    id_column: Optional[str] = "id") -> Tuple[List[str], List[Any]]:
    print(f"CSV 파일 로드: {file_path}")
    df = pd.read_csv(file_path)
    
    if text_column not in df.columns:
        raise ValueError(f"텍스트 컬럼 '{text_column}'이 데이터셋에 없습니다.")
    
    documents = []
    document_ids = []
    
    for idx, row in df.iterrows():
        if isinstance(row[text_column], str) and row[text_column].strip():
            documents.append(row[text_column])
            
            if id_column and id_column in df.columns:
                document_ids.append(row[id_column])
            else:
                document_ids.append(idx)
    
    print(f"{len(documents)} 문서 로드됨")
    return documents, document_ids

def chunk_documents(documents: List[str], 
                   document_ids: List[Any],
                   chunk_size: int = 500, 
                   overlap: int = 50) -> Tuple[List[str], List[Any]]:
    chunks = []
    chunk_ids = []
    
    for doc_idx, (doc, doc_id) in enumerate(zip(documents, document_ids)):
        if len(doc) <= chunk_size:
            chunks.append(doc)
            chunk_ids.append(doc_id)
        else:
            start = 0
            chunk_idx = 0
            
            while start < len(doc):
                end = min(start + chunk_size, len(doc))
                
                if end - start < chunk_size // 2 and start > 0:
                    break
                
                chunks.append(doc[start:end])
                chunk_ids.append(f"{doc_id}_chunk{chunk_idx}")
                
                start += chunk_size - overlap
                chunk_idx += 1
    
    print(f"총 {len(documents)} 문서에서 {len(chunks)} 청크 생성됨")
    return chunks, chunk_ids