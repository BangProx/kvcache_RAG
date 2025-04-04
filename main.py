import os
import argparse
import sys
from dotenv import load_dotenv
from dataset_utils import create_sample_dataset, load_csv_dataset
from distributed_rag import VoronoiKVCacheRAG

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    parser = argparse.ArgumentParser(description="Voronoi KV-Cache RAG 시스템")
    subparsers = parser.add_subparsers(dest="command", help="명령어")
    
    # 데이터셋 준비 명령어
    prepare_parser = subparsers.add_parser("prepare", help="데이터셋 샘플 생성")
    prepare_parser.add_argument("--dataset", type=str, default="natural_questions", help="데이터셋 이름")
    prepare_parser.add_argument("--split", type=str, default="validation", help="데이터셋 분할")
    prepare_parser.add_argument("--size", type=int, default=1000, help="샘플 크기")
    prepare_parser.add_argument("--output", type=str, help="출력 파일 경로")
    
    # 클러스터링 명령어
    cluster_parser = subparsers.add_parser("cluster", help="문서 클러스터링")
    cluster_parser.add_argument("--dataset", type=str, required=True, help="CSV 데이터셋 경로")
    cluster_parser.add_argument("--text_column", type=str, default="document", help="문서 텍스트 컬럼명")
    cluster_parser.add_argument("--id_column", type=str, default="id", help="문서 ID 컬럼명")
    cluster_parser.add_argument("--k", type=int, default=10, help="클러스터 개수 (자동 결정 비활성화 시)")
    cluster_parser.add_argument("--no_auto_k", action="store_true", help="자동 클러스터 개수 결정 비활성화")
    
    # KV-Cache 생성 명령어
    cache_parser = subparsers.add_parser("cache", help="KV-Cache 생성")
    cache_parser.add_argument("--force", action="store_true", help="기존 캐시 강제 재생성")
    
    # 쿼리 명령어
    query_parser = subparsers.add_parser("query", help="쿼리 실행")
    query_parser.add_argument("--text", type=str, required=True, help="쿼리 텍스트")
    query_parser.add_argument("--max_tokens", type=int, default=256, help="최대 생성 토큰 수")
    query_parser.add_argument("--temperature", type=float, default=0.7, help="생성 temperature")
    
    for subparser in [cluster_parser, cache_parser, query_parser]:
        subparser.add_argument("--cluster_dir", type=str, default="voronoi_clusters", help="클러스터 정보 디렉토리")
        subparser.add_argument("--cache_dir", type=str, default="kvcaches", help="KV-Cache 저장 디렉토리")
        subparser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2", help="임베딩 모델 이름")
        subparser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="언어 모델 이름")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "prepare":
        print("\n=== 데이터셋 준비 ===")
        
        output_file, num_samples = create_sample_dataset(
            dataset_name=args.dataset,
            split=args.split,
            sample_size=args.size,
            output_file=args.output
        )
        
        print(f"\n데이터셋 준비 완료: {output_file} ({num_samples} 샘플)")
        return
    
    rag = VoronoiKVCacheRAG(
        cluster_dir=args.cluster_dir if hasattr(args, 'cluster_dir') else "voronoi_clusters",
        cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else "kvcaches",
        embed_model=args.embed_model if hasattr(args, 'embed_model') else "all-MiniLM-L6-v2",
        llm_model=args.llm_model if hasattr(args, 'llm_model') else "meta-llama/Llama-3.2-1B-Instruct",
        hf_token=hf_token
    )
    
    try:
        if args.command == "cluster":
            print("\n=== 문서 클러스터링 ===")
            
            documents, document_ids = load_csv_dataset(
                args.dataset,
                text_column=args.text_column,
                id_column=args.id_column
            )
            
            result = rag.build_clusters(
                documents, 
                document_ids, 
                find_optimal_k=not args.no_auto_k, 
                k=args.k
            )
            
            print("\n클러스터링 결과:")
            print(f"총 문서 수: {result['num_documents']}")
            print(f"클러스터 수: {result['num_clusters']}")
            print(f"임베딩 차원: {result['dimension']}")
            
            print("\n클러스터별 문서 수:")
            for cluster_id, size in sorted(result["clusters"].items()):
                print(f"클러스터 {cluster_id}: {size}개 문서")
        
        elif args.command == "cache":
            print("\n=== KV-Cache 생성 ===")
            
            results = rag.build_kvcaches(force_rebuild=args.force)
            
            success = sum(1 for v in results.values() if v)
            total = len(results)
            
            print(f"\nKV-Cache 생성 완료: {success}/{total} 성공")
        
        elif args.command == "query":
            print("\n=== 쿼리 실행 ===")
            
            result = rag.query(
                args.text, 
                max_new_tokens=args.max_tokens, 
                temperature=args.temperature
            )
            
            print(f"\n질문: {result['query']}")
            print(f"클러스터: {result['cluster_id']} (문서 {result['cluster_size']}개)")
            print(f"\n응답: {result['response']}")
        
        else:
            parser.print_help()
    
    finally:
        if 'rag' in locals():
            rag.clean_up()

if __name__ == "__main__":
    main()