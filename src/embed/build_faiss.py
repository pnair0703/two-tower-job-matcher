import argparse
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from src.data.s3_utils import read_jsonl_from_s3, download_from_s3, upload_to_s3, write_jsonl
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = setup_logging("build_faiss")

def build_faiss_index(embeddings_jsonl_path, posting_ids):
    """Build FAISS IndexFlatIP from embeddings."""
    logger.info(f"Building FAISS index from {embeddings_jsonl_path}")

    # Load embeddings
    embeddings_list = []
    posting_id_to_idx = {}

    with open(embeddings_jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            posting_id = record["posting_id"]
            embedding = np.array(record["embedding"], dtype=np.float32)
            embeddings_list.append(embedding)
            posting_id_to_idx[posting_id] = idx

    embeddings_array = np.vstack(embeddings_list)  # (N, D)
    logger.info(f"Embeddings shape: {embeddings_array.shape}")

    # Ensure L2-normalized (for dot-product similarity)
    faiss.normalize_L2(embeddings_array)

    # Build index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (dot product)
    index.add(embeddings_array)

    logger.info(f"✓ Built index with {index.ntotal} vectors")

    return index, posting_id_to_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", required=True, choices=["a", "b"])
    parser.add_argument("--embeddings-s3", help="S3 path to embeddings JSONL")
    parser.add_argument("--embeddings-local", help="Local path to embeddings JSONL")
    parser.add_argument("--output-index-s3", help="S3 path to save FAISS index")
    parser.add_argument("--output-map-s3", help="S3 path to save posting_id_to_idx map")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    config = load_config()
    bucket = config["s3_bucket"]

    # Load embeddings
    if args.embeddings_s3 and not args.local:
        logger.info(f"Downloading embeddings from S3")
        embeddings_path = Path(f"/tmp/embeddings_{args.tower}.jsonl")
        download_from_s3(args.embeddings_s3, bucket, embeddings_path)
    else:
        embeddings_path = Path(args.embeddings_local)

    # Read postings to get IDs
    postings = read_jsonl_from_s3("postings/postings.jsonl", bucket)
    posting_ids = [p["id"] for p in postings]

    # Build index
    index, posting_id_to_idx = build_faiss_index(embeddings_path, posting_ids)

    # Save index
    index_path = Path(f"/tmp/tower_{args.tower}.faiss")
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")

    # Save mapping
    map_path = Path(f"/tmp/posting_id_map_{args.tower}.pkl")
    with open(map_path, "wb") as f:
        pickle.dump(posting_id_to_idx, f)
    logger.info(f"Saved mapping to {map_path}")

    # Upload to S3
    if not args.local:
        output_index_s3 = args.output_index_s3 or f"indices/tower_{args.tower}.faiss"
        output_map_s3 = args.output_map_s3 or f"indices/posting_id_map_{args.tower}.pkl"
        upload_to_s3(index_path, output_index_s3, bucket)
        upload_to_s3(map_path, output_map_s3, bucket)

if __name__ == "__main__":
    main()
