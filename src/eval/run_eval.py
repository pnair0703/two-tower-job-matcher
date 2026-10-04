"""Phase 5 — run all three systems, emit one results table."""

import argparse
import json
import numpy as np
import time
import torch
from pathlib import Path
from src.embed.index_utils import FAISSIndex
from src.eval.baseline import BM25Baseline
from src.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.data.s3_utils import read_jsonl_from_s3, download_from_s3
from src.utils.config import load_config, get_s3_paths
from src.utils.logging import setup_logging
from src.eval.eval_utils import format_results_table, save_results
from src.model.towers import TowerA, TowerB, JobResumePair

logger = setup_logging("run_eval")

def embed_resume_tower_a(resume_text, model, tokenizer):
    """Embed resume text with a trained TowerA model."""
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(resume_text).unsqueeze(0)
        embedding = model(tokens)
    return embedding.cpu().numpy()[0]

def embed_resume_tower_b(resume_text, model):
    """Embed resume text with a trained TowerB model."""
    model.eval()
    with torch.no_grad():
        embedding = model([resume_text])
    return embedding.cpu().numpy()[0]

def evaluate_system(search_fn, test_queries, test_labels, k_values=[5, 10, 20]):
    """
    Evaluate a retrieval system.

    Args:
        search_fn: callable(query, k) -> (posting_ids, scores). `query` is the raw
            resume text for BM25, or a precomputed embedding vector for FAISS —
            callers close over whatever the system needs.
        test_queries: List[Dict] with resume_text, posting_ids
        test_labels: Dict[posting_id] -> score
        k_values: List[int] of k values for metrics

    Returns:
        metrics: Dict with p@k, r@k, ndcg@k, latency
    """
    all_precisions_5 = []
    all_precisions_10 = []
    all_recalls_20 = []
    all_ndcgs_10 = []
    latencies = []

    for query in test_queries:
        resume_text = query["resume_text"]

        # Search
        start = time.time()
        posting_ids, scores = search_fn(resume_text, k=20)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        # Get labels for retrieved postings
        retrieved_labels = [test_labels.get(pid, 0) for pid in posting_ids]

        # Compute metrics
        p_5 = precision_at_k(retrieved_labels, k=5, threshold=2)  # Score >= 2 is positive
        p_10 = precision_at_k(retrieved_labels, k=10, threshold=2)
        r_20 = recall_at_k(retrieved_labels, k=20, threshold=2)
        ndcg_10 = ndcg_at_k(retrieved_labels, k=10)  # ideal ranking = retrieved_labels sorted descending

        all_precisions_5.append(p_5)
        all_precisions_10.append(p_10)
        all_recalls_20.append(r_20)
        all_ndcgs_10.append(ndcg_10)

    latencies = np.array(latencies)

    return {
        "p_5": np.mean(all_precisions_5),
        "p_10": np.mean(all_precisions_10),
        "r_20": np.mean(all_recalls_20),
        "ndcg_10": np.mean(all_ndcgs_10),
        "latency_p50_ms": np.percentile(latencies, 50),
        "latency_p99_ms": np.percentile(latencies, 99),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-dir", default="./indices")
    parser.add_argument("--resume-file", required=True, help="Path to resume text file")
    parser.add_argument("--test-labels-s3", default="labels/labels.jsonl")
    parser.add_argument("--model-a-path", default="./models/tower_a_model.pt", help="Path to trained TowerA state dict")
    parser.add_argument("--vocab-a-path", default="./models/tower_a_vocab.json", help="Path to TowerA vocab.json")
    parser.add_argument("--model-b-path", help="Path to fine-tuned TowerB state dict (optional)")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    config = load_config()
    bucket = config["s3_bucket"]

    # Load test labels
    logger.info("Loading test labels")
    all_labels = read_jsonl_from_s3(args.test_labels_s3, bucket)
    test_labels_list = [l for l in all_labels if l["split"] == "test"]
    test_labels_dict = {l["posting_id"]: l["score"] for l in test_labels_list}
    logger.info(f"Loaded {len(test_labels_dict)} test labels")

    # Load indices
    logger.info("Loading FAISS indices")
    index_a = FAISSIndex(
        f"{args.indices_dir}/tower_a.faiss",
        f"{args.indices_dir}/posting_id_map_a.pkl"
    )
    index_b = FAISSIndex(
        f"{args.indices_dir}/tower_b.faiss",
        f"{args.indices_dir}/posting_id_map_b.pkl"
    )

    # Load BM25 baseline
    postings = read_jsonl_from_s3("postings/postings.jsonl", bucket)
    bm25_baseline = BM25Baseline(postings)

    # Load resume
    with open(args.resume_file, "r") as f:
        resume_text = f.read()

    # Create test queries (just the resume, repeated)
    test_queries = [{"resume_text": resume_text} for _ in test_labels_list]

    # Load trained towers and embed the resume once per tower (query is constant
    # across test_queries, so there's no need to re-embed it per iteration)
    vocab = JobResumePair.load_vocab(args.vocab_a_path)
    tower_a = TowerA(vocab_size=vocab.vocab_size)
    tower_a.load_state_dict(torch.load(args.model_a_path, map_location="cpu", weights_only=True))
    resume_embedding_a = embed_resume_tower_a(resume_text, tower_a, vocab.tokenize)

    tower_b = TowerB()
    if args.model_b_path:
        tower_b.model.load_state_dict(torch.load(args.model_b_path, map_location="cpu", weights_only=True))
    resume_embedding_b = embed_resume_tower_b(resume_text, tower_b)

    # Evaluate all systems
    logger.info("Evaluating BM25 baseline")
    bm25_results = evaluate_system(
        lambda text, k: bm25_baseline.search(text, k=k), test_queries, test_labels_dict
    )

    logger.info("Evaluating Tower A")
    tower_a_results = evaluate_system(
        lambda text, k: index_a.search(resume_embedding_a, k=k), test_queries, test_labels_dict
    )

    logger.info("Evaluating Tower B")
    tower_b_results = evaluate_system(
        lambda text, k: index_b.search(resume_embedding_b, k=k), test_queries, test_labels_dict
    )

    # Aggregate results
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_set_size": len(test_labels_dict),
        "systems": {
            "bm25": bm25_results,
            "tower_a": tower_a_results,
            "tower_b": tower_b_results,
        },
    }

    # Save
    save_results(results, args.output)

    # Print table
    table = format_results_table(results)
    logger.info("Results:\n" + table)

if __name__ == "__main__":
    main()
