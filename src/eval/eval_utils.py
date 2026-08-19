import json
import numpy as np
from pathlib import Path
from src.utils.logging import setup_logging

logger = setup_logging("eval_utils")

def format_results_table(results):
    """Format results into markdown table."""
    template = """
| System | P@5 | P@10 | R@20 | NDCG@10 | Latency p50 | Latency p99 |
|--------|-----|------|------|---------|-------------|-------------|
| BM25 baseline | {bm25[p_5]:.3f} | {bm25[p_10]:.3f} | {bm25[r_20]:.3f} | {bm25[ndcg_10]:.3f} | {bm25[latency_p50_ms]:.1f}ms | {bm25[latency_p99_ms]:.1f}ms |
| Tower A (scratch) | {tower_a[p_5]:.3f} | {tower_a[p_10]:.3f} | {tower_a[r_20]:.3f} | {tower_a[ndcg_10]:.3f} | {tower_a[latency_p50_ms]:.1f}ms | {tower_a[latency_p99_ms]:.1f}ms |
| Tower B (pretrained) | {tower_b[p_5]:.3f} | {tower_b[p_10]:.3f} | {tower_b[r_20]:.3f} | {tower_b[ndcg_10]:.3f} | {tower_b[latency_p50_ms]:.1f}ms | {tower_b[latency_p99_ms]:.1f}ms |
"""
    return template.format(
        bm25=results["systems"]["bm25"],
        tower_a=results["systems"]["tower_a"],
        tower_b=results["systems"]["tower_b"],
    )

def save_results(results, output_path):
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")

def load_results(output_path):
    """Load results from JSON."""
    with open(output_path, "r") as f:
        return json.load(f)
