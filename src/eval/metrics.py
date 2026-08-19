"""Phase 5 — precision@k, recall@k, NDCG (hand-written)."""

import numpy as np


def precision_at_k(retrieved_labels, k=5, threshold=2):
    """
    Compute Precision@k.

    Args:
        retrieved_labels: List[int] of relevance scores (0-3)
        k: number of top results to consider
        threshold: score >= threshold is considered relevant

    Returns:
        float: precision at k
    """
    retrieved_k = retrieved_labels[:k]
    num_relevant = sum(1 for score in retrieved_k if score >= threshold)
    return num_relevant / k if k > 0 else 0.0


def recall_at_k(retrieved_labels, k=20, threshold=2):
    """
    Compute Recall@k.

    Args:
        retrieved_labels: List[int] of relevance scores (0-3)
        k: number of top results to consider
        threshold: score >= threshold is considered relevant

    Returns:
        float: recall at k
    """
    retrieved_k = retrieved_labels[:k]
    num_relevant_retrieved = sum(1 for score in retrieved_k if score >= threshold)
    num_relevant_total = sum(1 for score in retrieved_labels if score >= threshold)

    return num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0.0


def ndcg_at_k(retrieved_labels, k=10, relevances=None):
    """
    Compute NDCG@k with graded relevance.

    Args:
        retrieved_labels: List[int] of relevance scores (0-3)
        k: number of top results to consider
        relevances: List[int] of ideal ranking (for computing ideal DCG)

    Returns:
        float: NDCG at k
    """
    retrieved_k = retrieved_labels[:k]

    # Compute DCG
    dcg = 0.0
    for i, score in enumerate(retrieved_k):
        dcg += score / np.log2(i + 2)  # Position i+1, so log2(i+2)

    # Compute ideal DCG (IDCG)
    if relevances is None:
        relevances = sorted(retrieved_labels, reverse=True)

    ideal_k = relevances[:k]
    idcg = 0.0
    for i, score in enumerate(ideal_k):
        idcg += score / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0
