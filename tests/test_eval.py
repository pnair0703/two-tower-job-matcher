import pytest
from src.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k

def test_precision_at_k():
    """Test P@k computation."""
    retrieved_labels = [3, 2, 1, 0, 0]  # Relevance scores

    # P@5 with threshold 2: 2 relevant out of 5
    p5 = precision_at_k(retrieved_labels, k=5, threshold=2)
    assert p5 == 0.4

    # P@2: 2 relevant out of 2
    p2 = precision_at_k(retrieved_labels, k=2, threshold=2)
    assert p2 == 1.0

def test_recall_at_k():
    """Test R@k computation."""
    retrieved_labels = [3, 2, 1, 0, 0]
    # Total relevant (score >= 2): 2

    # R@5: 2 relevant retrieved out of 2 total = 1.0
    r5 = recall_at_k(retrieved_labels, k=5, threshold=2)
    assert r5 == 1.0

def test_ndcg_at_k():
    """Test NDCG@k computation."""
    retrieved_labels = [3, 2, 1, 0, 0]

    # Ideal ranking: [3, 2, 1, 0, 0] (same as retrieved, so NDCG = 1.0)
    ndcg = ndcg_at_k(retrieved_labels, k=5, relevances=retrieved_labels)
    assert ndcg == 1.0

    # Worst ranking: [0, 0, 1, 2, 3] (reversed)
    ndcg_bad = ndcg_at_k([0, 0, 1, 2, 3], k=5, relevances=[3, 2, 1, 0, 0])
    assert ndcg_bad < 1.0
