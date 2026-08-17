import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple

class FAISSIndex:
    """Wrapper for FAISS index with posting ID mapping."""

    def __init__(self, index_path: str, map_path: str):
        """Load FAISS index and posting ID mapping."""
        self.index = faiss.read_index(index_path)
        with open(map_path, "rb") as f:
            self.idx_to_posting_id = {v: k for k, v in pickle.load(f).items()}
        self.dimension = self.index.d

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search index for top-k nearest neighbors.

        Args:
            query_embedding: (1, D) or (D,) float32 array, L2-normalized
            k: number of results

        Returns:
            posting_ids: List[str], sorted by score (highest first)
            scores: List[float], dot-product scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]

        # Ensure L2-normalized
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        posting_ids = [self.idx_to_posting_id[int(idx)] for idx in indices[0]]
        scores = distances[0].tolist()

        return posting_ids, scores
