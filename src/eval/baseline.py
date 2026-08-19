"""Phase 5 — BM25 / TF-IDF keyword baseline."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BM25Baseline:
    """Simple TF-IDF baseline for job matching."""

    def __init__(self, postings):
        self.postings = postings
        self.posting_ids = [p["id"] for p in postings]
        self.posting_id_to_text = {p["id"]: f"{p['title']} {p['description']}" for p in postings}
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.posting_vectors = self.vectorizer.fit_transform(
            [self.posting_id_to_text[pid] for pid in self.posting_ids]
        )

    def search(self, query_text: str, k: int = 10):
        """Search index for top-k postings.

        Args:
            query_text: str, query text (e.g., resume)
            k: int, number of top results

        Returns:
            posting_ids: List[str], top-k posting IDs
            scores: List[float], similarity scores
        """
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.posting_vectors)[0]
        top_indices = np.argsort(-scores)[:k]

        result_ids = [self.posting_ids[i] for i in top_indices]
        result_scores = scores[top_indices].tolist()

        return result_ids, result_scores
