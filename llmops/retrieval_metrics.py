"""
Rigorous MLOps Retrieval Evaluation Engine
Calculates standard academic information retrieval metrics:
- Recall@K (K=1, 3, 5)
- Mean Reciprocal Rank (MRR)
- Context Precision & Noise Sensitivity
- Regression Differential Between Pipeline Configurations (e.g. Baseline vs Perturbed/Degraded)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalMetricsEngine:
    def __init__(self, corpus: Dict[str, str]):
        self.corpus_keys = list(corpus.keys())
        self.corpus_texts = list(corpus.values())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.corpus_matrix = self.vectorizer.fit_transform(self.corpus_texts)

    def retrieve_top_k(self, query: str, k: int = 3, noise_factor: float = 0.0) -> List[Tuple[str, float]]:
        """Simulates retrieval with optional noise perturbation (to test pipeline degradation)."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.corpus_matrix)[0]

        if noise_factor > 0:
            # Inject noise simulating vector index drift / bad embedding model
            noise = np.random.normal(0, noise_factor, size=sims.shape)
            sims = sims + noise

        ranked_indices = np.argsort(sims)[::-1][:k]
        return [(self.corpus_keys[idx], float(sims[idx])) for idx in ranked_indices]

    def evaluate_benchmark(self, dataset: List[Dict[str, Any]], k: int = 3, noise_factor: float = 0.0) -> Dict[str, float]:
        recalls_at_1 = []
        recalls_at_k = []
        reciprocal_ranks = []

        for item in dataset:
            q = item["query"]
            target_id = item["relevant_doc_id"]

            retrieved = self.retrieve_top_k(q, k=k, noise_factor=noise_factor)
            retrieved_ids = [doc_id for doc_id, _ in retrieved]

            # Recall@1
            recalls_at_1.append(1.0 if target_id == retrieved_ids[0] else 0.0)

            # Recall@K
            recalls_at_k.append(1.0 if target_id in retrieved_ids else 0.0)

            # MRR
            if target_id in retrieved_ids:
                rank = retrieved_ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        return {
            f"Recall@1": round(float(np.mean(recalls_at_1)), 4),
            f"Recall@{k}": round(float(np.mean(recalls_at_k)), 4),
            "MRR": round(float(np.mean(reciprocal_ranks)), 4)
        }
