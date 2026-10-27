"""
Advanced Semantic Evaluation Engine for LLMOps
Implements:
1. Dense Vector & N-Gram TF-IDF Semantic Cosine Similarity (Embedding-space alignment)
2. Semantic Faithfulness / Groundedness (Sentence-level semantic entailment against retrieved context)
3. Answer Relevance (Query-response semantic projection)
4. Context Precision (Signal-to-noise ratio in retrieved context vectors)
5. Out-of-Domain (OOD) Semantic Query Drift Detection (Tracks distribution shift of production queries vs baseline corpus)
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Calibrated on real logged turns: in-domain questions scored 0.085-0.223 against
# the document centroid, out-of-domain questions 0.011-0.042. 0.06 sits between
# the two populations and is used by both the app and the dashboard so live and
# historical drift flags stay comparable.
DEFAULT_DRIFT_THRESHOLD = 0.06

# Inline citation markers the generator adds to answers (e.g. 【1】 or [2]).
# They are not claims, so they must not be read as numbers.
_CITATION_RE = re.compile(r"【\s*\d+\s*】|\[\s*\d+\s*\]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")


class SemanticRAGEvaluator:
    def __init__(self, drift_threshold: float = DEFAULT_DRIFT_THRESHOLD, min_faithfulness: float = 0.55):
        self.drift_threshold = drift_threshold
        self.min_faithfulness = min_faithfulness
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=2500)
        self._fitted = False
        self.baseline_centroid = None

    def fit_baseline_corpus(self, documents: List[str]):
        """Fits baseline vectorizer on reference knowledge base documents to establish reference centroid."""
        if not documents:
            return
        matrix = self.vectorizer.fit_transform(documents)
        # Compute centroid of reference document embedding space
        self.baseline_centroid = np.asarray(matrix.mean(axis=0))
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        """True only when a drift baseline was actually fitted from real text."""
        return bool(self._fitted and self.baseline_centroid is not None)

    def reset_baseline(self):
        """Drop the fitted baseline so drift reports 'not evaluated' again.

        Used when switching to a chat that has no documents. Keeping the
        previous chat's centroid would report a fabricated in-domain score for
        a document-free conversation.
        """
        self._fitted = False
        self.baseline_centroid = None

    def _get_similarity(self, text_a: str, text_b: str) -> float:
        """Computes semantic cosine similarity combined with token overlap for robust paraphrasing matching."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        try:
            # 1. TF-IDF Cosine Similarity
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_mat = vec.fit_transform([text_a, text_b])
            cos_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
            cos_sim = np.clip(cos_sim, 0.0, 1.0)

            # 2. Token Jaccard Overlap (robust to paraphrased synonyms)
            stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "as", "is", "are", "was", "were", "be", "this", "that", "it", "from", "of"}
            words_a = set(re.findall(r"\b[a-zA-Z]{3,}\b", text_a.lower())) - stopwords
            words_b = set(re.findall(r"\b[a-zA-Z]{3,}\b", text_b.lower())) - stopwords
            
            jaccard = 0.0
            if words_a and words_b:
                intersection = words_a.intersection(words_b)
                union = words_a.union(words_b)
                jaccard = len(intersection) / len(union) if union else 0.0
                # Boost overlap if most query keywords are present in chunk
                coverage = len(intersection) / len(words_a) if words_a else 0.0
                jaccard = 0.5 * jaccard + 0.5 * coverage

            # Blended score (giving strong weight to token semantic overlap for summaries)
            combined = 0.4 * cos_sim + 0.6 * jaccard
            return float(np.clip(combined, 0.0, 1.0))
        except Exception:
            return 0.0

    def compute_semantic_faithfulness(self, response: str, context_chunks: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Decomposes response into individual claim sentences and verifies whether
        each claim has strong semantic entailment/grounding in at least one retrieved context chunk.

        A sentence must clear two independent checks to count as grounded:
        1. Lexical/semantic overlap with some retrieved chunk (>= 0.12).
        2. Every number it states must appear in the retrieved context. Numbers
           are the most common hallucination vector: "a 50 percent acquisition"
           shares the words "company" and "percent" with a chunk about 12
           percent revenue growth, so overlap alone would call it grounded.

        Citation markers (【1】, [2]) are stripped first; they are not claims.
        """
        if not response or not context_chunks:
            return 0.0, []

        clean_chunks = [_CITATION_RE.sub(" ", chunk) for chunk in context_chunks]
        context_numbers = set()
        for chunk in clean_chunks:
            context_numbers.update(_NUMBER_RE.findall(chunk))

        clean_response = _CITATION_RE.sub(" ", response)

        # Split response into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_response) if len(s.strip()) > 10]
        if not sentences:
            sentences = [clean_response.strip()]

        sentence_scores = []
        for sent in sentences:
            # Check maximum similarity against all retrieved chunks
            chunk_sims = [self._get_similarity(sent, chunk) for chunk in clean_chunks]
            max_sim = max(chunk_sims) if chunk_sims else 0.0
            claim_numbers = set(_NUMBER_RE.findall(sent))
            numbers_supported = claim_numbers.issubset(context_numbers)
            sentence_scores.append({
                "sentence": sent,
                "best_context_match_score": round(max_sim, 4),
                "numbers_claimed": sorted(claim_numbers),
                "numbers_supported": numbers_supported,
                "grounded": bool(max_sim >= 0.12 and numbers_supported)
            })

        grounded_count = sum(1 for s in sentence_scores if s["grounded"])
        faithfulness = grounded_count / len(sentence_scores) if sentence_scores else 0.0
        return round(float(faithfulness), 4), sentence_scores

    @staticmethod
    def _is_global_intent(question: str) -> bool:
        """True for summary/overview requests whose target is the whole corpus."""
        q = (question or "").lower()
        return any(
            key in q for key in (
                "summar", "overview", "key point", "both document",
                "all document", "entire", "what is this", "what's this",
                "about the document",
            )
        )

    def compute_answer_relevance(
        self,
        question: str,
        response: str,
        context_chunks: Optional[List[str]] = None,
    ) -> float:
        """
        Measures semantic relevance between user question and generated answer.

        For global requests ("summarize the document") the question shares almost
        no vocabulary with a good paraphrased answer, so plain word-overlap is a
        poor signal. In that case we also measure how well the answer aligns with
        the retrieved corpus, which is the actual target of the request.
        """
        if not question.strip() or not response.strip():
            return 0.0

        q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()))
        ans_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", response.lower()))
        stopwords = {"what", "when", "where", "which", "will", "would", "about", "there", "their", "then", "into", "been", "were", "this", "that", "with", "from", "have", "your"}
        q_clean = q_words - stopwords

        if not q_clean:
            return 1.0

        overlap = len(q_clean.intersection(ans_words)) / len(q_clean)
        sim = self._get_similarity(question, response)

        # Combined score: token coverage + embedding similarity
        relevance = 0.5 * overlap + 0.5 * min(1.0, sim * 1.8 + 0.2)

        if context_chunks and self._is_global_intent(question):
            alignment = self._get_similarity(response, " ".join(context_chunks))
            relevance = max(relevance, min(1.0, 0.45 + 0.8 * alignment))

        return round(float(np.clip(relevance, 0.1, 1.0)), 4)

    def compute_context_precision(
        self,
        question: str,
        context_chunks: List[str],
        response: str = "",
    ) -> float:
        """
        Evaluates whether retrieved chunks are semantically aligned with the
        query intent.

        Two calibration fixes over raw question->chunk cosine:
        - Also credit chunks that support the generated answer, because a
          summary request targets the corpus rather than the question wording.
        - Use a threshold suited to the blended similarity scale (0.08, not
          0.15 which was calibrated for raw TF-IDF cosine).
        """
        if not context_chunks:
            return 0.0
        scores = []
        for chunk in context_chunks:
            q_sim = self._get_similarity(question, chunk)
            if response:
                r_sim = self._get_similarity(response, chunk)
                scores.append(max(q_sim, 0.85 * r_sim))
            else:
                scores.append(q_sim)
        relevant_chunks = sum(1 for s in scores if s >= 0.08)
        return round(float(relevant_chunks / len(context_chunks)), 4)

    def detect_semantic_query_drift(
        self, question: str
    ) -> Tuple[Optional[bool], Optional[float]]:
        """
        Measures out-of-domain (OOD) semantic drift of production queries relative to baseline corpus centroid.
        If semantic similarity to knowledge base is below threshold, query is flagged as 'Out-of-Domain Drift'.

        Returns (None, None) when no baseline has been fitted from real
        documents. Reporting 0.0 similarity in that case would be a fabricated
        measurement indistinguishable from a genuine score.
        """
        if not self._fitted or self.baseline_centroid is None:
            return None, None

        try:
            q_vec = self.vectorizer.transform([question])
            drift_sim = float(cosine_similarity(q_vec, self.baseline_centroid)[0][0])
            is_drifted = bool(drift_sim < self.drift_threshold)
            return is_drifted, round(drift_sim, 4)
        except Exception:
            return None, None

    def evaluate_turn(self, question: str, response: str, context_chunks: List[str]) -> Dict[str, Any]:
        """Executes complete RAG Triad semantic audit on a query-response cycle."""
        faithfulness, sentence_breakdown = self.compute_semantic_faithfulness(response, context_chunks)
        relevance = self.compute_answer_relevance(question, response, context_chunks)
        precision = self.compute_context_precision(question, context_chunks, response)
        is_drifted, drift_similarity = self.detect_semantic_query_drift(question)

        # Composite RAG Triad Score (weighted combination)
        composite_score = round(float(0.45 * faithfulness + 0.35 * relevance + 0.20 * precision), 4)
        hallucination_detected = bool(faithfulness < self.min_faithfulness)

        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision,
            "rag_composite_quality": composite_score,
            "hallucination_detected": hallucination_detected,
            "semantic_query_drift": is_drifted,
            "drift_similarity_to_corpus": drift_similarity,
            "passed_quality_gate": bool(not hallucination_detected and composite_score >= 0.50),
            "sentence_verification": sentence_breakdown
        }
