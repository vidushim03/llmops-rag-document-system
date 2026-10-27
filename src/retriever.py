import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "from", "of", "about",
    "what", "which", "who", "whom", "when", "where", "why", "how", "tell",
    "explain", "describe", "please", "give", "show", "can", "could", "would",
    "should", "do", "does", "did", "any", "all", "some", "me", "my", "your",
}


def _keywords(text: str) -> set:
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower()))
    return words - _STOPWORDS


def rerank_docs(query: str, docs, top_k: int = 6):
    """Re-score retrieved chunks against the query and keep only the best.

    Uses a blended TF-IDF cosine + keyword-coverage score. This is a fast,
    dependency-light reranker that removes low-signal chunks before they reach
    the LLM (and the faithfulness checker). A cross-encoder can be swapped in
    later behind this same function.
    """
    if not docs:
        return []

    texts = [(doc.page_content or "") for doc in docs]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([query] + texts)
        cos_scores = cosine_similarity(matrix[0:1], matrix[1:])[0]
    except Exception as exc:
        # Degrade to keyword coverage only, but say so instead of silently
        # pretending the vector half of the reranker contributed a score.
        print(f"Reranker vector scoring failed, using keyword coverage only: {exc}")
        cos_scores = [0.0] * len(docs)

    query_words = _keywords(query)
    scored = []
    for doc, text, cos in zip(docs, texts, cos_scores):
        doc_words = _keywords(text)
        coverage = (
            len(query_words & doc_words) / len(query_words)
            if query_words else 0.0
        )
        score = 0.55 * float(cos) + 0.45 * coverage
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)

    query_lower = (query or "").lower()
    global_intent = any(
        token in query_lower
        for token in (
            "summar", "overview", "key point", "all document",
            "both document", "compare", "across", "entire",
        )
    )

    if global_intent:
        # Coverage-first: one best chunk per source so long multi-document
        # summaries do not silently drop an entire file, then fill by score.
        selected = []
        selected_keys = set()
        seen_sources = set()
        for score, doc in scored:
            source = (doc.metadata or {}).get("source")
            if source is None or source in seen_sources:
                continue
            key = (doc.page_content or "")[:100]
            selected.append((score, doc))
            selected_keys.add(key)
            seen_sources.add(source)
            if len(selected) >= top_k:
                break
        for score, doc in scored:
            if len(selected) >= top_k:
                break
            key = (doc.page_content or "")[:100]
            if key in selected_keys:
                continue
            selected.append((score, doc))
            selected_keys.add(key)
    else:
        # Precision-first: targeted questions only need the best matches.
        selected = scored[:top_k]
        positive = [(score, doc) for score, doc in selected if score > 0.0]
        if positive:
            selected = positive

    print(f"\n--- RERANKED {len(selected)} CHUNKS (from {len(docs)}) ---")
    for i, (score, doc) in enumerate(selected, 1):
        print(f"  [{i}] score={score:.3f} | {(doc.page_content or '')[:120]}")

    return [doc for _, doc in selected]


def retrieve_docs(db, query):
    """
    Smart retriever that searches with multiple query variations
    to handle cases where the document uses different terminology.
    """

    # Generate related search terms for common academic queries
    query_lower = query.lower()

    extra_queries = []

    if any(word in query_lower for word in ["conclusion", "conclude", "concluded"]):
        extra_queries = ["conclusion", "summary", "discussion", "findings", "result", "outcome"]

    elif any(word in query_lower for word in ["introduction", "background"]):
        extra_queries = ["introduction", "background", "overview", "motivation"]

    elif any(word in query_lower for word in ["result", "finding", "outcome"]):
        extra_queries = ["results", "findings", "outcome", "experiment", "analysis"]

    elif any(word in query_lower for word in ["method", "approach", "methodology"]):
        extra_queries = ["methodology", "method", "approach", "technique", "algorithm"]

    elif any(word in query_lower for word in ["abstract", "summary", "summarize", "summarise"]):
        extra_queries = ["abstract", "summary", "overview", "introduction"]

    # Collect docs from main query + extra queries
    seen = set()
    all_docs = []

    # Main query — fetch more docs
    main_docs = db.similarity_search(query, k=8)
    for d in main_docs:
        key = d.page_content[:100]
        if key not in seen:
            seen.add(key)
            all_docs.append(d)

    # Extra semantic queries
    for q in extra_queries:
        try:
            extra_docs = db.similarity_search(q, k=3)
            for d in extra_docs:
                key = d.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)
        except Exception as exc:
            print(f"Extra retrieval query failed for {q!r}: {exc}")

    # Ensure representation across all uploaded sources if multiple sources exist
    if hasattr(db, "list_sources") and db.source_count() > 1:
        sources = [s["source_name"] for s in db.list_sources()]
        for src_name in sources:
            src_represented = any(d.metadata.get("source") == src_name for d in all_docs)
            if not src_represented:
                # Retrieve specific chunks for under-represented source
                get_source_docs = getattr(db, "documents_for_source", None)
                if get_source_docs is not None:
                    src_docs = get_source_docs(src_name)
                    if src_docs:
                        all_docs.extend(src_docs[:3])

    print(f"\n--- Retrieved {len(all_docs)} candidate chunks ---")
    return rerank_docs(query, all_docs, top_k=6)
