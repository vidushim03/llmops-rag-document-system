import math
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.documents import Document


class SessionVectorStore:
    """In-memory, session-scoped vector store.

    Documents live only for the current session and disappear when the app
    restarts or redeploys. Mirrors the QdrantKnowledgeBase interface used by
    the app: add_documents, similarity_search, list_sources, delete_source,
    source_count, clear.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._entries = []
        self._sources = {}
        self._source_order = []

    def add_documents(self, documents, source_name: str) -> dict:
        docs = list(documents)
        if not docs:
            return {
                "source_id": None,
                "source_name": source_name,
                "chunk_count": 0,
            }

        for existing in list(self._sources.values()):
            if existing["source_name"].lower() == source_name.lower():
                self.delete_source(existing["source_id"])

        source_id = str(uuid.uuid4())
        page_numbers = set()

        for doc in docs:
            metadata = dict(doc.metadata or {})
            metadata.setdefault("source", source_name)
            page = metadata.get("page")
            if page is not None:
                page_numbers.add(page)
            vector = self.embeddings.embed_query(doc.page_content)
            self._entries.append(
                (
                    Document(page_content=doc.page_content, metadata=metadata),
                    vector,
                    source_id,
                )
            )

        self._sources[source_id] = {
            "source_id": source_id,
            "source_name": source_name,
            "chunk_count": len(docs),
            "page_count": len(page_numbers),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        self._source_order.append(source_id)
        return self._sources[source_id]

    def similarity_search(
        self, query: str, k: int = 4, source_name: Optional[str] = None
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        scored = []
        for doc, vector, _source_id in self._entries:
            if source_name is not None and (doc.metadata or {}).get("source") != source_name:
                continue
            # Cosine, not raw dot product: transformer embeddings are not
            # guaranteed to be unit-normalised, and the ranking must not depend
            # on vector magnitude.
            dot = sum(a * b for a, b in zip(query_vector, vector))
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if query_norm and vector_norm:
                score = dot / (query_norm * vector_norm)
            else:
                score = 0.0
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _dot, doc in scored[:k]]

    def all_texts(self) -> List[str]:
        """Return the raw text of every ingested chunk.

        Used to fit the semantic-drift baseline on the documents that are
        actually in this session, so drift is measured against real material
        instead of a hardcoded reference corpus.
        """
        return [doc.page_content for doc, _vector, _source_id in self._entries]

    def evaluation_questions(self, per_source: int = 5, source_names=None) -> List[dict]:
        """Create stable, document-derived checks for an uploaded corpus."""
        questions = []
        for source in self.list_sources():
            if source_names is not None and source["source_name"] not in source_names:
                continue
            docs = self.documents_for_source(source["source_name"])
            text = " ".join(doc.page_content for doc in docs)
            sentences = [part.strip() for part in __import__("re").split(r"(?<=[.!?])\s+", text) if len(part.split()) >= 6]
            seen = set()
            for sentence in sentences:
                normalized = sentence.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                words = sentence.split()
                question = "What does the document state about " + " ".join(words[:7]).strip(" ,.;:") + "?"
                questions.append({"question": question, "source": source["source_name"], "expected": sentence})
                if sum(q["source"] == source["source_name"] for q in questions) >= per_source:
                    break
        return questions

    def documents_for_source(self, source_name: str) -> List[Document]:
        """Return the stored chunks that belong to one source.

        Public counterpart to the private ``_entries`` list, so the retriever
        does not have to reach into storage internals.
        """
        return [
            doc
            for doc, _vector, _source_id in self._entries
            if (doc.metadata or {}).get("source") == source_name
        ]

    def list_sources(self) -> List[dict]:
        return [self._sources[source_id] for source_id in reversed(self._source_order)]

    def delete_source(self, source_id: str):
        if source_id not in self._sources:
            return
        self._sources.pop(source_id, None)
        self._source_order = [sid for sid in self._source_order if sid != source_id]
        self._entries = [entry for entry in self._entries if entry[2] != source_id]

    def source_count(self) -> int:
        return len(self._sources)

    def clear(self):
        self._entries.clear()
        self._sources.clear()
        self._source_order.clear()
