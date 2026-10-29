import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.knowledge_base import QdrantKnowledgeBase, build_chat_collection_name


class TinyEmbeddings:
    """Deterministic local embeddings for storage and isolation checks."""

    def embed_query(self, text):
        vector = [0.0] * 16
        for index, character in enumerate(text.lower()):
            vector[(ord(character) + index) % len(vector)] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class ChatKnowledgeBaseTests(unittest.TestCase):
    def test_chat_collections_are_isolated_and_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = QdrantKnowledgeBase(
                TinyEmbeddings(), "test", base_dir=root,
                collection_name=build_chat_collection_name("chat-one"),
            )
            first.add_documents(
                [Document(page_content="Oranges contain vitamin C and fiber.", metadata={"page": 1})],
                "fruit.txt",
            )
            first.close()

            reopened = QdrantKnowledgeBase(
                TinyEmbeddings(), "test", base_dir=root,
                collection_name=build_chat_collection_name("chat-one"),
            )
            self.assertEqual(reopened.source_count(), 1)
            self.assertEqual(reopened.all_texts(), ["Oranges contain vitamin C and fiber."])
            reopened.close()

            second = QdrantKnowledgeBase(
                TinyEmbeddings(), "test", base_dir=root,
                collection_name=build_chat_collection_name("chat-two"),
            )
            self.assertEqual(second.source_count(), 0)
            second.close()

    def test_evaluation_baseline_is_stable_and_comparison_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QdrantKnowledgeBase(
                TinyEmbeddings(), "test", base_dir=Path(directory),
                collection_name=build_chat_collection_name("chat-one"),
            )
            original = [{"question": "What vitamins are in oranges?", "expected": "vitamin C", "recall": 0.8}]
            baseline = store.save_evaluation_baseline(original)
            latest = store.save_evaluation_comparison(
                [{"question": "What vitamins are in oranges?", "expected": "vitamin C", "recall": 0.4}],
                new_document_results=[{"question": "What is new?", "expected": "new text", "recall": 1.0}],
            )

            self.assertEqual(store.get_evaluation_baseline(), baseline)
            self.assertEqual(latest["delta"], -0.4)
            self.assertEqual(latest["new_document_mean_keyword_coverage"], 1.0)
            self.assertEqual(store.get_latest_evaluation(), latest)
            store.close()

    def test_dynamic_questions_can_target_newly_uploaded_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QdrantKnowledgeBase(
                TinyEmbeddings(), "test", base_dir=Path(directory),
                collection_name=build_chat_collection_name("chat-one"),
            )
            store.add_documents(
                [Document(page_content="A red planet orbits the star called the Sun.")],
                "mars.txt",
            )
            store.add_documents(
                [Document(page_content="A blue ocean covers most of this small planet.")],
                "ocean.txt",
            )

            cases = store.evaluation_questions(source_names=["ocean.txt"])
            self.assertTrue(cases)
            self.assertTrue(all(case["source"] == "ocean.txt" for case in cases))
            self.assertTrue(any("blue ocean" in case["expected"] for case in cases))
            store.close()

    def test_later_upload_does_not_replace_earlier_source(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QdrantKnowledgeBase(
                TinyEmbeddings(), "test", base_dir=Path(directory),
                collection_name=build_chat_collection_name("chat-one"),
            )
            store.add_documents(
                [Document(page_content="Earlier file: customer records are kept for seven years.")],
                "earlier.pdf",
            )
            store.add_documents(
                [Document(page_content="Later file: model deployments use a canary rollout.")],
                "later.pdf",
            )

            self.assertEqual(store.source_count(), 2)
            earlier_hits = store.similarity_search(
                "How long are customer records kept?", k=3, source_name="earlier.pdf"
            )
            self.assertTrue(earlier_hits)
            self.assertTrue(all(hit.metadata["source"] == "earlier.pdf" for hit in earlier_hits))
            self.assertIn("seven years", earlier_hits[0].page_content)
            store.close()


if __name__ == "__main__":
    unittest.main()
