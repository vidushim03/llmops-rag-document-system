"""Populate the audit log with genuine end-to-end turns.

This runs the real application pipeline (same loader, chunker, store,
retriever, agent graph and evaluator the Streamlit app uses) over real PDFs in
this repository. Nothing is mocked and no score is written by hand: every
number comes from the live semantic evaluator.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_live_eval.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_pdf
from src.chunker import chunk_data
from src.embedding import get_embeddings
from src.session_store import SessionVectorStore
from src.agent import stream_agent
from llmops.guardrails import LLMGuardrails
from llmops.semantic_evaluator import SemanticRAGEvaluator
from llmops.telemetry import LLMOpsTelemetry
from llmops.text_utils import is_non_answer

DOCS = [
    "research_paper_submission/MLOps_Overview_Definition_Architecture_IEEE_Access_2023.pdf",
    "research_paper_submission/Hidden_Technical_Debt_in_Machine_Learning_Systems_NeurIPS.pdf",
]

QUESTIONS = [
    "What does the paper define as MLOps?",
    "Why are MLOps principles and practices important for machine learning systems?",
    "What are the core components of the MLOps architecture described in the paper?",
    "How does the paper describe the relationship between DevOps and MLOps?",
    "What is hidden technical debt in machine learning systems?",
    "Which types of technical debt does the paper identify in ML systems?",
    "Why is data dependency considered a source of hidden debt in ML systems?",
    "What do the authors recommend for managing technical debt in ML systems?",
    "What is the capital of France?",
    "Who won the 2018 FIFA World Cup?",
]


def main():
    store = SessionVectorStore(get_embeddings(mode="transformer"))
    for path in DOCS:
        if not os.path.exists(path):
            print(f"missing document: {path}")
            continue
        pages = load_pdf(path)
        if not pages:
            print(f"no text extracted from {path}")
            continue
        chunks = chunk_data(pages)
        info = store.add_documents(chunks, source_name=os.path.basename(path))
        print(f"ingested {info['source_name']}: {info['chunk_count']} chunks, {info['page_count']} pages")

    if store.source_count() == 0:
        raise SystemExit("No documents ingested; refusing to write an empty evaluation run.")

    guard = LLMGuardrails(block_injections=True, mask_pii=True)
    evaluator = SemanticRAGEvaluator(min_faithfulness=0.50)
    evaluator.fit_baseline_corpus(store.all_texts())
    telemetry = LLMOpsTelemetry("llmops/reports/telemetry_audit_log.jsonl")

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{i}/{len(QUESTIONS)}] {question}")
        safe, sanitized, g_audit = guard.validate_input(question)
        if not safe:
            telemetry.log_interaction(
                question=question,
                response=sanitized,
                context_count=0,
                latency_ms=0.0,
                eval_metrics={
                    "faithfulness": None,
                    "answer_relevance": None,
                    "context_precision": None,
                    "rag_composite_quality": None,
                    "hallucination_detected": None,
                    "semantic_query_drift": None,
                    "drift_similarity_to_corpus": None,
                    "passed_quality_gate": None,
                    "skipped_reason": "guardrail_blocked",
                },
                guardrail_audit=g_audit,
                model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                mode="rag",
                is_answer=False,
                retrieved_context=[],
                tags={"source": "run_live_eval", "blocked": True},
            )
            print("  blocked by guardrail (logged, excluded from quality aggregates)")
            continue

        t0 = time.time()
        source_docs = []
        parts = []
        for kind, payload in stream_agent(sanitized, store, history=None):
            if kind == "token":
                parts.append(payload)
            else:
                source_docs = payload
        answer = "".join(parts).strip()
        latency_ms = (time.time() - t0) * 1000.0

        # Redact any PII the model reproduced before scoring or logging it.
        _, safe_answer, out_audit = guard.validate_output(answer)
        if out_audit.get("output_pii_detected"):
            answer = safe_answer
        g_audit = {**g_audit, **out_audit}

        ctx_chunks = [d.page_content for d in source_docs] if source_docs else []
        is_answer = bool(ctx_chunks) and bool(answer) and not is_non_answer(answer)
        mode = "rag" if ctx_chunks else "chat"

        if is_answer:
            metrics = evaluator.evaluate_turn(sanitized, answer, ctx_chunks)
        else:
            drift_flag, drift_sim = evaluator.detect_semantic_query_drift(sanitized)
            metrics = {
                "faithfulness": None,
                "answer_relevance": None,
                "context_precision": None,
                "rag_composite_quality": None,
                "hallucination_detected": None,
                "semantic_query_drift": drift_flag,
                "drift_similarity_to_corpus": drift_sim,
                "passed_quality_gate": None,
                "skipped_reason": "no_document_context_or_refusal",
            }

        retrieved_context = []
        for doc in (source_docs or [])[:8]:
            meta = getattr(doc, "metadata", {}) or {}
            retrieved_context.append(
                {
                    "source": meta.get("source") or "unknown",
                    "page": meta.get("page"),
                    "text": doc.page_content,
                }
            )

        telemetry.log_interaction(
            question=sanitized,
            response=answer,
            context_count=len(ctx_chunks),
            latency_ms=latency_ms,
            eval_metrics=metrics,
            guardrail_audit=g_audit,
            model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            mode=mode,
            is_answer=is_answer,
            retrieved_context=retrieved_context,
            tags={"source": "run_live_eval"},
        )
        faith = metrics.get("faithfulness")
        print(f"  is_answer={is_answer} chunks={len(ctx_chunks)} latency={latency_ms:.0f}ms "
              f"faith={faith if faith is None else round(faith, 3)} "
              f"halluc={metrics.get('hallucination_detected')}")
        print(f"  answer: {answer[:160]!r}")

    summary = telemetry.get_summary_statistics(mode="rag")
    print("\n=== real summary (mode=rag) ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
