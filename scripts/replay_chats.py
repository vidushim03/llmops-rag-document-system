"""Replay the sidebar conversations through the real RAG pipeline.

Every turn in ``chats.json`` is re-asked against a single shared knowledge base
built from ``llmops/data/eval_corpus.json`` (all 24 passages, one corpus for all
chats). The answers, retrieved chunks, latency and quality scores are produced
live by the same components the Streamlit app uses:

    SessionVectorStore -> retrieve_docs/rerank_docs -> LangGraph agent
    -> LLMGuardrails -> SemanticRAGEvaluator -> LLMOpsTelemetry

Nothing is mocked: if a turn is refused, the refusal is logged as
``is_answer=false`` and stays out of the aggregates. At the end the real model
answers are written back into ``chats.json`` so the sidebar shows the transcripts
that actually produced the telemetry.

Usage:
    .venv\\Scripts\\python.exe scripts\\replay_chats.py --fresh
    .venv\\Scripts\\python.exe scripts\\replay_chats.py --only 21247b --log scratch.jsonl
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document

from src.agent import stream_agent
from src.embedding import get_embeddings
from src.session_store import SessionVectorStore
from llmops.guardrails import LLMGuardrails
from llmops.semantic_evaluator import SemanticRAGEvaluator
from llmops.telemetry import LLMOpsTelemetry
from llmops.text_utils import is_non_answer

CORPUS_PATH = ROOT / "llmops" / "data" / "eval_corpus.json"
CHATS_PATH = ROOT / "chats.json"
DEFAULT_LOG = "llmops/reports/telemetry_audit_log.jsonl"
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def build_store(corpus):
    """Load every corpus passage into one shared in-memory store."""
    store = SessionVectorStore(get_embeddings(mode="transformer"))
    for doc_id, passage in corpus.items():
        text = (passage or "").strip()
        if not text:
            continue
        store.add_documents(
            [Document(page_content=text, metadata={"source": doc_id})],
            source_name=doc_id,
        )
    return store


def archive_existing(log_path: Path):
    """Never silently destroy a real run: move it to reports/archive."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    archive_dir = ROOT / "llmops" / "reports" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    target = archive_dir / f"{log_path.stem}.superseded_{stamp}{log_path.suffix}"
    shutil.move(str(log_path), str(target))
    return target


def run_turn(question, store, evaluator, guardrails, telemetry, history, chat_id, turn_index):
    safe, sanitized, audit = guardrails.validate_input(question)
    if not safe:
        print(f"  blocked by guardrail: {sanitized}")
        # Blocked replay turns must still appear in the audit log, otherwise the
        # security counters would miss exactly the traffic they are meant for.
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
            guardrail_audit=audit,
            model_name=MODEL_NAME,
            mode="rag",
            is_answer=False,
            retrieved_context=[],
            tags={
                "chat_id": chat_id,
                "turn_index": turn_index,
                "replay": True,
                "blocked": True,
            },
        )
        return {
            "answer": sanitized,
            "metrics": {"skipped_reason": "guardrail_blocked"},
            "is_answer": False,
            "chunks": [],
            "latency_ms": 0.0,
            "blocked": True,
        }

    started = time.time()
    parts = []
    source_docs = []
    for kind, payload in stream_agent(sanitized, store, history=history):
        if kind == "token":
            parts.append(payload)
        else:
            source_docs = payload or []
    latency_ms = (time.time() - started) * 1000.0
    answer = "".join(parts).strip()

    # Redact PII the model reproduced before scoring, storing or displaying it.
    _, safe_answer, out_audit = guardrails.validate_output(answer)
    if out_audit.get("output_pii_detected"):
        answer = safe_answer
    audit = {**audit, **out_audit}

    ctx_chunks = [d.page_content for d in source_docs] if source_docs else []
    is_answer = bool(ctx_chunks) and bool(answer) and not is_non_answer(answer)

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
    for doc in source_docs[:8]:
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
        guardrail_audit=audit,
        model_name=MODEL_NAME,
        mode="rag",
        is_answer=is_answer,
        retrieved_context=retrieved_context,
        tags={"chat_id": chat_id, "turn_index": turn_index, "replay": True},
    )

    faith = metrics.get("faithfulness")
    print(
        f"  is_answer={is_answer} chunks={len(ctx_chunks)} "
        f"latency={latency_ms:.0f}ms faith={faith if faith is None else round(faith, 3)} "
        f"halluc={metrics.get('hallucination_detected')}"
    )
    print(f"  answer: {answer[:180]!r}")
    return {
        "answer": answer,
        "metrics": metrics,
        "is_answer": is_answer,
        "chunks": retrieved_context,
        "latency_ms": latency_ms,
        "blocked": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="archive the existing audit log first")
    parser.add_argument("--only", default="", help="comma-separated chat ids to replay")
    parser.add_argument("--log", default=DEFAULT_LOG, help="telemetry log path")
    parser.add_argument("--no-chat-rewrite", action="store_true", help="do not rewrite chats.json")
    args = parser.parse_args()

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    chats = json.loads(CHATS_PATH.read_text(encoding="utf-8"))
    selected = [c.strip() for c in args.only.split(",") if c.strip()]
    if selected:
        missing = [c for c in selected if c not in chats]
        if missing:
            raise SystemExit(f"unknown chat id(s): {missing}")
        chats = {c: chats[c] for c in selected}

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        archived = archive_existing(log_path)
        if archived:
            print(f"archived previous log -> {archived.relative_to(ROOT)}")
        log_path.write_text("", encoding="utf-8")

    store = build_store(corpus)
    if store.source_count() == 0:
        raise SystemExit("shared corpus is empty; refusing to replay")

    guardrails = LLMGuardrails(block_injections=True, mask_pii=True)
    evaluator = SemanticRAGEvaluator(min_faithfulness=0.50)
    evaluator.fit_baseline_corpus(store.all_texts())
    if not evaluator.is_fitted:
        raise SystemExit("drift baseline failed to fit; refusing to emit unscored rows")
    telemetry = LLMOpsTelemetry(str(log_path))

    print(
        f"shared corpus: {store.source_count()} sources / {len(store.all_texts())} chunks | "
        f"chats: {len(chats)} | model: {MODEL_NAME}"
    )

    rewritten = {}
    total = 0
    answered = 0
    for chat_id, chat in chats.items():
        messages = chat.get("messages") or []
        questions = [m for m in messages if m.get("role") == "user"]
        print(f"\n=== {chat_id} | {chat.get('title', 'Untitled')} ({len(questions)} turns) ===")
        history = []
        new_messages = []
        for turn_index, message in enumerate(questions, start=1):
            question = (message.get("content") or "").strip()
            if not question:
                continue
            total += 1
            print(f"[{total}] {question}")
            result = run_turn(
                question,
                store,
                evaluator,
                guardrails,
                telemetry,
                list(history),
                chat_id,
                turn_index,
            )
            if result["is_answer"]:
                answered += 1
            new_messages.append({"role": "user", "content": question})
            new_messages.append({"role": "assistant", "content": result["answer"]})
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result["answer"]})
        rewritten[chat_id] = {
            "title": chat.get("title", "Untitled"),
            "corpus": CORPUS_PATH.relative_to(ROOT).as_posix(),
            "messages": new_messages,
        }

    if not args.no_chat_rewrite and not selected:
        tmp = CHATS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CHATS_PATH)
        print(f"\nrewrote {CHATS_PATH.name} from the real transcripts")

    summary = telemetry.get_summary_statistics(mode="rag")
    print(f"\nturns replayed: {total} | answered from documents: {answered} | refused: {total - answered}")
    print("=== live summary (mode=rag, answered turns only) ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    try:
        shown = log_path.relative_to(ROOT)
    except ValueError:
        shown = log_path
    print(f"\naudit log: {shown}")


if __name__ == "__main__":
    main()
