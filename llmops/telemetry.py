"""LLMOps Telemetry, Latency, and Cost Observability Engine
Maintains an append-only audit trail of:
- Execution latency (retrieval vs generation)
- Token consumption and estimated cost
- Quality scores & hallucination alerts

Only real turns are recorded. There is no synthetic traffic sink: scripts and
demos that used to write invented rows were removed.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

REAL_RAG_MODE = "rag"


def _mean_or_none(values: List[float]) -> Optional[float]:
    """Average over the values that exist, or None when nothing was scored."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


class LLMOpsTelemetry:
    def __init__(self, log_path: str = "llmops/reports/telemetry_audit_log.jsonl"):
        # Always resolve path relative to the project root (where app.py lives)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if not os.path.isabs(log_path):
            self.log_path = os.path.join(project_root, log_path)
        else:
            self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_interaction(
        self,
        question: str,
        response: str,
        context_count: int,
        latency_ms: float,
        eval_metrics: Dict[str, Any],
        guardrail_audit: Dict[str, Any],
        model_name: Optional[str] = None,
        mode: str = REAL_RAG_MODE,
        is_answer: bool = True,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record telemetry entry for a query-response pair.

        Args:
            mode: Traffic source. "rag" for document-grounded turns and
                "chat" for turns with no document context.
            is_answer: False for refusals/empty/error turns that were never
                evaluated. These are excluded from aggregate quality metrics.
            retrieved_context: The actual chunks the answer was grounded in,
                stored so a turn can be inspected and re-scored later.
        """
        # Approximate token count (1 token ~= 4 chars). These are labelled as
        # estimates in the stored row so they are never mistaken for usage
        # reported by the provider.
        prompt_tokens = int(len(question) / 4) + context_count * 120
        completion_tokens = int(len(response) / 4)
        total_tokens = prompt_tokens + completion_tokens

        # Estimated cost (based on standard Groq/OpenAI cloud pricing tier)
        cost_usd = round((prompt_tokens * 0.00000059) + (completion_tokens * 0.00000079), 7)

        stored_context = []
        for chunk in (retrieved_context or [])[:8]:
            stored_context.append(
                {
                    "source": str(chunk.get("source") or "unknown")[:200],
                    "page": chunk.get("page"),
                    "text": str(chunk.get("text") or "")[:1200],
                }
            )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "mode": mode,
            "is_answer": bool(is_answer),
            "latency_ms": round(latency_ms, 2),
            "context_count": context_count,
            "question": (question or "")[:300],
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "tokens_estimated": True,
            "cost_usd": cost_usd,
            "cost_estimated": True,
            "response_preview": (response or "")[:280],
            "response": (response or "")[:6000],
            "retrieved_context": stored_context,
            "metrics": eval_metrics,
            "guardrails": guardrail_audit
        }

        if tags:
            entry["tags"] = dict(tags)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load all well-formed JSONL rows, skipping corrupt lines."""
        if not os.path.exists(self.log_path):
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    @staticmethod
    def _matches_scope(entry: Dict[str, Any], mode: Optional[str]) -> bool:
        if mode is None:
            return True
        entry_mode = entry.get("mode")
        if entry_mode is None:
            # Legacy rows written before mode tagging.
            return mode == "legacy"
        return entry_mode == mode

    @staticmethod
    def _metric(entry: Dict[str, Any], *keys: str, default=None):
        """Return the first present, numeric metric value, or `default`.

        Missing metrics are reported as None rather than 0.0 so that unscored
        turns are excluded from averages instead of silently dragging them down.
        """
        metrics = entry.get("metrics") or {}
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return default

    def get_mode_breakdown(self) -> Dict[str, int]:
        """Return entry counts grouped by traffic mode."""
        counts: Dict[str, int] = {}
        for entry in self._load_entries():
            mode = entry.get("mode", "legacy")
            counts[mode] = counts.get(mode, 0) + 1
        return counts

    def get_entries(
        self,
        mode: Optional[str] = REAL_RAG_MODE,
        only_answers: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return individual turns, newest first, for per-turn inspection.

        Unlike get_summary_statistics, this does not filter out excluded turns
        by default, so the UI can show refusals and skipped turns alongside the
        evaluated ones.
        """
        entries = [e for e in self._load_entries() if self._matches_scope(e, mode)]
        if only_answers:
            entries = [e for e in entries if e.get("is_answer", True) is not False]
        entries.reverse()
        if limit is not None:
            entries = entries[:limit]
        return entries

    def get_summary_statistics(
        self,
        mode: Optional[str] = REAL_RAG_MODE,
        only_answers: bool = True,
        min_faithfulness: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute aggregated observability metrics.

        By default this reports on real RAG document chats only, so synthetic
        hallucination stress-tests and skipped/refused turns cannot inflate the
        headline hallucination rate. Pass mode=None for all traffic.

        Aggregates are computed only over turns that actually carry a score.
        A metric with no scored turns is returned as None (unavailable) instead
        of 0.0, so the UI can never present a fabricated zero as a measurement.

        When ``min_faithfulness`` is supplied, the hallucination decision is
        recomputed from each turn's stored faithfulness score at that cutoff.
        This keeps the dashboard's cutoff slider consistent with the displayed
        rate instead of silently using the cutoff that was active at write time.
        """
        all_entries = [
            e for e in self._load_entries()
            if self._matches_scope(e, mode)
        ]
        entries = all_entries
        if only_answers:
            entries = [e for e in entries if e.get("is_answer", True) is not False]

        scope_label = mode or "all"

        # Guardrail counts cover every turn in scope, including blocked and
        # refused turns, because that is exactly the traffic guardrails act on.
        injection_attempts = 0
        pii_leaks_prevented = 0
        for e in all_entries:
            guardrails = e.get("guardrails") or {}
            if guardrails.get("prompt_injection_detected", False):
                injection_attempts += 1
            if (
                guardrails.get("pii_detected", False)
                or guardrails.get("output_pii_leaked", False)
            ):
                pii_leaks_prevented += 1

        empty_summary = {
            "total_requests": 0,
            "data_scope": scope_label,
            "avg_latency_ms": None,
            "p95_latency_ms": None,
            "avg_faithfulness": None,
            "avg_answer_relevance": None,
            "avg_rag_quality_score": None,
            "hallucination_rate_pct": None,
            "hallucination_threshold": min_faithfulness,
            "scored_turns": 0,
            "hallucination_decisions": 0,
            "security_alerts": {
                "prompt_injections_blocked": injection_attempts,
                "pii_redactions": pii_leaks_prevented
            }
        }
        if not entries:
            return empty_summary

        total = len(entries)
        latencies = sorted(
            float(e.get("latency_ms") or 0.0) for e in entries
        )
        faithfulness_scores = self._collect(entries, "faithfulness")
        relevance_scores = self._collect(entries, "answer_relevance")
        quality_scores = self._collect(entries, "rag_composite_quality", "rag_quality_score")

        hallucination_count = 0
        hallucination_decisions = 0
        for e in entries:
            metrics = e.get("metrics") or {}
            if min_faithfulness is not None:
                faith = self._metric(e, "faithfulness")
                decision = None if faith is None else bool(faith < min_faithfulness)
            else:
                decision = metrics.get("hallucination_detected", metrics.get("hallucination_flag"))
            if decision is not None:
                hallucination_decisions += 1
                if decision:
                    hallucination_count += 1

        # Nearest-rank P95 without off-by-one overflow.
        p95_index = min(total - 1, max(0, int(0.95 * total) - 1))

        return {
            "total_requests": total,
            "data_scope": scope_label,
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p95_latency_ms": round(latencies[p95_index], 2),
            "avg_faithfulness": _mean_or_none(faithfulness_scores),
            "avg_answer_relevance": _mean_or_none(relevance_scores),
            "avg_rag_quality_score": _mean_or_none(quality_scores),
            "hallucination_rate_pct": (
                round((hallucination_count / hallucination_decisions) * 100, 2)
                if hallucination_decisions else None
            ),
            "hallucination_threshold": min_faithfulness,
            "scored_turns": len(faithfulness_scores),
            "hallucination_decisions": hallucination_decisions,
            "security_alerts": {
                "prompt_injections_blocked": injection_attempts,
                "pii_redactions": pii_leaks_prevented
            }
        }

    def _collect(self, entries: List[Dict[str, Any]], *keys: str) -> List[float]:
        """Gather only the present values for a metric across turns."""
        values = []
        for entry in entries:
            value = self._metric(entry, *keys)
            if value is not None:
                values.append(value)
        return values
