"""
Interactive LLMOps Observability & Guardrails Dashboard
Run with: streamlit run llmops/dashboard.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd

# Ensure root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
AUDIT_LOG_PATH = os.path.join(PROJECT_ROOT, "llmops", "reports", "telemetry_audit_log.jsonl")

from llmops.guardrails import LLMGuardrails
from llmops.semantic_evaluator import SemanticRAGEvaluator
from llmops.telemetry import LLMOpsTelemetry

st.set_page_config(
    page_title="LLMOps Platform | AIM4142",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .status-pass {
        color: #22c55e;
        font-weight: bold;
    }
    .status-fail {
        color: #ef4444;
        font-weight: bold;
    }
    .status-warn {
        color: #f59e0b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session Components
def _telemetry_signature(log_path: str):
    """Cheap cache key so the drift baseline refits when new turns land."""
    if not os.path.exists(log_path):
        return (0, 0.0)
    stat = os.stat(log_path)
    return (stat.st_size, stat.st_mtime)


@st.cache_resource
def load_llmops_engine(log_signature):
    guardrails = LLMGuardrails(block_injections=True, mask_pii=True)
    evaluator = SemanticRAGEvaluator(min_faithfulness=0.50)
    telemetry = LLMOpsTelemetry(AUDIT_LOG_PATH)

    # Fit the semantic-drift baseline on the document chunks that really
    # produced logged answers. No hand-written reference corpus: if the log has
    # no stored context, drift stays unevaluated rather than reporting 0.0.
    baseline_docs = []
    for entry in telemetry.get_entries(mode="rag"):
        for chunk in entry.get("retrieved_context") or []:
            text = (chunk.get("text") or "").strip()
            if text:
                baseline_docs.append(text)
    if baseline_docs:
        evaluator.fit_baseline_corpus(baseline_docs)

    return guardrails, evaluator, telemetry

_audit_log_path = AUDIT_LOG_PATH
guardrails, evaluator, telemetry = load_llmops_engine(_telemetry_signature(_audit_log_path))

# Sidebar Navigation
st.sidebar.title("🛡️ LLMOps Platform")
view_mode = st.sidebar.radio(
    "Navigation Pane",
    ["📊 Real-Time Observability Dashboard", "🧪 Interactive Guardrail & Evaluation Playground", "📜 Historical Audit Telemetry"]
)

st.sidebar.divider()
st.sidebar.subheader("System Parameters")
faith_thresh = st.sidebar.slider("Min Faithfulness (Hallucination Cutoff)", 0.30, 0.90, 0.50, 0.05)
evaluator.min_faithfulness = faith_thresh

scope_label = "Real RAG chats"
data_scope = "rag"
st.sidebar.caption(
    "Metrics reflect evaluated document-chat turns only, computed from the "
    "append-only audit log. Metrics with no scored turns are shown as n/a."
)

# ----------------- VIEW 1: OBSERVABILITY DASHBOARD -----------------
if view_mode == "📊 Real-Time Observability Dashboard":
    st.title("📊 Production LLMOps Observability & Telemetry")
    st.write("Live operational monitoring, latency SLA profiling, hallucination detection rates, and safety gate metrics.")

    summary = telemetry.get_summary_statistics(mode=data_scope, min_faithfulness=faith_thresh)
    if summary.get("total_requests", 0) == 0:
        st.info(
            f"No evaluated entries found for scope: **{scope_label}**. "
            "Ask a question about an uploaded document to populate this view."
        )

    def _fmt_metric(value, suffix=""):
        if value is None:
            return "n/a"
        return f"{value}{suffix}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Evaluated RAG Queries", summary.get("total_requests", 0))
    with col2:
        st.metric("Average Latency (ms)", _fmt_metric(summary.get("avg_latency_ms"), " ms"))
    with col3:
        p95_ms = summary.get("p95_latency_ms")
        if p95_ms is None:
            st.metric("P95 End-to-End Latency", "n/a")
        else:
            st.metric(
                "P95 End-to-End Latency",
                f"{p95_ms} ms",
            )
    with col4:
        halluc_rate = summary.get("hallucination_rate_pct")
        if halluc_rate is None:
            st.metric("Hallucination Rate", "n/a")
        else:
            st.metric(
                "Hallucination Rate",
                f"{halluc_rate}%",
                delta=(
                    "Optimal (< 15%)" if halluc_rate < 15
                    else ("Safe (< 35%)" if halluc_rate < 35 else "Elevated (> 35%)")
                ),
                delta_color="normal" if halluc_rate < 35 else "inverse"
            )
    st.caption(
        f"Hallucination decisions are recomputed at the current faithfulness cutoff "
        f"of {faith_thresh:.2f} from each turn's stored score."
    )

    sample_n = summary.get("total_requests", 0)
    if 0 < sample_n < 20:
        st.caption(
            f"⚠️ P95 is unstable with only {sample_n} evaluated turn(s); it approximates the "
            "slowest query. Treat latency and rates as indicative until ~20+ real chats."
        )
    if sample_n and summary.get("scored_turns", 0) < sample_n:
        st.caption(
            f"{summary.get('scored_turns', 0)} of {sample_n} evaluated turn(s) carry a full "
            "quality score; averages exclude the rest instead of counting them as zero."
        )

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Average RAG Triad Quality Index")
        triad_rows = [
            ("Faithfulness (Groundedness)", summary.get("avg_faithfulness")),
            ("Answer Relevance", summary.get("avg_answer_relevance")),
            ("Composite Quality Score", summary.get("avg_rag_quality_score")),
        ]
        scored_rows = [(label, value) for label, value in triad_rows if value is not None]
        if scored_rows:
            metrics_df = pd.DataFrame(scored_rows, columns=["Metric", "Score"])
            st.bar_chart(metrics_df.set_index("Metric"))
        else:
            st.caption("No scored turns yet — there is nothing real to chart.")
        st.caption(
            "Relevance and precision are lexical proxies (TF-IDF + keyword coverage). "
            "Paraphrased summary answers legitimately score lower on word overlap."
        )

    with col_right:
        st.subheader("🛡️ Security & Guardrail Interceptions")
        sec_alerts = summary.get("security_alerts", {})
        sec_df = pd.DataFrame({
            "Category": ["Prompt Injections Intercepted", "PII Redactions Enforced"],
            "Incidents": [
                sec_alerts.get("prompt_injections_blocked", 0),
                sec_alerts.get("pii_redactions", 0)
            ]
        })
        st.bar_chart(sec_df.set_index("Category"))

    # No synthetic benchmark imagery is shown here. The observability view is
    # exclusively real document-chat telemetry.

    # ----------------- PER-TURN EVALUATION OF REAL CHATS -----------------
    st.divider()
    st.subheader("🔬 Per-Turn Evaluation of Real Chats")
    st.caption(
        "The latest 50 real chat turns, with the retrieved chunks each answer was "
        "grounded in and its sentence-level verification. Refusals and non-answers "
        "are listed but excluded from the aggregate rates above."
    )

    turns = telemetry.get_entries(mode=data_scope, limit=50)
    if not turns:
        st.info("No chat turns logged yet. Ask a question about an uploaded document.")
    else:
        evaluated_turns = [t for t in turns if t.get("is_answer", True) is not False]
        excluded_turns = [t for t in turns if t.get("is_answer", True) is False]

        e1, e2, e3 = st.columns(3)
        e1.metric("Turns Shown", len(turns))
        e2.metric("Evaluated", len(evaluated_turns))
        e3.metric("Excluded", len(excluded_turns))

        for idx, turn in enumerate(turns, start=1):
            m = turn.get("metrics") or {}
            is_evaluated = turn.get("is_answer", True) is not False
            faith_value = m.get("faithfulness")
            flagged = bool(
                faith_value is not None and float(faith_value) < faith_thresh
            )
            question = turn.get("question") or "(question not recorded)"
            stamp = str(turn.get("timestamp") or "")[:19].replace("T", " ")

            if is_evaluated:
                faith = m.get("faithfulness")
                faith_txt = f"faith {float(faith):.2f}" if faith is not None else "faith n/a"
                head = (
                    f"{stamp} · {question[:70]} · "
                    f"{faith_txt} · "
                    f"{'⚠️ flagged' if flagged else '✅ grounded'}"
                )
            else:
                reason = m.get("skipped_reason") or "not evaluated"
                head = f"{stamp} · {question[:70]} · excluded ({reason})"

            with st.expander(head):
                st.markdown(f"**Question:** {question}")

                answer_text = turn.get("response") or turn.get("response_preview") or ""
                st.markdown("**Answer:**")
                st.markdown(f"> {answer_text}" if answer_text else "_No answer recorded._")

                guardrail_audit = turn.get("guardrails") or {}
                if (
                    guardrail_audit.get("prompt_injection_detected")
                    or guardrail_audit.get("pii_detected")
                    or guardrail_audit.get("output_pii_leaked")
                ):
                    st.markdown("**Guardrail audit:**")
                    st.json(guardrail_audit)

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)

                def _turn_score(key):
                    value = m.get(key)
                    return f"{float(value):.2f}" if value is not None else "n/a"

                mc1.metric("Faithfulness", _turn_score("faithfulness") if is_evaluated else "—")
                mc2.metric("Relevance", _turn_score("answer_relevance") if is_evaluated else "—")
                mc3.metric("Precision", _turn_score("context_precision") if is_evaluated else "—")
                mc4.metric("Composite", _turn_score("rag_composite_quality") if is_evaluated else "—")
                mc5.metric("Latency", f"{float(turn.get('latency_ms') or 0):.0f} ms")

                drift_sim = m.get("drift_similarity_to_corpus")
                drift_flag = m.get("semantic_query_drift")
                if drift_sim is None:
                    st.caption("Semantic query drift: not evaluated (no document baseline fitted).")
                else:
                    st.caption(
                        f"Semantic query drift: similarity {float(drift_sim):.3f} to the "
                        f"document corpus — {'⚠️ out-of-domain' if drift_flag else 'in-domain'}."
                    )

                if not is_evaluated:
                    st.caption(
                        f"Excluded from quality aggregates: **{m.get('skipped_reason') or 'no document context or refusal'}**."
                    )

                st.markdown("**Retrieved context used for this answer:**")
                context = turn.get("retrieved_context") or []
                if not context:
                    st.caption(
                        "Context was not captured for this turn — it was logged before "
                        "retrieved-chunk storage was enabled."
                    )
                else:
                    for c_idx, chunk in enumerate(context, start=1):
                        src = chunk.get("source") or "unknown"
                        page = chunk.get("page")
                        label = f"[{c_idx}] {src}" + (f" · page {page}" if page is not None else "")
                        st.caption(label)
                        st.code(chunk.get("text") or "", language="text")

                sentences = m.get("sentence_verification") or []
                if sentences:
                    st.markdown("**Sentence-level grounding:**")
                    st.dataframe(
                        pd.DataFrame(sentences).rename(
                            columns={
                                "sentence": "Sentence",
                                "best_context_match_score": "Best match",
                                "grounded": "Grounded",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

# ----------------- VIEW 2: INTERACTIVE PLAYGROUND -----------------
elif view_mode == "🧪 Interactive Guardrail & Evaluation Playground":
    st.title("🧪 Live Interactive Evaluation & Guardrails Playground")
    st.write("Test incoming queries, trigger adversarial jailbreaks, inspect semantic grounding, and view real-time scoring.")

    st.caption(
        "Paste a real query, the chunks your retriever returned, and the model "
        "answer you want scored. Nothing is pre-filled and nothing is written to "
        "the production audit log."
    )

    with st.form("eval_form"):
        user_q = st.text_input("User Prompt / Query:", value="", placeholder="e.g. Summarize Section 2 of my uploaded document")
        retrieved_ctx = st.text_area("Retrieved Document Context Chunks (one per paragraph):", value="", height=100, placeholder="Paste or view the document chunk text here")
        model_ans = st.text_area("Model Output Response:", value="", height=100, placeholder="Model answer to be evaluated for hallucination")
        submitted = st.form_submit_button("🚀 Run Live LLMOps Inspection")

    if submitted and not (user_q.strip() and retrieved_ctx.strip() and model_ans.strip()):
        st.warning(
            "Enter the query, at least one retrieved context chunk, and the model "
            "answer before running an evaluation."
        )
        submitted = False

    if submitted:
        # Step 1: Guardrail Inspection
        is_safe, sanitized_query, guard_audit = guardrails.validate_input(user_q)

        st.subheader("1. Guardrail Security Audit")
        if not is_safe:
            st.error(f"🚫 REQUEST BLOCKED BY GUARDRAIL: {sanitized_query}")
            st.json(guard_audit)
        else:
            if guard_audit["pii_detected"]:
                st.warning(f"⚠️ PII Detected & Redacted: {sanitized_query}")
            else:
                st.success("✅ Input Query Passed Security Inspection (No Adversarial Injection / PII Clean)")

            # Step 2: Semantic Evaluation
            _, safe_model_ans, output_audit = guardrails.validate_output(model_ans)
            if output_audit["output_pii_detected"]:
                st.warning(
                    "⚠️ The model output contained PII and was redacted before scoring: "
                    f"{', '.join(output_audit['output_pii_types_found'])}"
                )
                model_ans = safe_model_ans

            ctx_list = [c.strip() for c in retrieved_ctx.split("\n") if c.strip()]
            eval_res = evaluator.evaluate_turn(sanitized_query, model_ans, ctx_list)

            st.subheader("2. Semantic RAG Triad Evaluation")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Faithfulness Score", f"{eval_res['faithfulness'] * 100:.1f}%")
            with m2:
                st.metric("Answer Relevance", f"{eval_res['answer_relevance'] * 100:.1f}%")
            with m3:
                st.metric("Context Precision", f"{eval_res['context_precision'] * 100:.1f}%")
            with m4:
                st.metric("Composite Quality", f"{eval_res['rag_composite_quality'] * 100:.1f}%")

            # Status Callouts
            if eval_res["hallucination_detected"]:
                st.error("🚨 HALLUCINATION DETECTED: The generated response contains claims unsubstantiated by retrieved context.")
            else:
                st.success("🎯 RESPONSE GROUNDED: Output is mathematically verified against retrieved context.")

            if eval_res.get("semantic_query_drift") is None:
                st.caption(
                    "Semantic query drift: not evaluated — no baseline has been fitted "
                    "from real audit-log context yet."
                )
            elif eval_res["semantic_query_drift"]:
                st.warning("⚠️ OUT-OF-DOMAIN QUERY DRIFT: This prompt falls outside the semantic centroid of baseline documents.")

            # Sentence Breakdown Table
            if eval_res.get("sentence_verification"):
                st.subheader("3. Sentence-by-Sentence Grounding Verification")
                st.dataframe(pd.DataFrame(eval_res["sentence_verification"]))

# ----------------- VIEW 3: HISTORICAL AUDIT TELEMETRY -----------------
elif view_mode == "📜 Historical Audit Telemetry":
    st.title("📜 Production Telemetry Audit Trail")
    st.write("Append-only JSONL telemetry log tracking latency, estimated token usage, and quality drift.")

    log_path = AUDIT_LOG_PATH
    if os.path.exists(log_path):
        records = []
        skipped = 0
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
        if skipped:
            st.warning(f"Skipped {skipped} malformed line(s) in the audit log.")

        if records:
            modes = ["All"] + sorted({r.get("mode", "legacy") for r in records})
            chosen_mode = st.selectbox("Filter by traffic mode", modes)
            if chosen_mode != "All":
                records = [r for r in records if r.get("mode", "legacy") == chosen_mode]
            df = pd.DataFrame(records)
            st.dataframe(df, width="stretch")
            st.download_button(
                label="📥 Download Telemetry Log (JSONL)",
                data="\n".join([json.dumps(r) for r in records]),
                file_name="telemetry_audit_log.jsonl",
                mime="application/json"
            )
        else:
            st.info("The audit log exists but has no well-formed rows yet.")
    else:
        st.info("No audit log yet. Ask a question about an uploaded document to create one.")
