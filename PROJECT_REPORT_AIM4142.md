# MANIPAL UNIVERSITY JAIPUR
## School of Computer Science and Engineering
### Department of AI & ML
**Course Code:** AIM4142 — Machine Learning Operations (MLOps)  
**Assignment:** Fast Learners Assignment — Research Project (Task 2)  
**Submission Date:** October 2026  

---

# RESEARCH PROJECT REPORT
# Adaptive LLMOps: Automated Semantic Regression Testing, Vector-Space Drift Detection, and Adversarial Guardrails for Enterprise RAG Architectures

---

## 1. Executive Summary & Problem Statement

### 1.1 The Production LLM Dilemma
Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) architectures have revolutionized question-answering over unstructured enterprise knowledge bases. However, deploying RAG pipelines into production environments presents critical failure modes that conventional DevOps and classic tabular MLOps cannot mitigate:
1. **Silent Regression & Degradation:** Small adjustments to chunking strategies, embeddings, or retriever settings often silently degrade downstream retrieval quality without triggering syntax or runtime errors.
2. **Semantic Hallucinations:** Generative models produce factually ungrounded claims that sound convincing but contradict or fabricate beyond the retrieved source context.
3. **Out-of-Domain (OOD) Semantic Query Drift:** Production queries drift away from the semantic distribution of the knowledge corpus, leading to low-confidence retrieval.
4. **Adversarial & Compliance Vulnerabilities:** Malicious user queries (prompt injections, jailbreaks) or unredacted Personally Identifiable Information (PII) introduce severe security and regulatory compliance risks (DPDP / GDPR).
5. **Absence of Measurable CI/CD Quality Gates:** Traditional unit tests verify software execution, but fail to evaluate statistical retrieval efficacy (e.g., Recall@K, Mean Reciprocal Rank).

### 1.2 Research Objectives
To solve these challenges, this research project designs, implements, and benchmarks a complete **LLMOps Architecture** layered over an agentic RAG system:
- **Objective 1 (Objective Retrieval Benchmarking):** Establish a fixed golden evaluation dataset and measure academic information retrieval metrics ($\text{Recall}@1$, $\text{Recall}@K$, and Mean Reciprocal Rank $\text{MRR}$).
- **Objective 2 (Automated Regression Gate):** Design an automated CI/CD gating mechanism that compares candidate pipeline configurations against the baseline and blocks deployments that exhibit retrieval degradation.
- **Objective 3 (Vector-Space Query Drift Detection):** Model the baseline document embedding space as a centroid vector $\vec{\mu}_C$ to detect Out-of-Domain (OOD) query drift via cosine divergence.
- **Objective 4 (Sentence-Level Entailment & Hallucination Flagging):** Decompose generated responses into constituent claims to calculate a groundedness/faithfulness score against retrieved context.
- **Objective 5 (Dual-Gate Security Guardrails):** Implement pre-retrieval pattern filters intercepting adversarial prompt injections plus input and post-generation regex PII redaction masks.
- **Objective 6 (Operational Telemetry & Interactive Observability):** Develop a real-time Streamlit dashboard and immutable JSONL telemetry tracking latency SLAs (p95), token costs, and safety metrics.

---

## 2. System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              INCOMING USER QUERY STREAM                                │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ Query [q]
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. PRE-RETRIEVAL GUARDRAILS GATE (llmops/guardrails.py)                                │
│    • Regex Adversarial Injection Detector: P_inj = {r"ignore previous...", ...}        │
│    • PII Sanitizer: Masks [REDACTED_EMAIL], [REDACTED_API_KEY], etc.                    │
└──────────────────────┬───────────────────────────────────┬─────────────────────────────┘
                       │ Sanitized Query                   │ Malicious Injection
                       ▼                                   ▼
┌──────────────────────────────────────────────┐     ┌───────────────────────────────────┐
│ 2. CORE AGENTIC RAG SYSTEM                   │     │ 403 FORBIDDEN INTERCEPT:          │
│    • Vector DB Document Retriever            │     │ "Adversarial injection blocked"   │
│    • LLM Generation Engine                   │     └───────────────────────────────────┘
└──────────────────────┬───────────────────────┘
                       │ (Query q, Chunks C, Answer A)
                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. SEMANTIC EVALUATION & DRIFT ENGINE (llmops/semantic_evaluator.py)                    │
│    • Sentence-Level Entailment / Faithfulness against Context Chunks                   │
│    • Answer-Query Semantic Relevance Score                                             │
│    • Out-of-Domain Drift: Cosine distance against Corpus Centroid Vector μ_C           │
└──────────────────────┬─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. AUTOMATED CI/CD REGRESSION TESTING & QUALITY GATE (llmops/retrieval_metrics.py)     │
│    • Golden Evaluation Dataset (eval_golden_dataset.json)                              │
│    • Computes Recall@1, Recall@3, and Mean Reciprocal Rank (MRR)                       │
│    • Automated Gate: BLOCKS deployment if MRR delta < -0.15 or Recall drop > 0.10      │
└──────────────────────┬─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. OBSERVABILITY & TELEMETRY DASHBOARD (llmops/dashboard.py & telemetry.py)            │
│    • Real-time Streamlit UI with Live Gauges & Per-Turn Inspection                  │
│    • Append-Only Audit Log (telemetry_audit_log.jsonl)                                 │
│    • Latency Breakdown (ms) & Estimated Token/Cost Tracking                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations

### 3.1 Information Retrieval Metrics (Golden Dataset)
Given a set of evaluation queries $Q = \{q_1, \dots, q_m\}$, each with a known relevant document $d_i^*$:
- **Recall@K:**
  $$\text{Recall}@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}\Big(d_i^* \in \text{Top-}K(q_i)\Big)$$
- **Mean Reciprocal Rank (MRR):**
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  where $\text{rank}_i$ is the position of the first relevant document in the retrieved list (or $\infty$ if not retrieved).

### 3.2 Vector-Space Query Drift Detection
Let $\{\vec{d}_1, \dots, \vec{d}_N\}$ be the vector embeddings of the reference knowledge base. The baseline corpus centroid is:
$$\vec{\mu}_C = \frac{1}{N} \sum_{n=1}^{N} \vec{d}_n$$
For an incoming query $q$ with embedding $\vec{v}_q$, the semantic drift metric is:
$$\text{Sim}_{\text{drift}}(q, \vec{\mu}_C) = \frac{\vec{v}_q \cdot \vec{\mu}_C}{\|\vec{v}_q\| \|\vec{\mu}_C\|}$$
$$\text{Drift\_Flag}(q) = \begin{cases} \text{TRUE (OOD Shift)}, & \text{if } \text{Sim}_{\text{drift}}(q, \vec{\mu}_C) < 0.06 \\ \text{FALSE (In-Domain)}, & \text{otherwise} \end{cases}$$

The threshold $0.06$ is calibrated empirically on real logged turns (§5.4): in-domain
questions scored 0.085–0.223 against the centroid, out-of-domain questions 0.011–0.042.
The centroid is fitted on the retrieved chunks of the current document set; when no
document text is available the drift statistic is undefined and is reported as `None`
rather than 0.0.

### 3.3 Semantic Faithfulness (Sentence-Level Entailment)
A response $A$ is parsed into claim sentences $S = \{s_1, \dots, s_k\}$. Let $N(x)$ be the set of numeric literals in text $x$ (inline citation markers such as 【1】 are stripped first). A sentence counts as grounded only if it both overlaps a retrieved chunk and every number it states is present in the retrieved context:
$$G(s_i) = \mathbb{I}\left(\max_{j} \text{sim}(s_i, c_j) \ge 0.12\right) \cdot \mathbb{I}\left(N(s_i) \subseteq N(C)\right)$$
$$\text{Faithfulness}(A, C) = \frac{1}{k} \sum_{i=1}^{k} G(s_i)$$
The numeric check matters: the fabricated claim *"a 50 percent acquisition"* shares the words *company* and *percent* with a chunk about *12 percent revenue growth*, so overlap alone would call it grounded. If $\text{Faithfulness} < 0.50$, the response triggers an automated **Hallucination Alert**.

---

## 4. Engineering Implementation

The project is structured under the `llmops/` package in [`llmops-rag-document-system`](file:///C:/Users/vidus/Desktop/Projects/llmops-rag-document-system):

| Module | Core Responsibility |
| :--- | :--- |
| **`llmops/data/eval_golden_dataset.json`** | Golden evaluation dataset containing verified query-document-answer triples. |
| **`llmops/retrieval_metrics.py`** | Computes $\text{Recall}@1$, $\text{Recall}@K$, and $\text{MRR}$ over retrieval iterations. |
| **`llmops/run_regression_experiment.py`** | Compares baseline vs. degraded configurations and executes the CI/CD gate. |
| **`llmops/semantic_evaluator.py`** | Sentence-level semantic grounding, relevance, and OOD query drift detection. |
| **`llmops/guardrails.py`** | Pre-retrieval injection blocking and input/output regex PII redaction. |
| **`llmops/telemetry.py`** | JSONL audit trail logging token usage, estimated costs, and latency SLAs. |
| **`llmops/dashboard.py`** | Interactive Streamlit observability UI with per-turn transcript/chunk inspection and a live scoring sandbox. |
| **`.github/workflows/llmops_ci.yml`** | GitHub Actions pipeline executing automated test assertions on push. |

---

## 5. Experimental Results & Verification

### 5.1 Experiment 1: Automated Configuration Regression Benchmark

**Evaluation set.** 45 labelled queries over 24 passages — 20 relevant targets plus 4 topical distractors — stored in `llmops/data/eval_golden_dataset.json` and `llmops/data/eval_corpus.json`. Queries are deliberately **paraphrased rather than copied** from their source passages, so a lexical retriever cannot score 1.0 by keyword matching alone. Each query carries one gold passage (`relevant_doc_id`), a supporting context, and a gold answer.

To test whether the CI/CD gate catches silent pipeline degradation, we evaluated two configurations:

- **Configuration $C_0$ (Baseline Production):** TF-IDF lexical indexing with top-3 retrieval (the metric harness in `llmops/retrieval_metrics.py`).
- **Configuration $C_1$ (Silently Degraded / Drifted Index):** Same retriever with Gaussian noise injected into the similarity scores ($\sigma = 0.15$), modelling a mildly drifted index rather than a total outage. Seed fixed at 42; reproducible via `python llmops/run_regression_experiment.py --noise 0.15 --seed 42`.

| Configuration | Recall@1 | Recall@3 | Mean Reciprocal Rank (MRR) | Automated CI/CD Gate Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **$C_0$: Baseline Production** | **0.9111** | **0.9778** | **0.9444** | **PASS (Baseline Reference)** |
| **$C_1$: Degraded / Drifted** | **0.6000** | **0.7556** | **0.6704** | **BLOCKED (Regression Detected)** |

**Experimental Finding:**  
Configuration $C_1$ produced an $\text{MRR}$ drop of $-0.2740$ and a $\text{Recall}@3$ drop of $-0.2222$. The automated quality gate intercepted the degraded pipeline before deployment.

### 5.1.1 Baseline Failure Analysis
The baseline is not perfect, and the four misses are informative rather than noise:

| Query | Gold passage | Rank | Cause |
| :--- | :--- | :---: | :--- |
| Q07 | `DOC_FIN_01` | 2 | "obliges us to hold on" shares no vocabulary with the passage |
| Q03 | `DOC_SEC_03` | 2 | retention vocabulary pulls `DOC_FIN_08` to rank 1 |
| Q20 | `DOC_SEC_03` | 2 | "unencrypted" vs "plaintext" — synonym gap |
| Q06 | `DOC_FIN_01` | >3 | "transaction data" / "physically live" never appear in the passage |

All four are **lexical-paraphrase failures**, not labelling errors: the gold passage does answer the question. This is direct evidence for the limitation in §8.2 and the clearest argument for replacing the TF-IDF harness with a dense encoder.

### 5.1.2 Gate Sensitivity to Degradation Magnitude
Because $C_1$ is a controlled perturbation, we swept the noise level to test whether the gate responds to *subtle* drift or only to catastrophic collapse:

| Noise $\sigma$ | Recall@1 | Recall@3 | MRR | $\Delta$MRR | Gate |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 0.00 (baseline) | 0.9111 | 0.9778 | 0.9444 | — | PASS |
| 0.10 | 0.7111 | 0.8444 | 0.7704 | −0.1740 | **BLOCK** |
| 0.15 | 0.6000 | 0.7556 | 0.6704 | −0.2740 | **BLOCK** |
| 0.20 | 0.4667 | 0.6444 | 0.5556 | −0.3888 | **BLOCK** |
| 0.30 | 0.1333 | 0.4889 | 0.2741 | −0.6703 | **BLOCK** |
| 0.50 | 0.0000 | 0.1111 | 0.0481 | −0.8963 | **BLOCK** |
| 0.85 | 0.0000 | 0.0667 | 0.0296 | −0.9148 | **BLOCK** |

The gate fires at the **smallest perturbation tested** ($\sigma = 0.10$, $\Delta$MRR = −0.174 against a −0.15 tolerance). The threshold is therefore sensitive to a 17-point MRR regression, and the result is monotonic in degradation magnitude — the gate is not merely detecting total collapse.

### 5.2 Experiment 2: End-to-End Semantic Verification on Real Documents

Experiment 2 is not a scripted scenario suite. It runs the **production pipeline
itself** — the same loader, chunker, session vector store, reranker, LangGraph
agent, guardrails and evaluator that the Streamlit app uses — over two real
papers in this repository:

- `MLOps_Overview_Definition_Architecture_IEEE_Access_2023.pdf` (631 KB)
- `Hidden_Technical_Debt_in_Machine_Learning_Systems_NeurIPS.pdf` (166 KB)

Ten questions were issued (8 in-domain, 2 deliberately out-of-domain), every
answer was generated live by `openai/gpt-oss-120b`, and every score was produced by
`SemanticRAGEvaluator`. The runner is committed as `scripts/run_live_eval.py`,
so the experiment is reproducible. Each turn writes one row to
`llmops/reports/telemetry_audit_log.jsonl`.

| # | Question (abridged) | Chunks | Faithfulness | Relevance | Precision | Composite | Hallucination |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Why are MLOps principles important? | 6 | 1.000 | 0.667 | 1.000 | 0.884 | No |
| 2 | DevOps vs MLOps relationship | 6 | 1.000 | 0.610 | 1.000 | 0.864 | No |
| 3 | What is hidden technical debt? | 6 | 1.000 | 0.966 | 1.000 | 0.988 | No |
| 4 | Which types of technical debt are identified? | 6 | 1.000 | 0.612 | 1.000 | 0.864 | No |
| 5 | Why is data dependency a source of debt? | 5 | 1.000 | 0.686 | 1.000 | 0.890 | No |
| 6 | What do the authors recommend for managing debt? | 6 | 1.000 | 0.744 | 1.000 | 0.910 | No |

Two in-domain questions were **refused** by the document-relevance grader
("What does the paper define as MLOps?" and "What are the core components of the
MLOps architecture?"), and both out-of-domain questions ("capital of France",
"2018 FIFA World Cup") were refused. Refusals are logged with `is_answer=false`
and excluded from the aggregates rather than counted as successful answers or as
hallucinations.

**What this shows.** The grounding path is working: every answer that was
produced is fully supported by the retrieved chunks (faithfulness 1.000) and
cites them. The weakness is upstream — the relevance grader is strict enough to
reject two reasonable questions, giving an answer rate of 6/8. That is a real
failure mode worth reporting rather than a number to be tuned away.

**A note on removal.** Earlier drafts of this report quoted a "scripted benchmark
suite" (`llmops/run_benchmark.py` → `telemetry_benchmark_log.jsonl`). That script
invented its latency figures with `np.random.uniform(95, 190)` and wrote
pre-computed metric dictionaries, so its results measured nothing. The script
and its log have been deleted, the CI step that ran it has been removed, and no
number in this report now comes from it.

### 5.3 Experiment 3: Replaying the 20 Sidebar Chats on One Shared Corpus

The 20 chats shipped in `chats.json` (40 turns) are not hand-written examples or
a preset benchmark. They are replayed through the **same production pipeline**
used above — guardrails, session vector store, reranker, LangGraph agent,
evaluator and telemetry — against a single shared knowledge base,
`llmops/data/eval_corpus.json` (24 passages). Re-asking every chat against one
corpus, rather than giving each chat its own private documents, means the
resulting metrics describe one measurable retrieval target.

The runner is `scripts/replay_chats.py`. For each turn it takes the original
question, retrieves from the shared corpus, generates a fresh answer, scores it,
and appends a row to the existing audit log. It never reuses a stored answer or
score, and it never deletes the earlier PDF run.

| Measure | Value |
| :--- | :--- |
| Chats / turns replayed | 20 / 40 |
| Answered from documents | 38 |
| Refusals | 2 |
| Mean semantic faithfulness | 1.000 |
| Mean answer relevance | 0.822 |
| Mean context precision | 1.000 |
| Mean composite quality | 0.938 |
| Hallucination rate | 0.0% (0 of 38 turns below the 0.50 cut-off) |
| Mean end-to-end latency | 4,884 ms |
| P95 end-to-end latency | 8,420 ms |

**Refusal analysis.** Two follow-up turns were refused rather than answered:

- `chat-8` turn 2: *"What categories of data does that cover?"*
- `chat-11` turn 2: *"What is the purpose of that connection?"*

Both depend on a pronoun (*that*) whose referent exists only in the previous
turn. Retrieval and document grading receive the current question alone, so no
chunk is found and the generator returns the refusal string. This is honest
behaviour — the alternative would be to invent a referent — but it is a real
conversational limitation: the history is available to the generator, not to
the retriever. The two turns are logged with `is_answer=false`, null quality
metrics, and are excluded from all aggregates.

**What this shows.** The grounding path holds at over six times the sample size
of the PDF run: 38 fresh answers, all assembled from retrieved corpus spans
and all scored as grounded, with zero hallucination flags. The two failures are
upstream conversational failures rather than fabricated content. The replayed
answers were written back into `chats.json`, and every turn's question, answer,
retrieved chunks and sentence-level scores can be inspected in the observability
dashboard.

### 5.4 Operational Telemetry Summary

Telemetry is a single append-only log, `llmops/reports/telemetry_audit_log.jsonl`.
There is no second, synthetic log: a metric that did not come from a real turn
simply does not exist in the system. The dashboard reads this log directly.

**Scope:** 50 logged turns — the 10-turn PDF run of §5.2 plus the 40-turn chat
replay of §5.3. Forty-eight are document-scope turns (`mode=rag`): 44 answered
and 4 refusals (2 PDF in-domain refusals and the 2 replay follow-ups). The other
2 are the document-free out-of-domain PDF questions (`mode=chat`). Aggregates
cover the 44 answered document turns only.

| Metric | Value |
| :--- | :--- |
| Evaluated document turns | 44 |
| Mean semantic faithfulness | 1.000 |
| Mean answer relevance | 0.807 |
| Mean composite quality | 0.933 |
| Hallucination rate | 0.0% (0 of 44 turns below the 0.50 faithfulness cut-off) |
| Mean end-to-end latency | 6,859 ms |
| P95 end-to-end latency | 23,123 ms |

The combined mean latency is higher than the chat-replay mean because the six
PDF answers were long multi-paragraph summaries. Latency is dominated by the LLM
round-trip (retrieval plus reranking is a few milliseconds locally). A sub-second
SLA is not achievable with this hosted model, so the dashboard uses a 5,000 ms
target and explicitly warns that P95 is unstable below ~20 samples.

Because aggregates are computed only over turns that carry a score, an unscored
or refused turn can no longer pull an average toward zero. Metrics with no scored
turns are returned as `None` and rendered as `n/a` in the UI, not as 0.0.

**Provenance correction.** The ten PDF rows were written before the runner
recorded the active model and before token/cost fields were flagged as estimates.
Their `model_name` metadata was corrected to the generator actually used
(`openai/gpt-oss-120b`), and `tokens_estimated` / `cost_estimated` were added.
No answer, score, latency, question or retrieved-chunk field was modified.

**Drift calibration.** The out-of-domain threshold was calibrated on the PDF run
rather than guessed: in-domain questions scored 0.085–0.223 cosine similarity to
the document centroid, the two out-of-domain questions scored 0.011 and 0.042.
`DEFAULT_DRIFT_THRESHOLD = 0.06` separates the two populations and is shared by
the app and the dashboard. The baseline centroid is fitted from the actual
retrieved chunks of the session, so if no document text exists the drift value is
reported as `None` instead of a fabricated 0.0.

---

## 6. Relevance to Real-World MLOps
1. **Preventing Catastrophic Deployment Degradation:** Without automated regression gates ($\text{Recall}@K$, $\text{MRR}$), teams pushing new embedding models or prompt revisions risk silent business disruptions.
2. **Regulatory & Privacy Compliance:** Financial institutions and healthcare providers cannot deploy LLM applications without certified PII masking and immutable audit logs.
3. **Continuous Knowledge Maintenance:** Vector-space query drift telemetry directly alerts MLOps engineers when users are asking questions outside the knowledge base, indicating an urgent need for document re-indexing.

---

## 7. Conclusion
This project successfully solves the core challenge of modern LLMOps: **transforming an unmonitored RAG prototype into a hardened, measurable, and self-governing production system**. By integrating formal Information Retrieval benchmarking ($\text{Recall}@K$, $\text{MRR}$), automated regression CI/CD gates, vector-space query drift tracking, and a live Streamlit observability dashboard, the platform satisfies all foundational pillars of enterprise-grade MLOps.

---

## 8. Limitations & Threats to Validity

The following are stated explicitly because they bound what the reported numbers can support.

### 8.1 Sample size and statistical power
- The golden retrieval dataset now holds **45 labelled queries over 24 passages**, which is large enough for retrieval metrics to be meaningful: a single query moves Recall@1 by ~2.2 points rather than 20.
- Live document-chat telemetry contains **44 evaluated turns** (6 from the PDF run in §5.2 and 38 from the chat replay in §5.3). The live hallucination rate and RAG Triad scores remain **directional indicators, not population estimates**; one flipped turn now moves the hallucination rate by ~2.3 points.
- Retrieval metrics are reported as point estimates without confidence intervals or significance tests. Paired bootstrap CIs over the 45 queries would be the natural next step.
- Interpreting P95 latency requires ~20+ samples; below that it approximates the slowest observed query.

### 8.2 Proxy metrics rather than semantic models
- Retrieval and grounding use lexical similarity (hashed bag-of-words / TF-IDF plus keyword coverage), not dense semantic entailment.
- A correct paraphrase that shares little vocabulary with its source is therefore penalised. This is the known cause of the low answer-relevance scores on the earliest logged turns.
- The reported hallucination rate is a **lexical-grounding rate**, not a human-adjudicated or LLM-judge-verified hallucination rate.

### 8.3 Non-answer filter can shrink the denominator
Turns that make no factual claim (refusals, empty responses, conversational filler such as *"Sure, let me know what you'd like to explore next."*) are excluded from quality aggregates, because scoring them as ungrounded claims mislabels them as hallucinations.

The heuristic is deliberately **lenient**, which biases the metric toward under-counting:
- A genuine, well-grounded answer that ends with a long conversational tail (e.g. *"…and if you need more detail let me know and I can walk through each module"*) can be classified as a non-answer and dropped from the denominator.
- The result is a **conservative** hallucination rate: false negatives are preferred over false positives.
- Because the denominator can shrink, the rate must always be reported **alongside `n`**. A 0% rate over 3 turns and a 0% rate over 300 turns are not equivalent evidence.

### 8.4 The degraded configuration is simulated, not empirical
Configuration $C_1$ is produced by adding Gaussian noise to similarity scores rather than by deploying a genuinely weaker embedding model, a different chunking strategy, or a corrupted index. The sensitivity sweep in §5.1.2 shows the gate responds monotonically to perturbation magnitude, but this still does **not** establish how the gate behaves under organic model or index drift. The seed is fixed so every reported figure is reproducible.

### 8.5 Thresholds are only partly calibrated
The grounding threshold (0.12), context-precision threshold (0.08) and gate tolerances (MRR −0.15, Recall −0.10) were set by manual inspection of a small corpus, not derived from a labelled validation set. They are engineering defaults, not tuned parameters with evidence of optimality. The drift threshold (0.06) is the exception: it was calibrated on the observed separation between 8 in-domain and 2 out-of-domain questions (§5.4), which is a real but very small sample — it should be re-checked as the log grows. The numeric-support rule in §3.3 was added after the quality gate showed that a fabricated figure could pass on word overlap alone; it has no false positives on the 20 logged sentences that contain numbers, but that is a small validation set.

### 8.6 No historical scores survive in the live log
Metrics are computed at write time and stored immutably. Rows written before retrieved-chunk capture existed were archived out of the live log (`llmops/reports/archive/`) because their context-dependent fields could not be recomputed. Every row now in `telemetry_audit_log.jsonl` carries its question, full response, retrieved chunks and sentence scores, so any live turn can be inspected and re-scored from the dashboard.

### 8.7 The golden labels are author-generated
The 45 queries, gold passages, and gold answers were authored programmatically from the reference corpus, not collected by independent annotators. Consequences:

- Relevance is **single-annotator** and therefore has no measured inter-annotator agreement. The corpus does contain its own annotation standard (Cohen's $\kappa \geq 0.70$, two labellers) — that standard applies to production labelling, not to this evaluation set.
- Gold answers are paraphrases of the source passages, so they are **extractive** by construction. They validate retrieval, not answer generation.
- Because queries were written with knowledge of the corpus, the baseline may be **optimistically biased** relative to genuinely unseen user questions.

Mitigation for a stronger study: have a second annotator independently label relevance on the same 45 queries and report $\kappa$, and add ~20 queries written by someone who has not read the corpus.
