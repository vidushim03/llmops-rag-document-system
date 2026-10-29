[Project_Report.docx](https://github.com/user-attachments/files/34125030/Project_Report.docx)# LLMOps: Production RAG Evaluation, Guardrails & Observability Platform

[![LLMOps CI/CD Quality Gate](https://img.shields.io/badge/CI%2FCD-Quality%20Gate%20Passing-success)](.github/workflows/llmops_ci.yml)
[![MLOps Track](https://img.shields.io/badge/AIM4142-Fast%20Learners%20Project-blue)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()

> Academic Research Project for **AIM4142 Machine Learning Operations (MLOps)**
>
> **School of Computer Science and Engineering, Manipal University Jaipur**

---

## 🚀 Overview

This repository extends an Agentic Retrieval-Augmented Generation (RAG) system with an enterprise-grade **LLMOps Architecture**, addressing four critical production failure modes:
1. **Hallucination & Faithfulness Drift Detection** using the formal **RAG Triad** framework.
2. **Pre-Retrieval & Post-Generation Safety Guardrails** (Adversarial Prompt Injections & PII Masking).
3. **Real-time Observability & Telemetry** (Estimated token usage, estimated cost, and P95 latency monitoring).
4. **Automated CI/CD Quality Gates** with GitHub Actions preventing degraded models from deploying.

---

## 📁 Repository Structure

```
llmops-rag-document-system/
├── llmops/
│   ├── guardrails.py              # Prompt Injection & PII Sanitization Engine
│   ├── semantic_evaluator.py      # RAG Triad scoring used by the live app
│   ├── telemetry.py               # Latency, Estimated Token Cost & Observability Logging
│   ├── test_quality_gate.py       # CI/CD Automated Unit Test Suite
│   ├── run_regression_experiment.py # Retrieval benchmark + regression gate
│   └── reports/
│       └── telemetry_audit_log.jsonl    # Append-only Production Audit Trail
├── scripts/
│   ├── run_live_eval.py           # Real PDF evaluation run
│   └── replay_chats.py            # Replays sidebar chats against one shared corpus
├── .github/workflows/
│   └── llmops_ci.yml              # GitHub Actions Automated CI/CD Workflow
├── src/                           # Core LangGraph RAG Agent Codebase
├── chats.json                     # 20 real sidebar chats (replayed transcripts)
├── PROJECT_REPORT_AIM4142.md      # Full Academic Project Report for Submission
└── README.md                      # Documentation & Instructions
```

---

## ⚡ Quick Start & Reproduction

### 1. Run the CI/CD Quality Gate (Unit Tests)
Verify that security guardrails and hallucination rejection tests pass:
```powershell
python llmops\test_quality_gate.py
```
*Expected Output:* `Ran 9 tests in 0.3s — OK`

The suite exercises the production evaluator (`SemanticRAGEvaluator`) and the
telemetry aggregator, not a legacy duplicate: it checks injection blocking, PII
redaction (email, API key, SSN, and model-output leaks), grounded-answer
faithfulness, detection of a fabricated numeric claim, drift reporting "not
evaluated" until a real baseline is fitted (and again after a reset), that
blocked/PII turns still count as security incidents, and that the displayed
hallucination rate follows the current cutoff.

### 2. Run the Retrieval Regression Gate
Evaluate retrieval quality on the labelled benchmark and verify the gate blocks a degraded configuration:
```powershell
python llmops\run_regression_experiment.py
```
This will:
- Score a baseline configuration (C0) with Recall@K and MRR.
- Score a perturbed configuration (C1) and check the MRR / Recall@3 tolerances.
- Print a PASS / BLOCKED decision for the CI/CD gate.

### 3. Generate Real Observability Data
The dashboard has no seeded or scripted numbers. Every metric it shows comes from
`llmops/reports/telemetry_audit_log.jsonl`, which only receives entries when you
actually chat. Every turn is logged — answered, refused, or guardrail-blocked —
but only answers grounded in a document enter the quality aggregates.

To reproduce the logged PDF run reported in §5.2 of the report (two real papers
from `research_paper_submission/`, live Groq answers, live scoring):

```powershell
python scripts\run_live_eval.py
```

### 4. Replay the 20 Sidebar Chats
The 20 chats in `chats.json` are re-asked turn by turn through the real pipeline
(guardrails → retrieval → generation → scoring → telemetry). Every chat uses the
**same** knowledge base, `llmops/data/eval_corpus.json` (24 passages), so the
resulting metrics describe one corpus rather than 20 private ones:

```powershell
python scripts\replay_chats.py
```

The script:
- retrieves, generates and scores each turn live (no pre-computed answers),
- appends its turns to the same audit log, so the earlier PDF run is preserved,
- rewrites `chats.json` with the actual model answers,
- logs refused turns as `is_answer=false`, excluded from quality aggregates.

Pass `--fresh` to archive the current log before replaying; the default appends.

Open a replayed chat in the sidebar and press **Load shared corpus for follow-ups**
to ask new questions against the same passages.

Both scripts append. They never erase an existing run.

---

## 📊 Key Results

All numbers below are produced by scripts in this repository. No value is hand-entered.

**Retrieval regression benchmark** (`python llmops\run_regression_experiment.py`, 45 labelled queries over 24 passages):

| Configuration | Recall@1 | Recall@3 | MRR | Gate |
| :--- | :--- | :--- | :--- | :--- |
| C0 baseline (TF-IDF, top-3) | 0.9111 | 0.9778 | 0.9444 | PASS |
| C1 perturbed (noise σ=0.15) | 0.6000 | 0.7556 | 0.6704 | **BLOCKED** |

**Chat replay** (`python scripts\replay_chats.py`, 20 chats / 40 turns over one shared corpus):

| Measure | Value |
| :--- | :--- |
| Turns replayed | 40 |
| Answered from documents | 38 |
| Honest refusals | 2 |
| Audited RAG turns (PDF run + chat replay) | 44 |
| Mean faithfulness (cutoff 0.50) | 1.000 |
| Mean answer relevance | 0.807 |
| Mean composite RAG quality | 0.933 |
| Hallucination rate | 0.0% (0 / 44 decisions) |
| Mean latency | 6,859 ms |
| P95 latency | 23,123 ms |

The two refusals are `chat-8` turn 2 ("What categories of data does that cover?")
and `chat-11` turn 2 ("What is the purpose of that connection?"). Both are
pronoun-dependent follow-ups, and retrieval/grading sees only the current
question, so the pipeline refused instead of guessing. They remain in the log
with null quality metrics and are excluded from the aggregates.

Latency includes live Groq inference plus local sentence-transformer embedding,
so it is an end-to-end batch measurement, not a served-SLA figure.

**Guardrails**: `python llmops\test_quality_gate.py` runs 9 unit tests covering
injection blocking, input and output PII redaction (including SSN), grounded-answer
acceptance, hallucination rejection, drift "not evaluated" before a baseline exists
and after a reset, security-count coverage for blocked turns, and
cutoff-consistent hallucination rates.

---

## 📄 Submission Documents
- **Complete Academic Report:** [Project_Report.docx](https://github.com/user-attachments/files/34125039/Project_Report.docx)
