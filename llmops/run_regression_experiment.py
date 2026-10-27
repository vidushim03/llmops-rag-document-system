"""
Automated Pipeline Regression Experiment
Directly tests the research question:
'Can automated evaluation gates detect performance regressions across pipeline configuration shifts?'
Compares:
- Baseline Configuration C0 (unperturbed TF-IDF retrieval, top-3)
- Degraded / Drifted Configuration C1 (Gaussian noise injected into similarity scores)

C1 models a *silent* regression (e.g. a subtly drifted index), not a total
outage, so the default perturbation is moderate (0.15). Pass --noise to study a
different point on the degradation curve.

Dataset: 45 labelled queries over 24 passages (20 relevant targets + 4 topical
distractors). Queries are paraphrased rather than copied from the passages, so
lexical retrieval cannot score 1.0 by keyword matching alone.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llmops.retrieval_metrics import RetrievalMetricsEngine

ROOT = Path(__file__).resolve().parent.parent


def run_regression_experiment(seed: int = 42, noise_factor: float = 0.15):
    # The degraded configuration injects Gaussian noise into the similarity
    # scores, so without a fixed seed the C1 metrics change on every run and the
    # experiment is not reproducible. Pin the seed for a stable reported result.
    np.random.seed(seed)

    print("==================================================================")
    print("  AIM4142 MLOps: Automated Pipeline Regression & Gate Experiment  ")
    print("==================================================================\n")

    # Single source of truth for the evaluation corpus. 24 passages: 20 relevant
    # targets and 4 topical distractors, so Recall@1 is actually earned.
    corpus_path = ROOT / "llmops" / "data" / "eval_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    golden_path = ROOT / "llmops" / "data" / "eval_golden_dataset.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    engine = RetrievalMetricsEngine(corpus)

    # 1. Evaluate Baseline Configuration C0 (Production Optimal: k=3, exact vector matching)
    c0_metrics = engine.evaluate_benchmark(golden_dataset, k=3, noise_factor=0.0)

    # 2. Evaluate Candidate Configuration C1 (Degraded Pipeline: noise perturbation / bad embeddings)
    c1_metrics = engine.evaluate_benchmark(golden_dataset, k=3, noise_factor=noise_factor)

    # Calculate Regression Deltas
    mrr_delta = c1_metrics["MRR"] - c0_metrics["MRR"]
    recall_delta = c1_metrics["Recall@3"] - c0_metrics["Recall@3"]

    # CI/CD Quality Gate Threshold: Disallow drop of MRR > 0.15 or Recall > 0.10
    gate_failed = bool(mrr_delta < -0.15 or recall_delta < -0.10)

    df_comp = pd.DataFrame([
        {"Configuration": "C0: Baseline Production", "Recall@1": c0_metrics["Recall@1"], "Recall@3": c0_metrics["Recall@3"], "MRR": c0_metrics["MRR"], "CI/CD Gate": "PASS (Baseline)"},
        {"Configuration": "C1: Degraded / Drifted", "Recall@1": c1_metrics["Recall@1"], "Recall@3": c1_metrics["Recall@3"], "MRR": c1_metrics["MRR"], "CI/CD Gate": "BLOCKED (Regression Detected)" if gate_failed else "PASS"}
    ])

    print(f"Golden dataset: {len(golden_dataset)} queries over {len(corpus)} passages")
    print(f"C1 perturbation: noise_factor = {noise_factor} | seed = {seed}\n")
    print(df_comp.to_string(index=False))
    print("\n------------------------------------------------------------------")
    print(f"Regression Analysis: MRR Delta = {mrr_delta:.4f} | Recall@3 Delta = {recall_delta:.4f}")
    if gate_failed:
        print("[CI/CD GATE TRIGGERED] Production deployment BLOCKED to prevent silent model degradation.")
    else:
        print("[CI/CD GATE PASSED] Performance within acceptable tolerances.")
    print("==================================================================\n")

    return df_comp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG regression gate experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.15,
        help="Gaussian noise standard deviation applied to C1 similarity scores",
    )
    args = parser.parse_args()
    run_regression_experiment(seed=args.seed, noise_factor=args.noise)

