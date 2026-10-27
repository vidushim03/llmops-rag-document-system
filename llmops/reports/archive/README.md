# Archived telemetry — not read by the app or the dashboard

Nothing in this folder is loaded at runtime. It exists only so that removed data
can be audited rather than silently disappearing.

| File | What it is | Why it is here |
| :--- | :--- | :--- |
| `telemetry_audit_log.pre_instrumentation.jsonl` | 4 RAG turns + 1 chat turn logged before questions and retrieved chunks were stored | Real turns, but their context-dependent fields cannot be recomputed, so they were taken out of the live log |
| `telemetry_audit_log.legacy.jsonl` | Pre-split log from the period when real and simulated traffic shared one file | Contains rows of mixed provenance; kept for audit, never aggregated |
| `telemetry_audit_log.pre_split.jsonl` | Snapshot taken when the logs were first separated | Superseded by the split |
| `telemetry_audit_log.before_reclassify.jsonl` | Snapshot before non-answer rows were reclassified | Superseded |
| `telemetry_audit_log.run1_drift_threshold_0.18.jsonl` | A real end-to-end run recorded before the drift threshold was calibrated to 0.06 | Same questions and pipeline as the current run; superseded only by the threshold fix |

Removed outright (not archived) because they contained no measured data:

- `telemetry_benchmark_log.jsonl` — 14 rows produced by the deleted
  `llmops/run_benchmark.py`, which fabricated latency with
  `np.random.uniform(95, 190)` and wrote pre-computed metric dictionaries.
- `telemetry_playground_log.jsonl` — an empty sink for playground demos.
- `llmops_metrics_dashboard.png` — a chart rendered from the fabricated benchmark.
