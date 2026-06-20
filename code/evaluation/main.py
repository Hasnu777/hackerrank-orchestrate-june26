#!/usr/bin/env python3
"""
Evaluation script for the damage claim evidence reviewer.

Runs the full pipeline on dataset/sample_claims.csv (which has expected outputs),
computes per-field accuracy metrics, and writes evaluation_report.md.

Usage (from repo root):
    python code/evaluation/main.py [options]
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Shared modules live in code/
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    Cache,
    OUTPUT_COLUMNS,
    build_user_history_map,
    load_csv,
    process_claim,
)

# ── Evaluation helpers ───────────────────────────────────────────────────────

# Fields to compute exact-match accuracy on
EXACT_MATCH_FIELDS = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]

# Set-valued fields: evaluate as set overlap (Jaccard)
SET_FIELDS = ["risk_flags", "supporting_image_ids"]

INPUT_FIELDS = {"user_id", "image_paths", "user_claim", "claim_object"}


def _str_to_set(val: str) -> set:
    return {v.strip().lower() for v in val.split(";") if v.strip()}


def compute_metrics(expected: list, predicted: list) -> dict:
    metrics: dict = {}

    for field in EXACT_MATCH_FIELDS:
        correct = total = 0
        for exp, pred in zip(expected, predicted):
            ev = str(exp.get(field, "")).strip().lower()
            pv = str(pred.get(field, "")).strip().lower()
            if ev:
                total += 1
                if ev == pv:
                    correct += 1
        metrics[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }

    for field in SET_FIELDS:
        jaccard_sum = total = 0.0
        for exp, pred in zip(expected, predicted):
            ev = _str_to_set(str(exp.get(field, "")))
            pv = _str_to_set(str(pred.get(field, "")))
            if ev:
                total += 1
                inter = len(ev & pv)
                union = len(ev | pv)
                jaccard_sum += inter / union if union else 1.0
        metrics[field] = {
            "avg_jaccard": round(jaccard_sum / total, 4) if total else 0.0,
            "total": int(total),
        }

    return metrics


def overall_accuracy(metrics: dict) -> float:
    totals = [m for f, m in metrics.items() if f in EXACT_MATCH_FIELDS]
    c = sum(m["correct"] for m in totals)
    t = sum(m["total"] for m in totals)
    return round(c / t, 4) if t else 0.0


# ── Report writer ────────────────────────────────────────────────────────────

def write_report(
    report_path: Path,
    metrics: dict,
    n_claims: int,
    n_images: int,
    model: str,
    elapsed_sec: float,
    cache_hits: int,
) -> None:
    oa = overall_accuracy(metrics)
    cache_misses = n_claims - cache_hits

    # Estimate tokens per claim: ~900 input (prompt + image) + ~200 output, times 3 stages
    est_input_tokens = cache_misses * 3 * 900
    est_output_tokens = cache_misses * 3 * 200
    cost_per_1k_in = 0.003
    cost_per_1k_out = 0.015
    est_cost = (
        (est_input_tokens / 1000) * cost_per_1k_in
        + (est_output_tokens / 1000) * cost_per_1k_out
    )

    lines = [
        "# Evaluation Report\n",
        "## Summary\n",
        f"- **Dataset**: `dataset/sample_claims.csv`",
        f"- **Claims evaluated**: {n_claims}",
        f"- **Images processed**: {n_images}",
        f"- **Model**: `{model}`",
        f"- **Cache hits / misses**: {cache_hits} / {cache_misses}",
        f"- **Elapsed time**: {elapsed_sec:.1f}s",
        "",
        "---\n",
        "## Per-Field Accuracy\n",
        "| Field | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for field in EXACT_MATCH_FIELDS:
        m = metrics[field]
        lines.append(
            f"| `{field}` | {m['correct']} | {m['total']} | {m['accuracy']:.1%} |"
        )
    lines += [
        f"| **Overall (exact-match fields)** | — | — | **{oa:.1%}** |",
        "",
        "### Set-valued fields (Jaccard similarity)\n",
        "| Field | Avg Jaccard | Samples |",
        "|---|---|---|",
    ]
    for field in SET_FIELDS:
        m = metrics[field]
        lines.append(
            f"| `{field}` | {m['avg_jaccard']:.3f} | {m['total']} |"
        )

    lines += [
        "",
        "---\n",
        "## Strategy Comparison\n",
        "Two strategies were considered:\n",
        "### Strategy A — Multi-stage Claude pipeline (final strategy used)\n",
        "Each claim is processed in three sequential Claude API calls:",
        "- **Stage 1 (history)**: Text-only credibility assessment from user claim history",
        "- **Stage 2 (vision)**: Image analysis — damage type, severity, evidence standard",
        "- **Stage 3 (decision)**: Synthesis of history and visual results into final verdict\n",
        "**Pros**: specialised prompts per stage, visual and text reasoning decoupled, "
        "each stage can use the most cost-effective model  ",
        "**Cons**: three API calls per claim, higher latency than a single-pass approach\n",
        "### Strategy B — Single-pass VLM (considered, not used)\n",
        "All context (images, history, evidence requirements) sent in one prompt. Simpler "
        "but the model must handle history risk and visual analysis simultaneously, "
        "reducing the quality of each sub-task.\n",
        "**Decision**: Strategy A was chosen for accuracy and output quality.",
        "",
        "---\n",
        "## Operational Analysis\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Model calls (sample) | {n_claims * 3} (3 per claim, 0 if fully cached) |",
        f"| Estimated input tokens | ~{est_input_tokens:,} |",
        f"| Estimated output tokens | ~{est_output_tokens:,} |",
        f"| Images processed | {n_images} |",
        f"| Approx. cost (Anthropic API) | ~${est_cost:.2f} USD |",
        f"| Pricing assumption | $0.003/1K input, $0.015/1K output |",
        f"| Approx. latency | {elapsed_sec:.1f}s for {n_claims} claims ({elapsed_sec/n_claims:.1f}s avg) |",
        "",
        "### Full test set projection\n",
        "- Test set: ~45 claims, ~80 images",
        "- Estimated model calls: 135 (3 per claim, with cache)",
        "- Estimated input tokens: ~121,500",
        "- Estimated output tokens: ~27,000",
        "- Estimated cost: ~$0.77 USD",
        "",
        "### Caching strategy\n",
        "- **Caching**: all model responses are persisted to `.cache/eval_responses.json`.",
        "  The cache key is SHA-256 of `stage|model|user_id|image_paths|user_claim|claim_object`.",
        "  Re-runs are fully instant for cached claims.",
        "- **Bad-cache detection**: cached entries with error/fallback values are automatically",
        "  invalidated and re-called on the next run.",
        "",
        "---\n",
        "_Report generated by `code/evaluation/main.py`_",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the damage claim system on sample_claims.csv."
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--history-model", default="claude-sonnet-4-6",
                        help="Model for history assessment stage")
    parser.add_argument("--vision-model", default="claude-opus-4-8",
                        help="Model for image analysis stage")
    parser.add_argument("--decision-model", default="claude-sonnet-4-6",
                        help="Model for final decision stage")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--predictions-out", default=None,
                        help="Optional path to save predictions CSV")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    dataset_root = (
        Path(args.dataset_root) if args.dataset_root else repo_root / "dataset"
    )
    cache_path = (
        Path(args.cache) if args.cache
        else Path(__file__).parent.parent / ".cache" / "eval_responses.json"
    )
    report_path = (
        Path(args.report_out) if args.report_out
        else Path(__file__).parent / "evaluation_report.md"
    )

    sample_path = dataset_root / "sample_claims.csv"
    sample_rows = load_csv(sample_path)
    user_history_rows = load_csv(dataset_root / "user_history.csv")
    evidence_requirements = load_csv(dataset_root / "evidence_requirements.csv")
    user_history_map = build_user_history_map(user_history_rows)
    cache = Cache(cache_path)

    # Separate inputs and expected outputs
    input_rows = [{k: r[k] for k in INPUT_FIELDS} for r in sample_rows]
    expected_rows = [
        {k: r[k] for k in r if k not in INPUT_FIELDS}
        for r in sample_rows
    ]

    n_images = sum(
        len([p for p in r["image_paths"].split(";") if p.strip()])
        for r in input_rows
    )

    print(
        f"Evaluating {len(input_rows)} sample claims ({n_images} images): "
        f"history={args.history_model} | vision={args.vision_model} | decision={args.decision_model}"
    )
    cache_hits_before = len(cache)
    t0 = time.time()

    predicted_rows = []
    for i, row in enumerate(input_rows, 1):
        if args.verbose:
            print(f"  [{i:2d}/{len(input_rows)}] {row['user_id']}")
        pred = process_claim(
            row, user_history_map, evidence_requirements,
            dataset_root, cache,
            verbose=args.verbose,
            history_model=args.history_model,
            vision_model=args.vision_model,
            decision_model=args.decision_model,
        )
        predicted_rows.append(pred)

    elapsed = time.time() - t0

    metrics = compute_metrics(expected_rows, predicted_rows)

    print("\n=== Evaluation Results ===")
    for field in EXACT_MATCH_FIELDS:
        m = metrics[field]
        print(f"  {field:32s}: {m['correct']}/{m['total']} = {m['accuracy']:.1%}")
    print(f"  {'risk_flags (Jaccard)':32s}: {metrics['risk_flags']['avg_jaccard']:.3f}")
    print(f"  {'supporting_image_ids (Jaccard)':32s}: {metrics['supporting_image_ids']['avg_jaccard']:.3f}")
    oa = overall_accuracy(metrics)
    print(f"\n  Overall exact-match accuracy: {oa:.1%}")
    print(f"  Elapsed: {elapsed:.1f}s")

    model_label = f"history={args.history_model}, vision={args.vision_model}, decision={args.decision_model}"
    write_report(
        report_path, metrics,
        n_claims=len(input_rows),
        n_images=n_images,
        model=model_label,
        elapsed_sec=elapsed,
        cache_hits=cache_hits_before,
    )

    if args.predictions_out:
        pout = Path(args.predictions_out)
        pout.parent.mkdir(parents=True, exist_ok=True)
        with open(pout, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for r in predicted_rows:
                writer.writerow({col: r.get(col, "") for col in OUTPUT_COLUMNS})
        print(f"Predictions saved to {pout}")


if __name__ == "__main__":
    main()
