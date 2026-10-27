#!/usr/bin/env python3
"""
evals/run_analyzer.py — Stage 3: live evaluation of analyze_sentiment_and_competitors.

Requires ANALYZER_MODEL env var (or --analyzer-model flag).
Results are cached by (model, case_id) so reruns don't spend quota.

Usage:
    cd ai_visibility_tracker
    python evals/run_analyzer.py --max-calls 40
    python evals/run_analyzer.py --include-unverified --max-calls 10
    python evals/run_analyzer.py --compare-analyzer groq:llama-3.1-8b-instant
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis import analyze_sentiment_and_competitors, fallback_extract_competitors
from evals.golden_loader import load_golden_cases, summarise_dataset
from evals.metrics import (
    sentiment_accuracy,
    set_precision_recall_f1,
    micro_average,
    macro_average,
    precision_recall_f1,
    cohens_kappa,
    jaccard_similarity,
    normalise_competitor_set,
    most_frequent_false_positives,
)

CACHE_FILE = _HERE / "analyzer_cache.json"
DEFAULT_MAX_CALLS = 40


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> Dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def cache_key(model: str, case_id: str) -> str:
    return f"{model}|{case_id}"


# ---------------------------------------------------------------------------
# Single-case analyzer call
# ---------------------------------------------------------------------------

def run_analyzer_on_case(
    case: Dict,
    analyzer_model: str,
    calls_made: int,
    max_calls: int,
    cache: Dict,
) -> Tuple[Dict, int]:
    """
    Returns (result_dict, calls_made_after).
    result_dict keys: sentiment, competitors, analyzer_status, analyzer_error,
                      from_cache (bool).
    """
    key = cache_key(analyzer_model, case["id"])
    if key in cache:
        cached = cache[key].copy()
        cached["from_cache"] = True
        return cached, calls_made

    if calls_made >= max_calls:
        return {
            "sentiment": "unknown",
            "competitors": [],
            "analyzer_status": "skipped",
            "analyzer_error": f"Call cap ({max_calls}) reached before this case.",
            "from_cache": False,
        }, calls_made

    try:
        result = analyze_sentiment_and_competitors(
            ai_response=case["answer_text"],
            brand_name=case["brand"],
            analyzer_model=analyzer_model,
            brand_mentioned=case["labels"]["mentioned"],
            known_competitors=None,
            aliases=case["aliases"] or None,
        )
        calls_made += 1
        entry = {
            "sentiment": result.get("sentiment", "unknown"),
            "competitors": result.get("competitors", []),
            "analyzer_status": result.get("analyzer_status", "ok"),
            "analyzer_error": result.get("analyzer_error"),
        }
        cache[key] = entry
        save_cache(cache)
        return {**entry, "from_cache": False}, calls_made

    except Exception as exc:
        err = str(exc)
        # Persistent 429: abort cleanly
        if "rate limit" in err.lower() or "429" in err:
            print(f"\n⚠  Rate limit hit on case {case['id']}. Stopping cleanly.", file=sys.stderr)
            return {
                "sentiment": "unknown",
                "competitors": [],
                "analyzer_status": "rate_limited",
                "analyzer_error": "Rate limit exceeded — partial report follows.",
                "from_cache": False,
            }, max_calls  # Force stop by exhausting cap
        calls_made += 1
        return {
            "sentiment": "unknown",
            "competitors": [],
            "analyzer_status": "failed",
            "analyzer_error": err[:200],
            "from_cache": False,
        }, calls_made


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_analyzer_metrics(case_results: List[Dict]) -> Dict:
    """Compute sentiment and competitor metrics from per-case results."""
    completed = [r for r in case_results if r["analyzer_result"]["analyzer_status"] not in ("skipped",)]
    failed = [r for r in completed if r["analyzer_result"]["analyzer_status"] in ("failed", "rate_limited")]
    unknown_rate = len(failed) / len(completed) if completed else 0.0

    pred_sentiments = []
    gold_sentiments = []
    for r in case_results:
        gold_sent = r["gold_labels"]["sentiment"]
        pred_sent = r["analyzer_result"]["sentiment"]
        if gold_sent and r["gold_labels"]["mentioned"]:
            pred_sentiments.append(pred_sent or "unknown")
            gold_sentiments.append(gold_sent)

    sent_metrics = sentiment_accuracy(pred_sentiments, gold_sentiments)

    # Competitor metrics (analyzer output vs gold)
    per_case_comp: List[Tuple[int, int, int]] = []
    per_case_prf: List[Tuple[float, float, float]] = []
    for r in case_results:
        pred_comps = normalise_competitor_set(
            r["analyzer_result"]["competitors"],
            r["brand"],
            r["case"].get("aliases") or [],
        )
        gold_comps = normalise_competitor_set(
            r["gold_labels"]["competitors"],
            r["brand"],
            r["case"].get("aliases") or [],
        )
        prec, rec, f1 = set_precision_recall_f1(pred_comps, gold_comps)
        tp = len(pred_comps & gold_comps)
        fp = len(pred_comps - gold_comps)
        fn = len(gold_comps - pred_comps)
        per_case_comp.append((tp, fp, fn))
        per_case_prf.append((prec, rec, f1))

    micro_p, micro_r, micro_f = micro_average(per_case_comp)
    macro_p, macro_r, macro_f = macro_average(per_case_prf)

    # Combined analyzer+fallback result
    combined_per_case: List[Tuple[int, int, int]] = []
    for r in case_results:
        pred_comps = normalise_competitor_set(
            r.get("combined_competitors", r["analyzer_result"]["competitors"]),
            r["brand"],
            r["case"].get("aliases") or [],
        )
        gold_comps = normalise_competitor_set(
            r["gold_labels"]["competitors"],
            r["brand"],
            r["case"].get("aliases") or [],
        )
        tp = len(pred_comps & gold_comps)
        fp = len(pred_comps - gold_comps)
        fn = len(gold_comps - pred_comps)
        combined_per_case.append((tp, fp, fn))
    comb_p, comb_r, comb_f = micro_average(combined_per_case)

    return {
        "calls_completed": len(completed),
        "calls_failed": len(failed),
        "unknown_rate": unknown_rate,
        "sentiment": sent_metrics,
        "competitor_analyzer": {
            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
            "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        },
        "competitor_combined": {
            "micro": {"precision": comb_p, "recall": comb_r, "f1": comb_f},
        },
    }


def compute_agreement(
    primary_results: List[Dict],
    compare_results: List[Dict],
) -> Dict:
    """Compute Cohen's kappa (sentiment) and Jaccard (competitors) between two analyzers."""
    primary_sentiments = []
    compare_sentiments = []
    jaccard_scores = []

    for pr, cr in zip(primary_results, compare_results):
        ps = pr["analyzer_result"]["sentiment"]
        cs = cr["analyzer_result"]["sentiment"]
        if ps and cs:
            primary_sentiments.append(ps)
            compare_sentiments.append(cs)

        pc = set(pr["analyzer_result"]["competitors"])
        cc = set(cr["analyzer_result"]["competitors"])
        jaccard_scores.append(jaccard_similarity(pc, cc))

    kappa = cohens_kappa(primary_sentiments, compare_sentiments) if primary_sentiments else 0.0
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    return {
        "sentiment_kappa": kappa,
        "competitor_avg_jaccard": avg_jaccard,
        "n_sentiment_pairs": len(primary_sentiments),
        "n_competitor_pairs": len(jaccard_scores),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_report(
    analyzer_model: str,
    cases: List[Dict],
    case_results: List[Dict],
    metrics: Dict,
    calls_made: int,
    max_calls: int,
    compare_model: Optional[str] = None,
    compare_results: Optional[List[Dict]] = None,
    agreement: Optional[Dict] = None,
) -> None:
    summary = summarise_dataset(cases)
    print("\n" + "=" * 72)
    print("  AI Visibility Tracker — Analyzer Eval Report (Stage 3)")
    print("=" * 72)
    print(f"  Analyzer model  : {analyzer_model}")
    print(f"  Cases evaluated : {summary['total']}  ({summary['verified']} verified)")
    print(f"  API calls made  : {calls_made} / {max_calls}")
    print(f"  Failures/unknown: {_pct(metrics['unknown_rate'])}")
    print()

    sent = metrics["sentiment"]
    print("┌─ SENTIMENT ACCURACY ────────────────────────────────────────────┐")
    print(f"│  Overall accuracy : {_pct(sent['accuracy'])}  (N={sent['n']})")
    print(f"│  Unknown/fail rate: {_pct(sent['unknown_rate'])}")
    for cls, m in sent.get("per_class", {}).items():
        print(f"│  {cls:<10} P={_pct(m['precision']):<8} R={_pct(m['recall']):<8} F1={_pct(m['f1'])}")
    if sent.get("confusion_matrix"):
        print("│  Confusion matrix (rows=actual, cols=predicted):")
        classes = list(sent["confusion_matrix"].keys())
        print(f"│    {'':12} " + "  ".join(f"{c:<10}" for c in classes))
        for actual_cls, row in sent["confusion_matrix"].items():
            vals = "  ".join(f"{row[c]:<10}" for c in classes)
            print(f"│    {actual_cls:<12} {vals}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    ca = metrics["competitor_analyzer"]
    cc = metrics["competitor_combined"]
    print("┌─ COMPETITOR METRICS ────────────────────────────────────────────┐")
    print(f"│  Analyzer-only  micro F1 : {_pct(ca['micro']['f1'])}")
    print(f"│  Analyzer+fallback  F1   : {_pct(cc['micro']['f1'])}")
    print(f"│  Macro (analyzer)        : P={_pct(ca['macro']['precision'])} R={_pct(ca['macro']['recall'])} F1={_pct(ca['macro']['f1'])}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    if compare_model and agreement:
        print("┌─ ANALYZER AGREEMENT ────────────────────────────────────────────┐")
        print(f"│  Compare model         : {compare_model}")
        print(f"│  Sentiment Cohen kappa : {agreement['sentiment_kappa']:.3f}")
        print(f"│  Competitor avg Jaccard: {agreement['competitor_avg_jaccard']:.3f}")
        print("└─────────────────────────────────────────────────────────────────┘")
        print()

    if calls_made >= max_calls:
        print(f"⚠  Call cap ({max_calls}) reached — this is a PARTIAL report.")
        processed = len([r for r in case_results if r["analyzer_result"]["analyzer_status"] != "skipped"])
        print(f"   {processed} of {len(cases)} cases processed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live evaluation of the analyzer model against the golden set."
    )
    parser.add_argument(
        "--analyzer-model",
        default=os.environ.get("ANALYZER_MODEL", ""),
        help="Analyzer model string (provider:model). Defaults to ANALYZER_MODEL env var.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"Hard cap on API calls (default {DEFAULT_MAX_CALLS}).",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include unverified cases (dry run).",
    )
    parser.add_argument(
        "--compare-analyzer",
        metavar="PROVIDER:MODEL",
        default=None,
        help="Run a second analyzer on the same cases and report agreement.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Write results JSON to DIR/analyzer_results.json.",
    )
    args = parser.parse_args(argv)

    analyzer_model = args.analyzer_model.strip()
    if not analyzer_model:
        print(
            "Error: ANALYZER_MODEL not set. Use --analyzer-model or set ANALYZER_MODEL env var.",
            file=sys.stderr,
        )
        return 1

    if args.compare_analyzer and args.compare_analyzer.strip() == analyzer_model:
        print(
            "Error: --compare-analyzer cannot be the same model as the primary analyzer.",
            file=sys.stderr,
        )
        return 1

    cases = load_golden_cases(include_unverified=args.include_unverified)
    if not cases:
        print(
            "No cases loaded. Use --include-unverified for a dry run.",
            file=sys.stderr,
        )
        return 1

    cache = load_cache()
    calls_made = 0
    max_calls = args.max_calls

    case_results: List[Dict] = []

    print(f"Running analyzer eval with model={analyzer_model}, max_calls={max_calls}")
    for i, case in enumerate(cases):
        result, calls_made = run_analyzer_on_case(
            case, analyzer_model, calls_made, max_calls, cache
        )
        # Combined = analyzer result; if analyzer failed, fallback was already used
        combined_competitors = result["competitors"]
        if result["analyzer_status"] in ("failed", "rate_limited", "skipped"):
            combined_competitors = fallback_extract_competitors(
                case["answer_text"], case["brand"], None, case["aliases"] or None
            )

        case_results.append({
            "case_id": case["id"],
            "brand": case["brand"],
            "case": case,
            "gold_labels": case["labels"],
            "analyzer_result": result,
            "combined_competitors": combined_competitors,
        })

        status_char = "✓" if result["analyzer_status"] == "ok" else ("C" if result["from_cache"] else "✗")
        print(f"  [{i+1:2d}/{len(cases)}] {case['id']:<12} {status_char} "
              f"sent={result['sentiment']:<10} "
              f"comps={len(result['competitors'])}", end="")
        if result["from_cache"]:
            print(" (cached)", end="")
        print()

        if calls_made >= max_calls:
            print(f"\n⚠  Call cap reached ({max_calls}). Stopping.", file=sys.stderr)
            break

    metrics = compute_analyzer_metrics(case_results)

    # Optional comparison analyzer
    compare_results = None
    agreement = None
    if args.compare_analyzer:
        compare_model = args.compare_analyzer.strip()
        compare_calls_made = 0
        compare_case_results: List[Dict] = []
        print(f"\nRunning comparison analyzer: {compare_model}")
        for i, case in enumerate(cases[:len(case_results)]):
            cr, compare_calls_made = run_analyzer_on_case(
                case, compare_model, compare_calls_made,
                max_calls - calls_made, cache
            )
            compare_case_results.append({
                "case_id": case["id"],
                "brand": case["brand"],
                "case": case,
                "gold_labels": case["labels"],
                "analyzer_result": cr,
            })
        compare_results = compare_case_results
        agreement = compute_agreement(case_results, compare_results)

    print_report(
        analyzer_model, cases, case_results, metrics,
        calls_made, max_calls,
        args.compare_analyzer, compare_results, agreement,
    )

    report = {
        "analyzer_model": analyzer_model,
        "max_calls": max_calls,
        "calls_made": calls_made,
        "summary": summarise_dataset(cases),
        "metrics": metrics,
        "agreement": agreement,
    }

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "analyzer_results.json"
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nResults written to {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
