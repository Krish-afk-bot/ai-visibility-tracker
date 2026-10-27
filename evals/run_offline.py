#!/usr/bin/env python3
"""
evals/run_offline.py — Stage 2: offline evaluation of deterministic functions.

No network or API keys required. Evaluates:
  * detect_brand_mention_deterministic  (mention detection + count + position)
  * fallback_extract_competitors        (competitor recall/precision)

Usage:
    cd ai_visibility_tracker
    python evals/run_offline.py
    python evals/run_offline.py --include-unverified   # dry-run with all 40 cases
    python evals/run_offline.py --output evals/results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the project root is on sys.path so analysis.py can be imported
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis import detect_brand_mention_deterministic, fallback_extract_competitors
from evals.golden_loader import load_golden_cases, summarise_dataset
from evals.metrics import (
    precision_recall_f1,
    set_precision_recall_f1,
    micro_average,
    macro_average,
    mention_count_metrics,
    position_metrics,
    most_frequent_false_positives,
    normalise_competitor_set,
    breakdown_by_tag,
)


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_mention(case: Dict) -> Dict:
    """Run detect_brand_mention_deterministic on one case and return results."""
    result = detect_brand_mention_deterministic(
        text=case["answer_text"],
        brand_name=case["brand"],
        aliases=case["aliases"] or None,
    )
    gold = case["labels"]
    return {
        "id": case["id"],
        "tags": case["tags"],
        "pred_mentioned": result["brand_mentioned"],
        "gold_mentioned": gold["mentioned"],
        "pred_count": result["mention_count"],
        "gold_count": gold["mention_count"],
        "pred_position": result["position"],
        "gold_position": gold["position"],
        "correct_mention": result["brand_mentioned"] == gold["mentioned"],
        "correct_count": result["mention_count"] == gold["mention_count"],
    }


def evaluate_competitor(case: Dict) -> Dict:
    """Run fallback_extract_competitors on one case and return results."""
    predicted_raw = fallback_extract_competitors(
        ai_response=case["answer_text"],
        brand_name=case["brand"],
        known_competitors=None,
        aliases=case["aliases"] or None,
    )
    gold_raw = case["labels"]["competitors"]

    pred_set = normalise_competitor_set(predicted_raw, case["brand"], case["aliases"])
    gold_set = normalise_competitor_set(gold_raw, case["brand"], case["aliases"])

    pred_lower = pred_set
    gold_lower = gold_set
    tp = len(pred_lower & gold_lower)
    fp = len(pred_lower - gold_lower)
    fn = len(gold_lower - pred_lower)
    prec, rec, f1 = precision_recall_f1(tp, fp, fn)

    false_positives = sorted(pred_lower - gold_lower)
    false_negatives = sorted(gold_lower - pred_lower)

    return {
        "id": case["id"],
        "tags": case["tags"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted": sorted(pred_lower),
        "gold": sorted(gold_lower),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_mention_metrics(
    results: List[Dict],
) -> Dict:
    """Compute P/R/F1, count accuracy, position accuracy from per-case results."""
    tp = sum(1 for r in results if r["pred_mentioned"] and r["gold_mentioned"])
    fp = sum(1 for r in results if r["pred_mentioned"] and not r["gold_mentioned"])
    fn = sum(1 for r in results if not r["pred_mentioned"] and r["gold_mentioned"])
    prec, rec, f1 = precision_recall_f1(tp, fp, fn)

    false_positive_ids = [r["id"] for r in results if r["pred_mentioned"] and not r["gold_mentioned"]]
    false_negative_ids = [r["id"] for r in results if not r["pred_mentioned"] and r["gold_mentioned"]]

    count_m = mention_count_metrics(
        [r["pred_count"] for r in results],
        [r["gold_count"] for r in results],
    )

    # Position only on cases where brand IS mentioned in gold
    mentioned_results = [r for r in results if r["gold_mentioned"]]
    pos_m = position_metrics(
        [r["pred_position"] for r in mentioned_results],
        [r["gold_position"] for r in mentioned_results],
    )

    return {
        "detection": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "false_positive_ids": false_positive_ids,
            "false_negative_ids": false_negative_ids,
        },
        "count": count_m,
        "position": pos_m,
    }


def aggregate_competitor_metrics(results: List[Dict]) -> Dict:
    """Compute micro/macro P/R/F1 and top false positives."""
    per_case_prf = [(r["precision"], r["recall"], r["f1"]) for r in results]
    per_case_counts = [(r["tp"], r["fp"], r["fn"]) for r in results]

    micro_p, micro_r, micro_f = micro_average(per_case_counts)
    macro_p, macro_r, macro_f = macro_average(per_case_prf)

    fp_lists = [r["false_positives"] for r in results]
    top_fps = most_frequent_false_positives(fp_lists, top_n=15)

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "top_false_positives": top_fps,
        "per_case": results,
    }


# ---------------------------------------------------------------------------
# Tag breakdown wrappers
# ---------------------------------------------------------------------------

def _mention_tag_metric_fn(subset: List[Dict]) -> Dict:
    results = [evaluate_mention(c) for c in subset]
    return aggregate_mention_metrics(results)


def _competitor_tag_metric_fn(subset: List[Dict]) -> Dict:
    results = [evaluate_competitor(c) for c in subset]
    return aggregate_competitor_metrics(results)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_report(
    cases: List[Dict],
    mention_results: List[Dict],
    competitor_results: List[Dict],
    mention_agg: Dict,
    competitor_agg: Dict,
    mention_by_tag: Dict,
    competitor_by_tag: Dict,
) -> None:
    summary = summarise_dataset(cases)
    total = summary["total"]
    verified = summary["verified"]

    print("\n" + "=" * 72)
    print("  AI Visibility Tracker — Offline Eval Report (Stage 2)")
    print("=" * 72)
    print(f"  Cases evaluated : {total}  ({verified} verified)")
    print(f"  Dataset path    : {Path(__file__).parent / 'golden' / 'cases.jsonl'}")
    print()

    # --- Mention detection ---
    det = mention_agg["detection"]
    cnt = mention_agg["count"]
    pos = mention_agg["position"]

    print("┌─ MENTION DETECTION ─────────────────────────────────────────────┐")
    print(f"│  Precision : {_pct(det['precision']):<8}  TP={det['tp']}  FP={det['fp']}  FN={det['fn']}")
    print(f"│  Recall    : {_pct(det['recall']):<8}")
    print(f"│  F1        : {_pct(det['f1']):<8}")
    if det["false_positive_ids"]:
        print(f"│  False positives : {', '.join(det['false_positive_ids'])}")
    if det["false_negative_ids"]:
        print(f"│  False negatives : {', '.join(det['false_negative_ids'])}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    print("┌─ MENTION COUNT ─────────────────────────────────────────────────┐")
    print(f"│  Exact-match accuracy : {_pct(cnt['exact_match_accuracy'])}")
    print(f"│  Mean absolute error  : {cnt['mean_absolute_error']:.3f}")
    print(f"│  N evaluated          : {cnt['n']}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    print("┌─ POSITION (brand-mentioned cases only) ─────────────────────────┐")
    print(f"│  Exact-match accuracy : {_pct(pos['exact_match_accuracy'])}")
    print(f"│  Within-1 accuracy    : {_pct(pos['within_1_accuracy'])}")
    print(f"│  N evaluated          : {pos['n']}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    # --- Competitor fallback ---
    comp_micro = competitor_agg["micro"]
    comp_macro = competitor_agg["macro"]
    top_fps = competitor_agg["top_false_positives"]

    print("┌─ COMPETITOR FALLBACK ───────────────────────────────────────────┐")
    print(f"│  Micro  P={_pct(comp_micro['precision']):<8} R={_pct(comp_micro['recall']):<8} F1={_pct(comp_micro['f1'])}")
    print(f"│  Macro  P={_pct(comp_macro['precision']):<8} R={_pct(comp_macro['recall']):<8} F1={_pct(comp_macro['f1'])}")
    if top_fps:
        print("│  Top false-positive strings (add to GENERIC_COMPETITOR_TERMS?):")
        for name, count in top_fps[:10]:
            print(f"│    [{count:3d}x]  {name}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    # --- Tag breakdowns ---
    print("┌─ MENTION DETECTION BY TAG ──────────────────────────────────────┐")
    for tag, metrics in sorted(mention_by_tag.items()):
        det_t = metrics["detection"]
        print(f"│  {tag:<26}  F1={_pct(det_t['f1']):<8}  "
              f"P={_pct(det_t['precision']):<8}  R={_pct(det_t['recall'])}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    print("┌─ COMPETITOR FALLBACK BY TAG ────────────────────────────────────┐")
    for tag, metrics in sorted(competitor_by_tag.items()):
        micro = metrics["micro"]
        print(f"│  {tag:<26}  F1={_pct(micro['f1']):<8}  "
              f"P={_pct(micro['precision']):<8}  R={_pct(micro['recall'])}")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report_dict(
    cases: List[Dict],
    mention_results: List[Dict],
    competitor_results: List[Dict],
) -> Dict:
    """Build a serialisable report dict for JSON output or threshold checks."""
    mention_agg = aggregate_mention_metrics(mention_results)
    competitor_agg = aggregate_competitor_metrics(competitor_results)
    mention_by_tag = breakdown_by_tag(cases, _mention_tag_metric_fn)
    competitor_by_tag = breakdown_by_tag(cases, _competitor_tag_metric_fn)

    return {
        "summary": summarise_dataset(cases),
        "mention": mention_agg,
        "competitor_fallback": {
            "micro": competitor_agg["micro"],
            "macro": competitor_agg["macro"],
            "top_false_positives": competitor_agg["top_false_positives"],
        },
        "by_tag": {
            "mention": {
                tag: m["detection"] for tag, m in mention_by_tag.items()
            },
            "competitor": {
                tag: m["micro"] for tag, m in competitor_by_tag.items()
            },
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline evaluation of deterministic analysis functions."
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include cases with label_status='needs_human_review' (dry run).",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Write results JSON to DIR/offline_results.json.",
    )
    args = parser.parse_args(argv)

    cases = load_golden_cases(include_unverified=args.include_unverified)
    if not cases:
        print(
            "No cases loaded. Run with --include-unverified for a dry run, "
            "or verify some cases in cases.jsonl (set label_status: 'verified').",
            file=sys.stderr,
        )
        return 1

    mention_results = [evaluate_mention(c) for c in cases]
    competitor_results = [evaluate_competitor(c) for c in cases]
    mention_agg = aggregate_mention_metrics(mention_results)
    competitor_agg = aggregate_competitor_metrics(competitor_results)
    mention_by_tag = breakdown_by_tag(cases, _mention_tag_metric_fn)
    competitor_by_tag = breakdown_by_tag(cases, _competitor_tag_metric_fn)

    print_report(
        cases,
        mention_results,
        competitor_results,
        mention_agg,
        competitor_agg,
        mention_by_tag,
        competitor_by_tag,
    )

    report = build_report_dict(cases, mention_results, competitor_results)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "offline_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Results written to {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
