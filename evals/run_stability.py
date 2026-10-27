#!/usr/bin/env python3
"""
evals/run_stability.py — Stage 4: stability eval (optional, live, capped).

Repeats a small fixed prompt set N times against selected models and reports
variance in visibility scores and mention/no-mention flip rates.

Usage:
    cd ai_visibility_tracker
    python evals/run_stability.py --models groq:llama3 --runs 3
    python evals/run_stability.py --models groq:m1,openrouter:m2 --runs 5 --yes
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_client import query_ai, parse_provider_model, RateLimitExceededError, LLMClientError
from analysis import detect_brand_mention_deterministic

# ── Fixed stability prompt set ──────────────────────────────────────────────
# These prompts are chosen because they exercise both clear mentions and
# borderline cases. They are intentionally varied in phrasing.
STABILITY_PROMPTS = [
    {"brand": "Notion", "prompt": "What are the best productivity apps for students?"},
    {"brand": "Slack", "prompt": "Which team communication tools are most popular in 2024?"},
    {"brand": "Figma", "prompt": "What design tools do UX professionals use?"},
    {"brand": "HubSpot", "prompt": "What CRM is best for small businesses?"},
    {"brand": "Zapier", "prompt": "How can I automate repetitive tasks between apps?"},
]

DEFAULT_RUNS = 3
CONFIRM_THRESHOLD = 30


def estimate_calls(models: List[str], runs: int, prompts: int) -> int:
    return len(models) * runs * prompts


def run_single(
    prompt_text: str,
    brand: str,
    provider: str,
    model: str,
) -> Dict:
    """Call the model and detect brand mention. Returns result dict."""
    try:
        response = query_ai(prompt_text, provider, model, max_retries=2)
        mention_info = detect_brand_mention_deterministic(response, brand)
        return {
            "status": "ok",
            "mentioned": mention_info["brand_mentioned"],
            "position": mention_info["position"],
            "response_len": len(response),
        }
    except RateLimitExceededError as e:
        return {"status": "rate_limited", "mentioned": None, "error": str(e)[:120]}
    except LLMClientError as e:
        return {"status": "failed", "mentioned": None, "error": str(e)[:120]}


def compute_stability_metrics(
    run_results: List[List[Dict]],
) -> Dict:
    """
    run_results: list of runs, each run is a list of per-prompt result dicts
                 (same order as STABILITY_PROMPTS).

    Returns: per-model stats dict.
    """
    n_prompts = len(STABILITY_PROMPTS)
    n_runs = len(run_results)

    # Per-run visibility scores (% mentioned, successful calls only)
    run_scores = []
    for run in run_results:
        ok = [r for r in run if r.get("status") == "ok"]
        if ok:
            score = sum(1 for r in ok if r["mentioned"]) / len(ok) * 100.0
            run_scores.append(score)

    if not run_scores:
        return {
            "mean_score": None,
            "std_dev": None,
            "min_score": None,
            "max_score": None,
            "flip_rate": None,
            "n_runs": n_runs,
            "n_prompts_ok": 0,
        }

    mean_score = sum(run_scores) / len(run_scores)
    variance = sum((s - mean_score) ** 2 for s in run_scores) / len(run_scores)
    std_dev = math.sqrt(variance)

    # Flip rate: per prompt, fraction of prompt pairs (run_i vs run_j) that flipped
    flip_count = 0
    total_pairs = 0
    for p_idx in range(n_prompts):
        outcomes = []
        for run in run_results:
            r = run[p_idx]
            if r.get("status") == "ok":
                outcomes.append(r["mentioned"])
        # Check if any pair differs
        for i in range(len(outcomes)):
            for j in range(i + 1, len(outcomes)):
                total_pairs += 1
                if outcomes[i] != outcomes[j]:
                    flip_count += 1

    flip_rate = flip_count / total_pairs if total_pairs > 0 else 0.0

    return {
        "mean_score": mean_score,
        "std_dev": std_dev,
        "min_score": min(run_scores),
        "max_score": max(run_scores),
        "run_scores": run_scores,
        "flip_rate": flip_rate,
        "n_runs": n_runs,
        "n_prompts_ok": sum(
            1 for run in run_results for r in run if r.get("status") == "ok"
        ),
    }


def _pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1f}%"


def print_stability_report(
    model_key: str,
    metrics: Dict,
    within_noise_threshold: float = 5.0,
) -> None:
    std = metrics.get("std_dev")
    within_noise = std is not None and std <= within_noise_threshold
    noise_note = (
        "within run-to-run noise" if within_noise else "may reflect real variance"
    )

    print(f"\n  Model: {model_key}")
    print(f"    Mean visibility score : {_pct(metrics.get('mean_score'))}")
    print(f"    Std deviation         : {_pct(metrics.get('std_dev'))} — {noise_note}")
    print(f"    Range (min–max)       : {_pct(metrics.get('min_score'))} – {_pct(metrics.get('max_score'))}")
    if metrics.get("run_scores"):
        run_strs = [_pct(s) for s in metrics["run_scores"]]
        print(f"    Per-run scores        : {' | '.join(run_strs)}")
    print(f"    Mention-flip rate     : {_pct(metrics.get('flip_rate'))} of cross-run prompt pairs")
    print(f"    Calls completed       : {metrics.get('n_prompts_ok', 0)}")
    if within_noise:
        print(f"    ✓ Differences are within run-to-run noise (σ ≤ {within_noise_threshold}%)")
    else:
        print(f"    ⚠ Differences exceed noise threshold (σ > {within_noise_threshold}%)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stability eval: repeat fixed prompts N times per model."
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of provider:model strings.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of repetitions per prompt (default {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=300,
        help="Hard cap on total API calls.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=f"Skip confirmation when estimated calls exceed {CONFIRM_THRESHOLD}.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Write JSON results to DIR/stability_results.json.",
    )
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Error: no models specified.", file=sys.stderr)
        return 1

    n_prompts = len(STABILITY_PROMPTS)
    estimated = estimate_calls(models, args.runs, n_prompts)
    print(f"Stability eval: {len(models)} model(s), {args.runs} runs, {n_prompts} prompts")
    print(f"Estimated API calls: {estimated} (cap: {args.max_calls})")

    if estimated > CONFIRM_THRESHOLD and not args.yes:
        ans = input(f"\nThis will make ~{estimated} calls. Type 'yes' to proceed: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            return 0

    total_calls = 0
    all_model_metrics: Dict[str, Dict] = {}
    all_model_raw: Dict[str, List] = {}

    for model_str in models:
        try:
            provider, model_name = parse_provider_model(model_str)
        except ValueError as e:
            print(f"Skipping invalid model string '{model_str}': {e}", file=sys.stderr)
            continue

        print(f"\n── Model: {model_str} ──")
        run_results: List[List[Dict]] = []

        for run_idx in range(args.runs):
            run_row: List[Dict] = []
            print(f"  Run {run_idx + 1}/{args.runs}:", end="", flush=True)
            for p in STABILITY_PROMPTS:
                if total_calls >= args.max_calls:
                    print("\n  ⚠  Call cap reached. Stopping.", file=sys.stderr)
                    run_row.append({"status": "skipped", "mentioned": None})
                    continue
                result = run_single(p["prompt"], p["brand"], provider, model_name)
                run_row.append(result)
                total_calls += 1
                char = "✓" if result.get("mentioned") else ("✗" if result.get("status") == "ok" else "!")
                print(char, end="", flush=True)
            print()
            run_results.append(run_row)

        metrics = compute_stability_metrics(run_results)
        all_model_metrics[model_str] = metrics
        all_model_raw[model_str] = run_results

    print("\n" + "=" * 72)
    print("  AI Visibility Tracker — Stability Eval Report (Stage 4)")
    print("=" * 72)
    print(f"  Total API calls made: {total_calls}")

    for model_str, metrics in all_model_metrics.items():
        print_stability_report(model_str, metrics)

    report = {
        "models": models,
        "runs": args.runs,
        "prompts": [p["prompt"] for p in STABILITY_PROMPTS],
        "total_calls": total_calls,
        "max_calls": args.max_calls,
        "results": all_model_metrics,
    }

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "stability_results.json"
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nResults written to {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
