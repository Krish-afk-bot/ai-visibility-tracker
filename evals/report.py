#!/usr/bin/env python3
"""
evals/report.py — Stage 6: generate timestamped report.md and report.html.

Usage:
    cd ai_visibility_tracker
    python evals/report.py --offline evals/results/offline_results.json
    python evals/report.py --offline ... --analyzer ... --stability ...
    python evals/report.py --offline ... --baseline evals/baseline.json
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _HERE / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _delta(current: Optional[float], baseline: Optional[float]) -> str:
    """Return a delta string like '+2.3 pp' or '–1.1 pp'."""
    if current is None or baseline is None:
        return ""
    diff = (current - baseline) * 100
    sign = "+" if diff >= 0 else "−"
    return f" ({sign}{abs(diff):.1f} pp vs baseline)"


def load_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_markdown_report(
    offline: Optional[Dict],
    analyzer: Optional[Dict],
    stability: Optional[Dict],
    baseline: Optional[Dict],
    config: Dict,
) -> str:
    lines: List[str] = []
    ts = config.get("timestamp", "")

    lines.append(f"# AI Visibility Tracker — Eval Report")
    lines.append(f"\n**Generated:** {ts}  ")
    lines.append(f"**Analyzer model:** `{config.get('analyzer_model', 'N/A')}`  ")
    lines.append(f"**Date:** {config.get('date', '')}  \n")

    # ── Headline metrics ────────────────────────────────────────────────────
    lines.append("## Headline Metrics\n")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")

    from evals.thresholds import load_thresholds
    thresholds = load_thresholds()

    if offline:
        det = offline.get("mention", {}).get("detection", {})
        pos = offline.get("mention", {}).get("position", {})
        comp = offline.get("competitor_fallback", {}).get("micro", {})

        m_f1 = det.get("f1")
        p_w1 = pos.get("within_1_accuracy")
        c_f1 = comp.get("f1")

        bl_m_f1 = baseline.get("mention_f1") if baseline else None
        bl_p_w1 = baseline.get("position_within1") if baseline else None
        bl_c_f1 = baseline.get("competitor_f1") if baseline else None

        thr_m = thresholds.get("mention_f1", 0.95)
        thr_p = thresholds.get("position_within1", 0.80)
        thr_c = thresholds.get("competitor_fallback_f1", 0.70)

        def _status(val, thr):
            if val is None:
                return "⚪ N/A"
            return "✅ PASS" if val >= thr else "❌ FAIL"

        lines.append(f"| Mention F1 | {_pct(m_f1)}{_delta(m_f1, bl_m_f1)} | ≥ {_pct(thr_m)} | {_status(m_f1, thr_m)} |")
        lines.append(f"| Position within-1 | {_pct(p_w1)}{_delta(p_w1, bl_p_w1)} | ≥ {_pct(thr_p)} | {_status(p_w1, thr_p)} |")
        lines.append(f"| Competitor fallback F1 | {_pct(c_f1)}{_delta(c_f1, bl_c_f1)} | ≥ {_pct(thr_c)} | {_status(c_f1, thr_c)} |")

    if analyzer:
        sent_acc = analyzer.get("metrics", {}).get("sentiment", {}).get("accuracy")
        bl_sent = baseline.get("sentiment_accuracy") if baseline else None
        thr_sent = thresholds.get("sentiment_accuracy", 0.70)
        lines.append(f"| Sentiment accuracy | {_pct(sent_acc)}{_delta(sent_acc, bl_sent)} | ≥ {_pct(thr_sent)} | {_status(sent_acc, thr_sent)} |")
        calls = analyzer.get("calls_made", 0)
        lines.append(f"| Analyzer calls made | {calls} | — | — |")

    lines.append("")

    # ── Offline details ─────────────────────────────────────────────────────
    if offline:
        lines.append("## Offline Evaluation Details\n")
        det = offline.get("mention", {}).get("detection", {})
        cnt = offline.get("mention", {}).get("count", {})
        pos = offline.get("mention", {}).get("position", {})
        comp_micro = offline.get("competitor_fallback", {}).get("micro", {})
        comp_macro = offline.get("competitor_fallback", {}).get("macro", {})

        lines.append("### Mention Detection")
        lines.append(f"- Precision: {_pct(det.get('precision'))}, Recall: {_pct(det.get('recall'))}, F1: {_pct(det.get('f1'))}")
        lines.append(f"- TP: {det.get('tp', 0)}, FP: {det.get('fp', 0)}, FN: {det.get('fn', 0)}")
        fps = det.get("false_positive_ids", [])
        fns = det.get("false_negative_ids", [])
        if fps:
            lines.append(f"- **False positive case IDs:** {', '.join(fps)}")
        if fns:
            lines.append(f"- **False negative case IDs:** {', '.join(fns)}")

        lines.append("\n### Mention Count")
        lines.append(f"- Exact-match accuracy: {_pct(cnt.get('exact_match_accuracy'))}")
        lines.append(f"- Mean absolute error: {cnt.get('mean_absolute_error', 0):.3f}  (N={cnt.get('n', 0)})")

        lines.append("\n### Position (mentioned cases only)")
        lines.append(f"- Exact-match: {_pct(pos.get('exact_match_accuracy'))}, Within-1: {_pct(pos.get('within_1_accuracy'))}  (N={pos.get('n', 0)})")

        lines.append("\n### Competitor Fallback")
        lines.append("| Avg | Precision | Recall | F1 |")
        lines.append("|-----|-----------|--------|----|")
        lines.append(f"| Micro | {_pct(comp_micro.get('precision'))} | {_pct(comp_micro.get('recall'))} | {_pct(comp_micro.get('f1'))} |")
        lines.append(f"| Macro | {_pct(comp_macro.get('precision'))} | {_pct(comp_macro.get('recall'))} | {_pct(comp_macro.get('f1'))} |")

        top_fps = offline.get("competitor_fallback", {}).get("top_false_positives", [])
        if top_fps:
            lines.append("\n**Most frequent false-positive competitor strings** (consider adding to GENERIC_COMPETITOR_TERMS):")
            for name, count in top_fps[:10]:
                lines.append(f"  - `{name}` ({count}×)")

        # Tag breakdown
        lines.append("\n### Per-Tag Breakdown\n")
        by_tag = offline.get("by_tag", {})
        mention_by_tag = by_tag.get("mention", {})
        comp_by_tag = by_tag.get("competitor", {})

        if mention_by_tag:
            lines.append("**Mention Detection by Tag**")
            lines.append("| Tag | Precision | Recall | F1 |")
            lines.append("|-----|-----------|--------|----|")
            for tag, m in sorted(mention_by_tag.items()):
                lines.append(
                    f"| {tag} | {_pct(m.get('precision'))} | {_pct(m.get('recall'))} | {_pct(m.get('f1'))} |"
                )
            lines.append("")

        if comp_by_tag:
            lines.append("**Competitor Fallback by Tag**")
            lines.append("| Tag | Precision | Recall | F1 |")
            lines.append("|-----|-----------|--------|----|")
            for tag, m in sorted(comp_by_tag.items()):
                lines.append(
                    f"| {tag} | {_pct(m.get('precision'))} | {_pct(m.get('recall'))} | {_pct(m.get('f1'))} |"
                )
            lines.append("")

    # ── Analyzer details ────────────────────────────────────────────────────
    if analyzer:
        lines.append("## Analyzer Evaluation Details (Live)\n")
        metrics = analyzer.get("metrics", {})
        sent = metrics.get("sentiment", {})
        lines.append(f"- **Calls made:** {analyzer.get('calls_made', 0)} / {analyzer.get('max_calls', 40)}")
        lines.append(f"- **Unknown/fail rate:** {_pct(metrics.get('unknown_rate'))}")
        lines.append(f"- **Sentiment accuracy:** {_pct(sent.get('accuracy'))}  (N={sent.get('n', 0)})")

        per_class = sent.get("per_class", {})
        if per_class:
            lines.append("\n| Class | Precision | Recall | F1 |")
            lines.append("|-------|-----------|--------|----|")
            for cls, m in per_class.items():
                lines.append(f"| {cls} | {_pct(m.get('precision'))} | {_pct(m.get('recall'))} | {_pct(m.get('f1'))} |")

        comp_a = metrics.get("competitor_analyzer", {}).get("micro", {})
        comp_c = metrics.get("competitor_combined", {}).get("micro", {})
        lines.append(f"\n- **Competitor (analyzer only) micro F1:** {_pct(comp_a.get('f1'))}")
        lines.append(f"- **Competitor (analyzer+fallback) micro F1:** {_pct(comp_c.get('f1'))}")

        agreement = analyzer.get("agreement")
        if agreement:
            lines.append(f"\n- **Sentiment kappa (vs compare model):** {agreement.get('sentiment_kappa', 0):.3f}")
            lines.append(f"- **Competitor avg Jaccard (vs compare model):** {agreement.get('competitor_avg_jaccard', 0):.3f}")
        lines.append("")

    # ── Stability details ────────────────────────────────────────────────────
    if stability:
        lines.append("## Stability Evaluation Details (Live)\n")
        results = stability.get("results", {})
        lines.append(f"- **Runs per model:** {stability.get('runs', 'N/A')}")
        lines.append(f"- **Total calls:** {stability.get('total_calls', 0)}")
        lines.append("")
        lines.append("| Model | Mean Score | Std Dev | Min | Max | Flip Rate |")
        lines.append("|-------|-----------|---------|-----|-----|-----------|")
        for model_key, m in results.items():
            lines.append(
                f"| {model_key} | {_pct(m.get('mean_score'), 1) if m.get('mean_score') is not None else 'N/A'} | "
                f"{_pct(m.get('std_dev'), 1) if m.get('std_dev') is not None else 'N/A'} | "
                f"{_pct(m.get('min_score'), 1) if m.get('min_score') is not None else 'N/A'} | "
                f"{_pct(m.get('max_score'), 1) if m.get('max_score') is not None else 'N/A'} | "
                f"{_pct(m.get('flip_rate'), 1) if m.get('flip_rate') is not None else 'N/A'} |"
            )
        lines.append("")

    # ── Configuration used ───────────────────────────────────────────────────
    lines.append("## Configuration\n")
    lines.append(f"- **Analyzer model:** `{config.get('analyzer_model', 'N/A')}`")
    lines.append(f"- **Report date:** {config.get('date', '')}")
    lines.append(f"- **Golden dataset:** `evals/golden/cases.jsonl`")
    lines.append("")

    # ── Limitations ─────────────────────────────────────────────────────────
    lines.append("## What These Numbers Do and Don't Tell You\n")
    lines.append(
        "> **Small sample:** The golden dataset contains 40 synthetic cases. "
        "These numbers reflect performance on a curated set of scenarios, "
        "not a statistically representative sample of all possible model outputs.\n"
    )
    lines.append(
        "> **Synthetic answers:** All golden answers were hand-crafted to cover "
        "known failure modes. Real model outputs may contain patterns not represented here.\n"
    )
    lines.append(
        "> **English focus:** Most cases are in English. The `non_english` tag covers "
        "one Spanish and one German case only. Performance on other languages is untested.\n"
    )
    lines.append(
        "> **Deterministic functions only:** The offline eval tests rule-based "
        "detection. The live eval tests one specific analyzer model — results will "
        "differ with a different model or temperature.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report builder (self-contained, HTML-escaped)
# ---------------------------------------------------------------------------

def build_html_report(markdown_content: str, title: str = "Eval Report") -> str:
    """Convert the markdown to a self-contained HTML report with app visual tokens."""
    escaped_content = html.escape(markdown_content)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2d3040;
    --accent: #7c6af7;
    --text: #e6e8f0;
    --muted: #8b8fa8;
    --pass: #22c55e;
    --fail: #ef4444;
    --warn: #f59e0b;
    --font: 'Segoe UI', system-ui, sans-serif;
    --mono: 'Cascadia Code', 'Fira Code', monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    line-height: 1.7;
    padding: 2rem;
    max-width: 900px;
    margin: 0 auto;
  }}
  h1 {{ color: var(--accent); font-size: 1.8rem; margin-bottom: 0.5rem; }}
  h2 {{ color: var(--text); font-size: 1.3rem; margin: 2rem 0 0.75rem;
        border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
  h3 {{ color: var(--muted); font-size: 1rem; margin: 1.25rem 0 0.5rem; }}
  p {{ margin: 0.5rem 0; }}
  code {{ font-family: var(--mono); background: var(--surface);
           border: 1px solid var(--border); border-radius: 4px;
           padding: 0.1em 0.4em; font-size: 0.875em; }}
  pre {{ background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 1rem; overflow-x: auto;
          font-family: var(--mono); font-size: 0.875em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0;
            font-size: 0.9rem; }}
  th {{ background: var(--surface); color: var(--muted); padding: 0.5rem 0.75rem;
         text-align: left; border: 1px solid var(--border); }}
  td {{ padding: 0.5rem 0.75rem; border: 1px solid var(--border); }}
  tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
  blockquote {{
    border-left: 3px solid var(--accent);
    padding: 0.75rem 1rem;
    background: var(--surface);
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: var(--muted);
    font-style: italic;
  }}
  ul, ol {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
  li {{ margin: 0.25rem 0; }}
  .pass {{ color: var(--pass); }}
  .fail {{ color: var(--fail); }}
  .warn {{ color: var(--warn); }}
  strong {{ color: var(--text); font-weight: 600; }}
</style>
</head>
<body>
<pre>
{escaped_content}
</pre>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Save baseline
# ---------------------------------------------------------------------------

def save_baseline(offline: Optional[Dict], analyzer: Optional[Dict], out_path: Path) -> None:
    """Write a baseline.json from current results."""
    baseline: Dict[str, Any] = {}
    if offline:
        det = offline.get("mention", {}).get("detection", {})
        pos = offline.get("mention", {}).get("position", {})
        comp = offline.get("competitor_fallback", {}).get("micro", {})
        baseline["mention_f1"] = det.get("f1")
        baseline["mention_precision"] = det.get("precision")
        baseline["mention_recall"] = det.get("recall")
        baseline["position_exact"] = pos.get("exact_match_accuracy")
        baseline["position_within1"] = pos.get("within_1_accuracy")
        baseline["competitor_f1"] = comp.get("f1")
    if analyzer:
        sent = analyzer.get("metrics", {}).get("sentiment", {})
        baseline["sentiment_accuracy"] = sent.get("accuracy")
    baseline["timestamp"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Baseline saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate eval report.md and report.html.")
    parser.add_argument("--offline", metavar="FILE", help="Path to offline_results.json")
    parser.add_argument("--analyzer", metavar="FILE", help="Path to analyzer_results.json")
    parser.add_argument("--stability", metavar="FILE", help="Path to stability_results.json")
    parser.add_argument("--baseline", metavar="FILE", help="Path to baseline.json for deltas")
    parser.add_argument("--save-baseline", action="store_true", help="Save current results as baseline.json")
    parser.add_argument("--output-dir", metavar="DIR", default=None, help="Output directory (default: auto-timestamped)")
    args = parser.parse_args(argv)

    offline = load_json(args.offline) if args.offline else None
    analyzer = load_json(args.analyzer) if args.analyzer else None
    stability = load_json(args.stability) if args.stability else None
    baseline = load_json(args.baseline) if args.baseline else None

    if not offline and not analyzer and not stability:
        print("Error: provide at least one of --offline, --analyzer, --stability.", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    config = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
        "date": now.strftime("%Y-%m-%d"),
        "analyzer_model": (analyzer or {}).get("analyzer_model", os.environ.get("ANALYZER_MODEL", "N/A")),
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    md_content = build_markdown_report(offline, analyzer, stability, baseline, config)
    html_content = build_html_report(md_content, title=f"Eval Report — {config['date']}")

    md_path = out_dir / "report.md"
    html_path = out_dir / "report.html"

    md_path.write_text(md_content, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")

    print(f"Report written to:")
    print(f"  {md_path}")
    print(f"  {html_path}")

    if args.save_baseline:
        baseline_path = _HERE / "baseline.json"
        save_baseline(offline, analyzer, baseline_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
