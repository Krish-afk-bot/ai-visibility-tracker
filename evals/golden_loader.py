"""
evals/golden_loader.py — load and filter the golden dataset.

Usage:
    from evals.golden_loader import load_golden_cases

    cases = load_golden_cases()                   # verified only (default)
    cases = load_golden_cases(include_unverified=True)  # all 40
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

GOLDEN_PATH = Path(__file__).parent / "golden" / "cases.jsonl"


def load_golden_cases(
    include_unverified: bool = False,
    path: Optional[Path] = None,
) -> List[Dict]:
    """Load cases from cases.jsonl.

    Args:
        include_unverified: when True, includes cases whose label_status is
            'needs_human_review'. Default False (production scoring).
        path: override path to the JSONL file (for testing).

    Returns:
        List of case dicts, each guaranteed to have: id, brand, aliases,
        prompt, answer_text, labels, tags, label_status.
    """
    fpath = path or GOLDEN_PATH
    if not fpath.exists():
        raise FileNotFoundError(f"Golden dataset not found at {fpath}")

    cases: List[Dict] = []
    with open(fpath, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {fpath}: {exc}"
                ) from exc
            _validate_case(case, lineno)
            status = case.get("label_status", "needs_human_review")
            if include_unverified or status == "verified":
                cases.append(case)

    return cases


def _validate_case(case: Dict, lineno: int) -> None:
    required = ("id", "brand", "aliases", "prompt", "answer_text", "labels", "tags")
    for field in required:
        if field not in case:
            raise ValueError(
                f"Case on line {lineno} missing required field '{field}': {case.get('id', '?')}"
            )
    labels = case["labels"]
    label_fields = ("mentioned", "mention_count", "position", "sentiment", "competitors")
    for lf in label_fields:
        if lf not in labels:
            raise ValueError(
                f"Case '{case['id']}' labels missing field '{lf}'"
            )


def summarise_dataset(cases: List[Dict]) -> Dict:
    """Return summary statistics for a set of cases."""
    tags: Dict[str, int] = {}
    for case in cases:
        for tag in case.get("tags", []):
            tags[tag] = tags.get(tag, 0) + 1

    verified = sum(1 for c in cases if c.get("label_status") == "verified")
    return {
        "total": len(cases),
        "verified": verified,
        "unverified": len(cases) - verified,
        "tag_counts": dict(sorted(tags.items())),
        "brands": sorted({c["brand"] for c in cases}),
    }
