"""
evals/metrics.py — shared metric calculations for the eval harness.

All functions are pure (no I/O, no network). Tested by tests/test_eval_metrics.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Basic precision / recall / F1
# ---------------------------------------------------------------------------

def precision_recall_f1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    """Return (precision, recall, F1).

    Edge-cases follow scikit-learn convention:
    - Zero denominator for precision → 0.0 (no positives predicted)
    - Zero denominator for recall   → 0.0 (no actual positives)
    - F1 undefined when P+R == 0   → 0.0
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def set_precision_recall_f1(
    predicted: Set[str], actual: Set[str]
) -> Tuple[float, float, float]:
    """Compute P/R/F1 for two sets of strings (case-insensitive)."""
    pred_lower = {s.lower() for s in predicted}
    actual_lower = {s.lower() for s in actual}
    tp = len(pred_lower & actual_lower)
    fp = len(pred_lower - actual_lower)
    fn = len(actual_lower - pred_lower)
    return precision_recall_f1(tp, fp, fn)


def micro_average(
    per_case: List[Tuple[int, int, int]]
) -> Tuple[float, float, float]:
    """Micro-average P/R/F1 from a list of (tp, fp, fn) tuples."""
    tp = sum(t[0] for t in per_case)
    fp = sum(t[1] for t in per_case)
    fn = sum(t[2] for t in per_case)
    return precision_recall_f1(tp, fp, fn)


def macro_average(
    per_case: List[Tuple[float, float, float]]
) -> Tuple[float, float, float]:
    """Macro-average P/R/F1 from a list of (precision, recall, f1) tuples."""
    if not per_case:
        return 0.0, 0.0, 0.0
    p = sum(t[0] for t in per_case) / len(per_case)
    r = sum(t[1] for t in per_case) / len(per_case)
    f = sum(t[2] for t in per_case) / len(per_case)
    return p, r, f


# ---------------------------------------------------------------------------
# Mention-count metrics
# ---------------------------------------------------------------------------

def mention_count_metrics(
    predictions: List[int], actuals: List[int]
) -> Dict[str, float]:
    """Exact-match accuracy and mean absolute error for mention counts."""
    if not predictions or len(predictions) != len(actuals):
        return {"exact_match_accuracy": 0.0, "mean_absolute_error": 0.0, "n": 0}
    n = len(predictions)
    exact = sum(1 for p, a in zip(predictions, actuals) if p == a)
    mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
    return {
        "exact_match_accuracy": exact / n,
        "mean_absolute_error": mae,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Position metrics
# ---------------------------------------------------------------------------

def position_metrics(
    predictions: List[Optional[int]], actuals: List[Optional[int]]
) -> Dict[str, float]:
    """Exact-match and within-1 accuracy for brand positions.

    Only cases where both prediction and actual are non-None are evaluated.
    """
    paired = [
        (p, a)
        for p, a in zip(predictions, actuals)
        if p is not None and a is not None
    ]
    if not paired:
        return {"exact_match_accuracy": 0.0, "within_1_accuracy": 0.0, "n": 0}
    n = len(paired)
    exact = sum(1 for p, a in paired if p == a)
    within1 = sum(1 for p, a in paired if abs(p - a) <= 1)
    return {
        "exact_match_accuracy": exact / n,
        "within_1_accuracy": within1 / n,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Sentiment metrics
# ---------------------------------------------------------------------------

SENTIMENT_CLASSES = ("positive", "neutral", "negative")


def sentiment_accuracy(
    predictions: List[Optional[str]], actuals: List[Optional[str]]
) -> Dict[str, object]:
    """Overall accuracy and per-class P/R/F1 for sentiment labels.

    Ignores pairs where either value is None/unknown.
    Returns confusion matrix as a nested dict.
    """
    valid = [
        (p.lower(), a.lower())
        for p, a in zip(predictions, actuals)
        if p and a and p.lower() in SENTIMENT_CLASSES and a.lower() in SENTIMENT_CLASSES
    ]
    if not valid:
        return {
            "accuracy": 0.0,
            "n": 0,
            "per_class": {},
            "confusion_matrix": {},
            "unknown_rate": 1.0,
        }

    n = len(valid)
    unknown_count = sum(
        1 for p, a in zip(predictions, actuals)
        if not p or p.lower() not in SENTIMENT_CLASSES
    )
    unknown_rate = unknown_count / len(predictions) if predictions else 0.0

    correct = sum(1 for p, a in valid if p == a)
    accuracy = correct / n

    confusion: Dict[str, Dict[str, int]] = {
        c: {c2: 0 for c2 in SENTIMENT_CLASSES} for c in SENTIMENT_CLASSES
    }
    for p, a in valid:
        confusion[a][p] += 1  # rows=actual, cols=predicted

    per_class: Dict[str, Dict[str, float]] = {}
    for cls in SENTIMENT_CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in SENTIMENT_CLASSES if other != cls)
        fn = sum(confusion[cls][other] for other in SENTIMENT_CLASSES if other != cls)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1}

    return {
        "accuracy": accuracy,
        "n": n,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "unknown_rate": unknown_rate,
    }


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def cohens_kappa(
    labels_a: List[str], labels_b: List[str], classes: Optional[List[str]] = None
) -> float:
    """Compute Cohen's kappa for two lists of categorical labels.

    Returns 1.0 for identical inputs, 0.0 for chance agreement,
    and negative values for systematic disagreement.
    Returns 1.0 when both annotators always agree (trivially consistent).
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have the same length.")
    if not labels_a:
        return 1.0  # vacuously agree on nothing

    if classes is None:
        classes = sorted(set(labels_a) | set(labels_b))

    n = len(labels_a)
    # Observed agreement
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement
    pe = 0.0
    for cls in classes:
        pa = labels_a.count(cls) / n
        pb = labels_b.count(cls) / n
        pe += pa * pb

    if pe == 1.0:
        return 1.0  # Perfect expected agreement (all same class)
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------------------
# Jaccard similarity for set comparison (competitors)
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard index = |A ∩ B| / |A ∪ B|. Returns 1.0 for both-empty."""
    a = {s.lower() for s in set_a}
    b = {s.lower() for s in set_b}
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Alias-aware competitor normalisation
# ---------------------------------------------------------------------------

def normalise_competitor_set(
    names: List[str],
    brand: str,
    aliases: Optional[List[str]] = None,
) -> Set[str]:
    """Lower-case a competitor list and remove the target brand / its aliases."""
    blocked = {brand.strip().lower()}
    if aliases:
        blocked.update(a.strip().lower() for a in aliases if a and a.strip())
    result: Set[str] = set()
    for name in names:
        if isinstance(name, str):
            n = name.strip().lower()
            if n and n not in blocked:
                result.add(n)
    return result


# ---------------------------------------------------------------------------
# Tag-breakdown helper
# ---------------------------------------------------------------------------

def breakdown_by_tag(
    cases: List[Dict],
    metric_fn,
) -> Dict[str, Dict]:
    """Run metric_fn(subset_of_cases) for each unique tag across all cases.

    metric_fn receives a list of case dicts and returns a metrics dict.
    """
    tags: Dict[str, List[Dict]] = {}
    for case in cases:
        for tag in case.get("tags", []):
            tags.setdefault(tag, []).append(case)
    return {tag: metric_fn(subset) for tag, subset in sorted(tags.items())}


# ---------------------------------------------------------------------------
# Most-frequent false-positive helper
# ---------------------------------------------------------------------------

def most_frequent_false_positives(
    fp_lists: List[List[str]], top_n: int = 20
) -> List[Tuple[str, int]]:
    """Count and rank the most common false-positive competitor strings."""
    counts: Dict[str, int] = {}
    for lst in fp_lists:
        for name in lst:
            key = name.strip().lower()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])[:top_n]
