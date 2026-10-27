from analysis import compute_brand_report_metrics


def _result(prompt, brand_mentioned, competitors, mention_count=None):
    return {
        "status": "ok",
        "prompt": prompt,
        "provider": "groq",
        "model": "llama",
        "brand_mentioned": brand_mentioned,
        "mention_count": mention_count if mention_count is not None else (1 if brand_mentioned else 0),
        "position": 1 if brand_mentioned else None,
        "sentiment": "neutral",
        "competitors": competitors,
        "analyzer_status": "ok",
        "scan_id": 1,
        "scan_created_at": "2026-10-01",
    }


def test_share_of_voice_counts_answers_once_per_brand():
    results = [
        # Alpha is repeated six times in the answer but counts once.
        _result("A", True, ["Beta", "beta"], mention_count=6),
        # Duplicate spellings in one answer count once.
        _result("B", False, ["Beta", "Gamma"]),
        _result("C", True, [], mention_count=2),
        _result("D", False, []),
    ]

    metrics = compute_brand_report_metrics(results, target_brand="Alpha")

    # Union of answers naming Alpha or any competitor: A, B, C -> n = 3.
    assert metrics["sov_n"] == 3
    assert metrics["answers_naming_any_tracked_brand"] == 3
    assert metrics["target_brand_mentions"] == 2  # answers, not 8 occurrences
    assert metrics["share_of_voice_pct"] == 66.7

    beta = next(r for r in metrics["comparison_table"] if r["brand"] == "Beta")
    gamma = next(r for r in metrics["comparison_table"] if r["brand"] == "Gamma")

    # Competitors and the target use the same unit.
    assert beta["mentions"] == 2
    assert beta["share_of_mentions_pct"] == 66.7
    assert gamma["mentions"] == 1
    assert gamma["share_of_mentions_pct"] == 33.3


def test_repeated_mentions_never_inflate_share_of_voice():
    results = [
        _result("A", True, ["Beta"], mention_count=50),
        _result("B", False, ["Beta"]),
    ]

    metrics = compute_brand_report_metrics(results, target_brand="Alpha")

    # Occurrence-based math would report 50/51 ~= 98%; answers report 1 of 2.
    assert metrics["share_of_voice_pct"] == 50.0
    assert metrics["target_brand_mentions"] == 1
    assert metrics["sov_n"] == 2


def test_share_of_voice_is_unavailable_when_no_brand_appears():
    results = [_result("A", False, [])]

    metrics = compute_brand_report_metrics(results, target_brand="Alpha")

    assert metrics["sov_n"] == 0
    assert metrics["share_of_voice_pct"] is None
    assert metrics["sov_available"] is False
