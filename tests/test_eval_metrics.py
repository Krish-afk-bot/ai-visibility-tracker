import pytest
from evals.metrics import (
    precision_recall_f1,
    cohens_kappa,
    set_precision_recall_f1,
    mention_count_metrics,
    position_metrics,
    sentiment_accuracy,
    macro_average,
    jaccard_similarity,
)


def test_precision_recall_f1_edge_cases():
    # Zero-division follows scikit-learn convention: undefined -> 0.0
    p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=0)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

    # No predicted positives -> precision undefined (0), recall 0 (no actual)
    p, r, f1 = precision_recall_f1(tp=0, fp=5, fn=0)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

    p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=5)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

    # Standard positive case
    p, r, f1 = precision_recall_f1(tp=8, fp=2, fn=2)
    assert p == 0.8
    assert r == 0.8
    assert pytest.approx(f1, 0.001) == 0.8


def test_cohens_kappa_identical():
    rater1 = ["pos", "neg", "neu", "pos", "neg"]
    rater2 = ["pos", "neg", "neu", "pos", "neg"]
    k = cohens_kappa(rater1, rater2)
    assert pytest.approx(k, 0.001) == 1.0


def test_cohens_kappa_disagreement():
    rater1 = ["pos", "pos", "pos", "pos"]
    rater2 = ["neg", "neg", "neg", "neg"]
    k = cohens_kappa(rater1, rater2)
    assert k <= 0.0


def test_cohens_kappa_empty():
    # Vacuous agreement on nothing -> 1.0 (matches implementation contract)
    assert cohens_kappa([], []) == 1.0


def test_set_precision_recall_f1():
    # Empty predicted and non-empty actual -> 0, 0, 0
    p, r, f1 = set_precision_recall_f1(set(), {"Competitor A", "Competitor B"})
    assert (p, r, f1) == (0.0, 0.0, 0.0)

    # Empty both -> zero-division convention -> 0, 0, 0
    p, r, f1 = set_precision_recall_f1(set(), set())
    assert (p, r, f1) == (0.0, 0.0, 0.0)

    # Partial overlap
    pred = {"A", "B"}
    actual = {"B", "C"}
    p, r, f1 = set_precision_recall_f1(pred, actual)
    assert p == 0.5
    assert r == 0.5
    assert pytest.approx(f1, 0.001) == 0.5


def test_mention_count_metrics():
    # Empty lists
    res = mention_count_metrics([], [])
    assert res == {"exact_match_accuracy": 0.0, "mean_absolute_error": 0.0, "n": 0}

    # Matches and offsets
    res = mention_count_metrics([1, 2, 3], [1, 1, 3])
    assert res["n"] == 3
    # exact matches: (1,1) and (3,3) -> 2/3
    assert pytest.approx(res["exact_match_accuracy"], 0.01) == 2.0 / 3.0
    # absolute errors: |1-1|=0, |2-1|=1, |3-3|=0 -> mean 1/3
    assert pytest.approx(res["mean_absolute_error"], 0.01) == 1.0 / 3.0


def test_position_metrics():
    # All None values -> n=0
    res = position_metrics([None, None], [None, None])
    assert res == {"exact_match_accuracy": 0.0, "within_1_accuracy": 0.0, "n": 0}

    # Realistic positions
    res = position_metrics([1, 2, 4], [1, 3, 2])
    assert res["n"] == 3
    # exact matches: only (1,1) -> 1/3
    assert pytest.approx(res["exact_match_accuracy"], 0.01) == 1.0 / 3.0
    # within 1: errors 0, 1, 2 -> 0 and 1 within 1 (2 out of 3)
    assert pytest.approx(res["within_1_accuracy"], 0.01) == 2.0 / 3.0


def test_sentiment_accuracy():
    # Empty lists -> dict with zero accuracy
    res = sentiment_accuracy([], [])
    assert res["accuracy"] == 0.0
    assert res["n"] == 0

    # Perfect match (labels must be full class names: positive/neutral/negative)
    res = sentiment_accuracy(["positive", "negative"], ["positive", "negative"])
    assert res["accuracy"] == 1.0
    assert res["n"] == 2

    # 50% match
    res = sentiment_accuracy(["positive", "negative"], ["positive", "positive"])
    assert res["accuracy"] == 0.5

    # Short/unknown labels are treated as unknown, not counted as correct
    res = sentiment_accuracy(["pos"], ["pos"])
    assert res["accuracy"] == 0.0
    assert res["n"] == 0


def test_macro_average():
    # Empty list
    assert macro_average([]) == (0.0, 0.0, 0.0)

    # Standard list
    scores = [(1.0, 0.5, 0.66), (0.5, 1.0, 0.66)]
    p, r, f1 = macro_average(scores)
    assert pytest.approx(p, 0.01) == 0.75
    assert pytest.approx(r, 0.01) == 0.75
    assert pytest.approx(f1, 0.01) == 0.66


def test_jaccard_similarity():
    # Both empty -> 1.0
    assert jaccard_similarity(set(), set()) == 1.0

    # No overlap -> 0.0
    assert jaccard_similarity({"A"}, {"B"}) == 0.0

    # Overlap
    assert pytest.approx(jaccard_similarity({"A", "B"}, {"B", "C"}), 0.01) == 1.0 / 3.0
