from unittest.mock import patch

from analysis import (
    analyze_sentiment_and_competitors,
    extract_bold_competitor_names,
    extract_markdown_table_first_column,
    fallback_extract_competitors,
)


def test_known_competitors_use_word_boundaries():
    text = "Asana leads sprints, but Asanas is not a product. Trello is close."

    found = fallback_extract_competitors(
        text, "HubSpot", known_competitors=["Asana", "Trello", "ClickUp"]
    )

    assert found[:2] == ["Asana", "Trello"]
    assert "ClickUp" not in found


def test_markdown_table_first_column_skips_header_and_separator():
    text = """
| Brand | Best for | Price |
| --- | --- | --- |
| Asana | Teams | $10 |
| **ClickUp** | Everything | $7 |
"""

    assert extract_markdown_table_first_column(text) == ["Asana", "ClickUp"]


def test_bold_names_are_extracted_but_generic_labels_are_dropped():
    text = "**Asana** and **Monday.com** are strong. **Best for** is a label, **Pricing** too."

    names = extract_bold_competitor_names(text)

    assert "Asana" in names
    assert "Monday.com" in names
    assert "Best for" not in names
    assert "Pricing" not in names


def test_fallback_never_reports_the_target_brand_or_its_aliases():
    text = "**HubSpot** vs **Salesforce**. HubSpot is the target; HubSpot CRM is an alias."

    found = fallback_extract_competitors(
        text, "HubSpot", aliases=["HubSpot CRM"]
    )

    assert "HubSpot" not in found
    assert "HubSpot CRM" not in found
    assert "Salesforce" in found


def test_fallback_combines_known_table_and_bold_without_duplicates():
    text = (
        "Known: Salesforce.\n\n"
        "| Brand | Notes |\n"
        "|---|---|\n"
        "| Zendesk | x |\n"
        "| Salesforce | y |\n\n"
        "**Intercom** is another."
    )

    found = fallback_extract_competitors(text, "HubSpot", ["Salesforce"])

    assert found[0] == "Salesforce"  # configured competitors keep their order
    assert {"salesforce", "zendesk", "intercom"} <= {name.lower() for name in found}
    assert len(found) == len({name.lower() for name in found})


def test_analyzer_empty_competitor_list_is_filled_deterministically():
    with patch(
        "analysis.query_ai",
        return_value='{"sentiment": "positive", "competitors": []}',
    ):
        analysis = analyze_sentiment_and_competitors(
            ai_response="**Asana** and **Trello** are options.",
            brand_name="HubSpot",
            analyzer_model="groq:llama-3.1-8b-instant",
            known_competitors=["Asana"],
        )

    assert analysis["analyzer_status"] == "ok"
    assert "Asana" in analysis["competitors"]
    assert "Trello" in analysis["competitors"]


def test_analyzer_failure_still_counts_known_competitors():
    with patch("analysis.query_ai", side_effect=RuntimeError("429 rate limited")):
        analysis = analyze_sentiment_and_competitors(
            ai_response="Try Asana or Trello for this.",
            brand_name="HubSpot",
            analyzer_model="groq:llama-3.1-8b-instant",
            known_competitors=["Asana", "Trello"],
        )

    assert analysis["analyzer_status"] == "failed"
    assert analysis["competitors"] == ["Asana", "Trello"]
    assert analysis["sentiment"] == "unknown"
