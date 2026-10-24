"""Backward compatibility wrapper redirecting to analysis.py."""
from analysis import (
    analyze_brand_mention,
    extract_competitors,
    detect_brand_mention_deterministic
)
from llm_client import query_ai

__all__ = [
    "query_ai",
    "analyze_brand_mention",
    "extract_competitors",
    "detect_brand_mention_deterministic"
]
