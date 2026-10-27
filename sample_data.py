"""Helpers that keep the demo dashboard isolated from real scans.

The demo dataset belongs to the app, but its identity lives here so the
sidebar configuration can never relabel it, and loading it never touches the
database.
"""

from collections.abc import Iterable, MutableMapping
from copy import deepcopy
from typing import Any, Dict

SAMPLE_BRAND = "HubSpot"
SAMPLE_DATA_LABEL = "Sample data, no API calls were made."


def resolve_display_brand(is_sample: bool, configured_brand: str) -> str:
    """Return the brand a view should display.

    Sample results always render as their own brand, regardless of what the
    user typed in the sidebar.
    """
    if is_sample:
        return SAMPLE_BRAND
    return (configured_brand or "").strip()


def load_sample_into_state(
    state: MutableMapping[str, Any],
    sample_results: Iterable[Dict[str, Any]],
) -> None:
    """Load the demo dataset into session state.

    Deliberately leaves ``scan_brand`` untouched and never calls the database:
    the sample is a view, not a scan. ``scan_id`` is cleared so call accounting
    never looks up a real scan that is not being displayed.
    """
    state["scan_results"] = deepcopy(list(sample_results))
    state["is_sample"] = True
    state["scan_id"] = None
