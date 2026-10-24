import os
import json
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import altair as alt
from dotenv import load_dotenv

# Load environment variables. override=True makes "streamlit run" pick up
# keys and model IDs edited in .env without restarting the server.
load_dotenv(override=True)

import hmac
from config import get_setting, get_int_setting

from database import (
    init_db,
    create_scan,
    save_result,
    update_scan_score,
    get_scan_results,
    get_cached_response,
    set_cached_response,
    get_all_scans,
    get_scans_and_results_in_range,
    get_database_bytes,
    restore_database_from_bytes,
    create_project,
    get_projects,
    get_project,
    update_project,
    delete_project,
    create_prompt,
    get_prompts,
    update_prompt,
    toggle_prompt_active,
    delete_prompt,
    bulk_add_prompts,
    VALID_PROMPT_TAGS,
)
from llm_client import (
    query_ai,
    parse_provider_model,
    get_available_models as _raw_get_available_models,
    get_analyzer_model,
    validate_analyzer_config,
    LLMClientError,
    ConfigurationError,
    ModelUnavailableError,
    RateLimitExceededError,
)
from analysis import (
    detect_brand_mention_deterministic,
    analyze_sentiment_and_competitors,
    calculate_visibility_score,
    calculate_model_scores,
    find_model_disagreements,
    generate_comparison_summary,
    compute_brand_report_metrics,
    find_visibility_gaps,
    generate_prompt_ideas,
    generate_gap_recommendation,
)

logger = logging.getLogger(__name__)

# Run init_db once per process using st.cache_resource
@st.cache_resource
def cached_init_db():
    init_db()
    return True

cached_init_db()

# Cache get_available_models with TTL=60s so reruns don't inspect settings repeatedly
@st.cache_data(ttl=60)
def get_available_models() -> List[str]:
    return _raw_get_available_models()

# Page configuration
st.set_page_config(
    page_title="AI Visibility Tracker",
    layout="wide"
)

# Canonical example prompts used consistently across landing page buttons and samples
EXAMPLE_PROMPTS = [
    "What are the best CRM tools for growing startups?",
    "Recommend top project management software for agile teams.",
    "Which customer support platform has the best AI automation?",
    "Compare the top sales automation platforms for B2B teams."
]

# Canonical mock sample results demonstrating:
# 1) Disagreement on prompts
# 2) One model failing without breaking comparison
# 3) Mentions, rank positions, and positive sentiment
SAMPLE_RESULTS = [
    # Prompt 0: Best CRM tools
    {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "prompt": EXAMPLE_PROMPTS[0],
        "run_index": 1,
        "ai_response": "1. HubSpot: Excellent CRM for startups with strong marketing tools.\n2. Salesforce: Enterprise standard.\n3. Pipedrive: Focused on sales pipeline.",
        "brand_mentioned": True,
        "mention_count": 1,
        "sentiment": "positive",
        "position": 1,
        "competitors": ["Salesforce", "Pipedrive"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "prompt": EXAMPLE_PROMPTS[0],
        "run_index": 1,
        "ai_response": "Leading CRMs include Salesforce, Zoho CRM, and HubSpot. HubSpot provides an accessible free tier for small teams.",
        "brand_mentioned": True,
        "mention_count": 2,
        "sentiment": "positive",
        "position": 3,
        "competitors": ["Salesforce", "Zoho CRM"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    {
        "provider": "openrouter",
        "model": "mistralai/mistral-7b-instruct:free",
        "prompt": EXAMPLE_PROMPTS[0],
        "run_index": 1,
        "ai_response": None,
        "brand_mentioned": False,
        "mention_count": 0,
        "sentiment": "neutral",
        "position": None,
        "competitors": [],
        "status": "failed",
        "error": "HTTP 429 Rate limit exceeded on free endpoint",
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    # Prompt 1: Project management software (Disagreement: Gemini mentions, Groq does not!)
    {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "prompt": EXAMPLE_PROMPTS[1],
        "run_index": 1,
        "ai_response": "For agile teams, HubSpot Projects integrates with marketing, while Jira and Linear lead engineering teams.",
        "brand_mentioned": True,
        "mention_count": 1,
        "sentiment": "neutral",
        "position": 1,
        "competitors": ["Jira", "Linear"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "prompt": EXAMPLE_PROMPTS[1],
        "run_index": 1,
        "ai_response": "For agile teams, consider Jira for sprints, Asana for workflow management, and Monday.com for team flexibility.",
        "brand_mentioned": False,
        "mention_count": 0,
        "sentiment": "neutral",
        "position": None,
        "competitors": ["Jira", "Asana", "Monday.com"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    {
        "provider": "openrouter",
        "model": "mistralai/mistral-7b-instruct:free",
        "prompt": EXAMPLE_PROMPTS[1],
        "run_index": 1,
        "ai_response": "Jira and ClickUp dominate project management for agile development.",
        "brand_mentioned": False,
        "mention_count": 0,
        "sentiment": "neutral",
        "position": None,
        "competitors": ["Jira", "ClickUp"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    # Prompt 2: Customer support platform
    {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "prompt": EXAMPLE_PROMPTS[2],
        "run_index": 1,
        "ai_response": "Top platforms are Zendesk, Intercom, and HubSpot Service Hub with automated ticketing.",
        "brand_mentioned": True,
        "mention_count": 1,
        "sentiment": "positive",
        "position": 3,
        "competitors": ["Zendesk", "Intercom"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "prompt": EXAMPLE_PROMPTS[2],
        "run_index": 1,
        "ai_response": "Intercom and Zendesk are leaders in AI-driven customer support automation.",
        "brand_mentioned": False,
        "mention_count": 0,
        "sentiment": "neutral",
        "position": None,
        "competitors": ["Intercom", "Zendesk"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    },
    {
        "provider": "openrouter",
        "model": "mistralai/mistral-7b-instruct:free",
        "prompt": EXAMPLE_PROMPTS[2],
        "run_index": 1,
        "ai_response": "Zendesk provides the best automated AI agent features.",
        "brand_mentioned": False,
        "mention_count": 0,
        "sentiment": "neutral",
        "position": None,
        "competitors": ["Zendesk"],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None
    }
]

# Session state initialization
# Default to sample dashboard on first load
if "scan_results" not in st.session_state:
    st.session_state.scan_results = SAMPLE_RESULTS
    st.session_state.scan_brand = "HubSpot"
    st.session_state.is_sample = True
if "scan_brand" not in st.session_state:
    st.session_state.scan_brand = "HubSpot"
if "scan_id" not in st.session_state:
    st.session_state.scan_id = None
if "prompts_text" not in st.session_state:
    st.session_state.prompts_text = ""
if "validation_error" not in st.session_state:
    st.session_state.validation_error = None
if "is_sample" not in st.session_state:
    st.session_state.is_sample = True
if "calls_used" not in st.session_state:
    st.session_state.calls_used = 0
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = None
if "prompt_ideas" not in st.session_state:
    st.session_state.prompt_ideas = []


# Helper for running a single query unit (supports parallel execution)
def execute_query_unit(
    prompt: str,
    pm_str: str,
    run_idx: int,
    brand_name: str,
    aliases: List[str],
    known_competitors: List[str],
    analyzer_model: str,
    use_cache: bool
) -> Dict[str, Any]:
    provider, model = parse_provider_model(pm_str)
    ai_response = None
    status = "ok"
    error_msg = None
    cached_hit = False

    if use_cache:
        cached = get_cached_response(provider, model, prompt, run_idx)
        if cached:
            ai_response = cached
            cached_hit = True

    if not cached_hit:
        try:
            ai_response = query_ai(prompt, provider, model)
            if use_cache and ai_response:
                set_cached_response(provider, model, prompt, ai_response, run_idx)
        except (ConfigurationError, ModelUnavailableError, RateLimitExceededError, LLMClientError) as e:
            status = "failed"
            error_msg = str(e)
            logger.warning(f"Error querying {pm_str}: {e}")
        except Exception as e:
            status = "failed"
            error_msg = f"Unexpected failure: {str(e)}"
            logger.exception(f"Unexpected error querying {pm_str}")

    analyzer_status = "ok"
    analyzer_error = None

    # Analyze response if successful
    if status == "ok" and ai_response:
        mention_info = detect_brand_mention_deterministic(ai_response, brand_name, aliases)
        brand_mentioned = mention_info["brand_mentioned"]
        mention_count = mention_info["mention_count"]
        position = mention_info["position"]

        analysis = analyze_sentiment_and_competitors(
            ai_response=ai_response,
            brand_name=brand_name,
            analyzer_model=analyzer_model,
            brand_mentioned=brand_mentioned,
            known_competitors=known_competitors
        )
        sentiment = analysis.get("sentiment", "neutral")
        competitors = analysis.get("competitors", [])
        analyzer_status = analysis.get("analyzer_status", "ok")
        analyzer_error = analysis.get("analyzer_error")
    else:
        brand_mentioned = False
        mention_count = 0
        position = None
        sentiment = "neutral"
        competitors = []

    return {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "run_index": run_idx,
        "ai_response": ai_response,
        "brand_mentioned": brand_mentioned,
        "mention_count": mention_count,
        "sentiment": sentiment,
        "position": position,
        "competitors": competitors,
        "status": status,
        "error": error_msg,
        "analyzer_status": analyzer_status,
        "analyzer_error": analyzer_error,
        "cached": cached_hit
    }


def highlight_brand(text: str, brand_name: str, aliases: Optional[List[str]] = None) -> str:
    """Highlight brand and alias mentions with high-contrast HTML mark tags."""
    if not text:
        return ""
    terms = [brand_name.strip()]
    if aliases:
        for a in aliases:
            if a and a.strip() and a.strip().lower() not in [t.lower() for t in terms]:
                terms.append(a.strip())

    highlighted = text
    for term in terms:
        esc = re.escape(term)
        pref = r"\b" if re.match(r"^\w", term) else r"(?:^|\s)"
        suff = r"\b" if re.search(r"\w$", term) else r"(?:$|\s)"
        pattern = re.compile(rf"({pref})({esc})({suff})", re.IGNORECASE)
        highlighted = pattern.sub(r'\1<mark class="brand-mark">\2</mark>\3', highlighted)

    return highlighted


# ==========================================
# Presentation
# ==========================================
MODEL_COLORS = ["#3B5B8C", "#5E8C6A", "#A57C2B", "#8C8279"]
FONT_FAMILY = "'Schibsted Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif"

_PROVIDER_LABELS = {
    "openrouter": "OpenRouter",
    "groq": "Groq",
    "gemini": "Gemini",
}

_MODEL_ACRONYMS = {
    "gpt": "GPT",
    "oss": "OSS",
    "ai": "AI",
    "llm": "LLM",
    "gemma": "Gemma",
    "llama": "Llama",
    "qwen": "Qwen",
    "mistral": "Mistral",
    "gemini": "Gemini",
    "claude": "Claude",
    "nova": "Nova",
}


def provider_label(provider_model: str) -> str:
    provider = provider_model.split(":", 1)[0]
    return _PROVIDER_LABELS.get(provider, provider.capitalize())


def model_label(provider_model: str) -> str:
    """Turn a provider:model key into a short, readable name."""
    known = {
        "gemini:gemini-3.8-flash": "Gemini 3.8 Flash",
        "groq:openai/gpt-oss-20b": "GPT-OSS 20B",
        "groq:openai/gpt-oss-120b": "GPT-OSS 120B",
        "openrouter:google/gemma-4-31b-it:free": "Gemma 4 31B",
    }
    if provider_model in known:
        return known[provider_model]

    _, _, model = provider_model.partition(":")
    base = model.split("/")[-1].replace(":free", "").replace(":", " ")
    parts = [p for p in re.split(r"[-_\s]+", base) if p]
    words = []
    for part in parts:
        lower = part.lower()
        if lower in _MODEL_ACRONYMS:
            words.append(_MODEL_ACRONYMS[lower])
        elif re.fullmatch(r"\d+[a-z]", part):
            words.append(part.upper())
        elif part[0].isdigit():
            words.append(part)
        else:
            words.append(part.capitalize())
    return " ".join(words) or model


def model_color_map(model_keys) -> Dict[str, str]:
    ordered = sorted(set(model_keys))
    return {name: MODEL_COLORS[i % len(MODEL_COLORS)] for i, name in enumerate(ordered)}


def escape_text(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def build_lede(model_scores: Dict[str, Dict[str, Any]], brand_name: str) -> str:
    brand_mark = f'<mark class="brand-mark">{escape_text(brand_name)}</mark>'
    ranked = [
        (pm, stats) for pm, stats in model_scores.items()
        if stats.get("score") is not None and stats.get("successful_calls", 0) > 0
    ]
    if not ranked:
        return "No model returned an answer. Check provider access, then run the scan again."

    ranked.sort(key=lambda item: item[1]["score"], reverse=True)
    top_pm, top = ranked[0]

    if len(ranked) == 1:
        return (
            f"{model_label(top_pm)} names {brand_mark} in "
            f"{top['mentions']} of {top['successful_calls']} answers."
        )

    low_pm, low = ranked[-1]
    if top["mentions"] == low["mentions"]:
        return (
            f"{model_label(top_pm)} and {model_label(low_pm)} each name {brand_mark} in "
            f"{top['mentions']} of {top['successful_calls']} answers."
        )

    first = (
        f"{model_label(top_pm)} names {brand_mark} in "
        f"{top['mentions']} of {top['successful_calls']} answers."
    )
    if low["mentions"] == 0:
        second = f"{model_label(low_pm)} names it in none."
    else:
        second = (
            f"{model_label(low_pm)} names it in "
            f"{low['mentions']} of {low['successful_calls']}."
        )
    return f"{first} {second}"


def build_ranking_html(model_scores: Dict[str, Dict[str, Any]], colors: Dict[str, str]) -> str:
    ordered = sorted(
        model_scores.items(),
        key=lambda item: (
            item[1].get("score") is None,
            -(item[1].get("score") if item[1].get("score") is not None else 0),
        ),
    )
    rows = []
    for pm, stats in ordered:
        label = escape_text(model_label(pm))
        color = colors.get(pm, MODEL_COLORS[0])
        if stats.get("score") is None:
            failed = stats.get("failed_calls", 0)
            meta = (
                f"{failed} failed call{'s' if failed != 1 else ''}. "
                "See the answers for the error."
            )
            row = (
                f'<div class="rank-row" style="--identity:{color}">'
                f'<div class="rank-name">{label}</div>'
                f'<div class="rank-track"></div>'
                f'<div class="rank-pct failed">Failed</div>'
                f'<div class="rank-meta">{meta}</div>'
                f'</div>'
            )
        else:
            score = float(stats["score"])
            position = stats.get("avg_position")
            position_line = (
                f"Average position #{position:.1f}"
                if position is not None else "No mention rank"
            )
            meta = (
                f"{position_line}<br>"
                f"Mentioned in {stats['mentions']} of {stats['successful_calls']} answers"
            )
            row = (
                f'<div class="rank-row" style="--identity:{color}">'
                f'<div class="rank-name">{label}</div>'
                f'<div class="rank-track"><div class="rank-fill" style="width:{score:.1f}%"></div></div>'
                f'<div class="rank-pct">{score:.1f}%</div>'
                f'<div class="rank-meta">{meta}</div>'
                f'</div>'
            )
        rows.append(row)
    return '<div class="rank-list">' + "".join(rows) + "</div>"


def build_coverage_html(
    unique_prompts: List[str],
    unique_models: List[str],
    results: List[Dict[str, Any]],
    colors: Dict[str, str],
) -> str:
    headers = [
        '<th class="prompt">Prompt</th>'
    ]
    for pm in unique_models:
        headers.append(
            f'<th class="model" style="--identity:{colors.get(pm, MODEL_COLORS[0])}">'
            f'<span>{escape_text(model_label(pm))}</span>'
            f'<small>{escape_text(provider_label(pm))}</small>'
            f'</th>'
        )

    body = []
    for prompt in unique_prompts:
        cells = [f'<td class="prompt">{escape_text(prompt)}</td>']
        for pm in unique_models:
            matching = [
                r for r in results
                if r["prompt"] == prompt and f"{r['provider']}:{r['model']}" == pm
            ]
            if not matching:
                cells.append('<td><span class="no">Not available</span></td>')
                continue

            values = []
            for item in matching:
                if item["status"] == "failed":
                    values.append('<span class="fail">Failed</span>')
                elif item["brand_mentioned"]:
                    position = f", #{item['position']}" if item.get("position") else ""
                    values.append(f'<span class="yes">Yes{position}</span>')
                else:
                    values.append('<span class="no">No</span>')
            cells.append("<td>" + "<br>".join(values) + "</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="table-scroll"><table class="editorial-table coverage">'
        "<thead><tr>" + "".join(headers) + "</tr></thead>"
        "<tbody>" + "".join(body) + "</tbody>"
        "</table></div>"
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --paper: #F4F5F3;
        --graphite: #23272C;
        --pencil: #5F666D;
        --rule: #D5D9DC;
        --highlighter: #F7D546;
        --model-blue: #3B5B8C;
        --model-sage: #5E8C6A;
        --model-ochre: #A57C2B;
        --model-stone: #8C8279;
        --brick: #B3402F;
        --radius: 6px;
        --font: 'Schibsted Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }

    html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button,
    [data-baseweb="select"] {
        font-family: var(--font);
    }

    .stApp {
        background: var(--paper);
        color: var(--graphite);
        font-size: 16px;
        line-height: 1.55;
        font-variant-numeric: tabular-nums;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 48px;
        padding-bottom: 72px;
        padding-left: 40px;
        padding-right: 40px;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        font-family: var(--font) !important;
        color: var(--graphite) !important;
    }

    .stApp h1 {
        font-size: 40px !important;
        line-height: 1.1 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin: 0 0 10px 0 !important;
        padding: 0 !important;
    }

    .stApp h2, .stApp h3, .stApp h4 {
        font-size: 22px !important;
        line-height: 1.25 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        margin: 0 0 12px 0 !important;
        padding: 0 !important;
    }

    .stApp button, .stApp input, .stApp textarea, .stApp label,
    .stApp [data-baseweb="select"], .stApp [data-baseweb="tab"] {
        font-family: var(--font) !important;
    }

    .stMarkdown, .stMarkdown * {
        font-family: var(--font) !important;
    }

    p, li {
        font-size: 16px;
        line-height: 1.55;
        color: var(--graphite);
    }

    .stMarkdown p, .stMarkdown li {
        max-width: 72ch;
    }

    small, .stCaption, [data-testid="stCaptionContainer"] p {
        font-size: 13px;
        line-height: 1.45;
        color: var(--pencil);
    }

    .page-sub {
        font-size: 16px;
        line-height: 1.55;
        color: var(--pencil);
        max-width: 60ch;
        margin: 0 0 28px 0;
    }

    .start-line {
        font-size: 22px;
        line-height: 1.35;
        font-weight: 500;
        letter-spacing: -0.01em;
        max-width: 62ch;
        margin: 8px 0 18px 0;
    }

    .lede {
        font-size: 26px;
        line-height: 1.38;
        font-weight: 500;
        letter-spacing: -0.015em;
        color: var(--graphite);
        max-width: 850px;
        margin: 4px 0 10px 0;
    }

    .lede mark.brand-mark {
        padding: 0 0.1em;
    }

    .lede-note {
        font-size: 13px;
        line-height: 1.5;
        color: var(--pencil);
        max-width: 72ch;
        margin: 0 0 30px 0;
    }

    mark.brand-mark {
        background: var(--highlighter);
        color: var(--graphite);
        padding: 0 0.12em;
        border-radius: 0;
    }

    .section-note {
        font-size: 13px;
        line-height: 1.45;
        color: var(--pencil);
        margin: -4px 0 16px 0;
        max-width: 72ch;
    }

    .rank-list {
        margin: 6px 0 4px 0;
        border-top: 1px solid var(--rule);
    }

    .rank-row {
        display: grid;
        grid-template-columns: minmax(150px, 1.5fr) minmax(170px, 3fr) 62px minmax(150px, 1.5fr);
        gap: 16px;
        align-items: center;
        padding: 13px 0;
        border-bottom: 1px solid var(--rule);
    }

    .rank-name {
        font-size: 16px;
        font-weight: 600;
        line-height: 1.3;
        color: var(--graphite);
    }

    .rank-track {
        height: 10px;
        background: var(--rule);
        overflow: hidden;
    }

    .rank-fill {
        height: 100%;
        background: var(--identity);
        transform-origin: left center;
        animation: bar-in 520ms cubic-bezier(0.2, 0.7, 0.2, 1) both;
    }

    .rank-pct {
        font-size: 16px;
        font-weight: 600;
        text-align: right;
        color: var(--graphite);
    }

    .rank-pct.failed, .failed {
        color: var(--brick);
    }

    .rank-meta {
        font-size: 13px;
        line-height: 1.45;
        color: var(--pencil);
    }

    @keyframes bar-in {
        from { transform: scaleX(0); }
        to { transform: scaleX(1); }
    }

    .model-column {
        border-top: 3px solid var(--identity);
        padding-top: 12px;
        min-height: 1px;
    }

    .model-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 10px;
    }

    .model-name {
        font-size: 16px;
        line-height: 1.3;
        font-weight: 600;
        color: var(--graphite);
    }

    .model-provider {
        font-size: 12px;
        color: var(--pencil);
        white-space: nowrap;
    }

    .answer-body {
        font-size: 16px;
        line-height: 1.55;
        color: var(--graphite);
        max-height: 300px;
        overflow-y: auto;
        padding-right: 8px;
        margin-bottom: 12px;
    }

    .answer-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 16px;
        border-top: 1px solid var(--rule);
        padding-top: 10px;
        font-size: 13px;
        line-height: 1.45;
        color: var(--pencil);
    }

    .answer-meta strong {
        font-weight: 600;
        color: var(--graphite);
    }

    .answer-competitors {
        font-size: 13px;
        line-height: 1.45;
        color: var(--pencil);
        margin-top: 8px;
    }

    .failure {
        font-size: 14px;
        line-height: 1.55;
        color: var(--brick);
        margin: 4px 0 6px 0;
    }

    .failure-fix {
        font-size: 13px;
        line-height: 1.45;
        color: var(--pencil);
    }

    .table-scroll {
        width: 100%;
        overflow-x: auto;
    }

    .editorial-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        line-height: 1.45;
        color: var(--graphite);
    }

    .editorial-table th {
        font-size: 13px;
        font-weight: 600;
        text-align: left;
        color: var(--graphite);
        border-bottom: 1px solid var(--graphite);
        padding: 10px 12px;
        vertical-align: bottom;
    }

    .editorial-table td {
        padding: 11px 12px;
        border-bottom: 1px solid var(--rule);
        vertical-align: top;
    }

    .editorial-table .num {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .editorial-table .yes {
        color: var(--graphite);
        font-weight: 500;
    }

    .editorial-table .no {
        color: var(--pencil);
    }

    .editorial-table .fail {
        color: var(--brick);
    }

    .coverage {
        table-layout: fixed;
    }

    .coverage th.prompt, .coverage td.prompt {
        width: 34%;
    }

    .coverage th.model {
        border-top: 3px solid var(--identity);
        padding-top: 11px;
    }

    .coverage th.model span {
        display: block;
        font-weight: 600;
    }

    .coverage th.model small {
        display: block;
        margin-top: 2px;
        font-weight: 400;
    }

    .coverage td.prompt {
        color: var(--graphite);
    }

    .sov {
        border-top: 1px solid var(--graphite);
        margin-top: 4px;
    }

    .sov-row {
        display: grid;
        grid-template-columns: minmax(140px, 1.2fr) minmax(160px, 3fr) 52px;
        gap: 16px;
        align-items: center;
        padding: 10px 12px;
        border-bottom: 1px solid var(--rule);
    }

    .sov-row.brand {
        background: var(--highlighter);
    }

    .sov-name {
        font-size: 14px;
        font-weight: 500;
        color: var(--graphite);
    }

    .sov-track {
        height: 8px;
        background: var(--rule);
        overflow: hidden;
    }

    .sov-fill {
        height: 100%;
        background: var(--pencil);
    }

    .sov-row.brand .sov-fill {
        background: var(--graphite);
    }

    .sov-count {
        text-align: right;
        font-size: 14px;
        font-weight: 500;
        color: var(--graphite);
    }

    .provider-line {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 12.5px;
        line-height: 1.45;
        color: var(--pencil);
        border-bottom: 1px solid var(--rule);
        padding: 6px 0;
    }

    .provider-line span:last-child {
        color: var(--pencil);
        text-align: right;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: var(--radius);
        border: 1px solid var(--rule);
        background: transparent;
        color: var(--graphite);
        font-family: var(--font);
        font-size: 14px;
        font-weight: 500;
        padding: 0.5rem 0.9rem;
        box-shadow: none;
        transition: background-color 120ms ease, border-color 120ms ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--rule);
        border-color: var(--pencil);
        color: var(--graphite);
    }

    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: var(--graphite);
        border-color: var(--graphite);
        color: var(--paper);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--pencil);
        border-color: var(--pencil);
        color: var(--paper);
    }

    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        border-radius: var(--radius);
        border-color: var(--rule);
        background: var(--paper);
        color: var(--graphite);
        font-family: var(--font);
    }

    div[data-baseweb="select"] > div {
        border-radius: var(--radius);
        border-color: var(--rule);
        background: var(--paper);
    }

    [data-baseweb="tag"] {
        background: var(--graphite) !important;
        color: var(--paper) !important;
        border-radius: var(--radius);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid var(--rule);
        margin-top: 26px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--pencil);
        font-size: 16px;
        font-weight: 500;
        padding: 8px 0 10px 0;
    }

    .stTabs [aria-selected="true"] {
        color: var(--graphite);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--graphite);
        height: 2px;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--rule);
        border-radius: 0;
        background: transparent;
    }

    div[data-testid="stAlert"] {
        border-radius: 0;
        border: 1px solid var(--rule);
        background: transparent;
    }

    div[data-testid="stAlert"] p {
        color: var(--graphite);
    }

    div[data-testid="stExpander"] summary {
        font-family: var(--font);
        font-size: 14px;
        color: var(--graphite);
    }

    [data-testid="stSidebar"] {
        background: var(--paper);
        border-right: 1px solid var(--rule);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 24px;
        padding-left: 24px;
        padding-right: 24px;
    }

    [data-testid="stSidebar"] h4 {
        font-size: 15px !important;
        font-weight: 600 !important;
        margin: 18px 0 8px 0 !important;
    }

    [data-testid="stSidebar"] label {
        font-size: 13px;
        color: var(--pencil);
    }

    hr {
        border-color: var(--rule);
        margin: 32px 0;
    }

    @media (prefers-reduced-motion: reduce) {
        .rank-fill { animation: none; }
        .stButton > button, .stDownloadButton > button { transition: none; }
    }

    @media (max-width: 1000px) {
        .block-container {padding-left: 20px; padding-right: 20px;}
        h1 {font-size: 34px;}
        .lede {font-size: 22px;}
        .rank-row {grid-template-columns: 1fr 58px; row-gap: 6px;}
        .rank-track {grid-column: 1 / -1; grid-row: 2;}
        .rank-meta {grid-column: 1 / -1;}
        .sov-row {grid-template-columns: 1fr 52px; row-gap: 6px;}
        .sov-row .sov-track {grid-column: 1 / -1; grid-row: 2;}
        .coverage {table-layout: auto;}
        .coverage th.prompt, .coverage td.prompt {width: auto;}
        div[data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
        div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Sidebar Configuration
# ==========================================
with st.sidebar:
    projects = get_projects()
    project_options = {"(None / Custom scan)": None}
    for p in projects:
        project_options[f"{p['brand_name']} ({p.get('industry') or 'General'})"] = p["id"]
    
    selected_proj_label = st.selectbox(
        "Project",
        options=list(project_options.keys()),
        index=0 if not st.session_state.selected_project_id else (
            [i for i, (lbl, pid) in enumerate(project_options.items()) if pid == st.session_state.selected_project_id] or [0]
        )[0],
        help="Select a saved project to load its brand settings and prompts."
    )
    current_proj_id = project_options[selected_proj_label]
    if current_proj_id != st.session_state.selected_project_id:
        st.session_state.selected_project_id = current_proj_id
        if current_proj_id:
            proj = get_project(current_proj_id)
            if proj:
                st.session_state.scan_brand = proj["brand_name"]
                active_p = get_prompts(current_proj_id, active_only=True)
                if active_p:
                    st.session_state.prompts_text = "\n".join([p["text"] for p in active_p])
        st.rerun()

    active_proj = get_project(current_proj_id) if current_proj_id else None

    brand_input = st.text_input(
        "Brand name",
        value=active_proj["brand_name"] if active_proj else (st.session_state.scan_brand if st.session_state.scan_brand else ""),
        placeholder="e.g. HubSpot, Nike, Monday.com",
        help="The primary brand or product name you want to track."
    )

    aliases_default = ", ".join(active_proj["aliases"]) if (active_proj and active_proj.get("aliases")) else ""
    aliases_input = st.text_input(
        "Brand aliases",
        value=aliases_default,
        placeholder="e.g. Monday, monday.com",
        help="Alternative spellings, separated by commas."
    )

    industry_default = active_proj.get("industry") or "" if active_proj else ""
    industry_input = st.text_input(
        "Industry",
        value=industry_default,
        placeholder="e.g. CRM software",
        help="Optional context for categorizing saved scans."
    )

    competitors_input = st.text_input(
        "Known competitors",
        placeholder="e.g. Salesforce, Zoho, Pipedrive",
        help="Optional competitors to monitor, separated by commas."
    )

    prompts_area = st.text_area(
        "Prompts (one per line)",
        value=st.session_state.prompts_text,
        height=140,
        placeholder="What is the best CRM for small teams?\nWhich sales platform has the top automation?",
        help="Duplicate and blank lines are removed automatically."
    )

    available_models = get_available_models()
    analyzer_model = get_analyzer_model()

    st.markdown("#### Models")
    if analyzer_model:
        st.caption(f"Analyzer: {analyzer_model}")
    else:
        st.error("ANALYZER_MODEL is not set in .env. Add it to grade sentiment and competitors.")

    if not available_models:
        st.error("No test models are available. Add provider keys and model IDs to .env.")
        selected_models = []
    else:
        default_selection = available_models[:min(4, len(available_models))]
        selected_models = st.multiselect(
            "Models to compare",
            options=available_models,
            default=default_selection,
            help="Select at least two models."
        )

    runs_per_prompt = st.slider(
        "Runs per prompt",
        min_value=1,
        max_value=5,
        value=1,
        help="Query each model several times to observe output variance."
    )

    use_cache = st.checkbox(
        "Use cached results",
        value=True,
        help="Skip network calls when an identical query has already been cached."
    )

    raw_prompts = [p.strip() for p in prompts_area.split("\n") if p.strip()]
    deduped_prompts = list(dict.fromkeys(raw_prompts))
    total_calls = len(deduped_prompts) * len(selected_models) * runs_per_prompt

    if total_calls > 0:
        st.caption(
            f"Estimated API calls: {total_calls} "
            f"({len(deduped_prompts)} prompts x {len(selected_models)} models x {runs_per_prompt} runs)"
        )
        if total_calls > 30:
            st.caption("Large scans can hit free-tier rate limits. Retries run automatically.")

    app_password = get_setting("APP_PASSWORD")
    max_session_calls = get_int_setting("MAX_CALLS_PER_SESSION", 30)

    # Password check if APP_PASSWORD is configured
    if app_password:
        if not st.session_state.authenticated:
            st.markdown("#### Authentication required")
            pwd_input = st.text_input("App password", type="password", help="Enter password to run real scans.")
            if st.button("Unlock scans"):
                if hmac.compare_digest(pwd_input, str(app_password)):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        else:
            st.caption("🔒 Authenticated for real scans")
    else:
        # Show session call cap status
        remaining_calls = max(0, max_session_calls - st.session_state.calls_used)
        st.caption(f"⚡ Session quota: **{remaining_calls}/{max_session_calls}** API calls remaining (cached queries free).")

    run_btn = st.button("Run scan", type="primary", use_container_width=True)

    # Validation checks
    if run_btn:
        brand_clean = brand_input.strip()

        is_analyzer_valid, analyzer_err = validate_analyzer_config(analyzer_model, selected_models)

        # Quota and auth check
        if app_password and not st.session_state.authenticated:
            st.session_state.validation_error = "Please enter the App Password in the sidebar to run scans."
        elif not app_password and total_calls > (max_session_calls - st.session_state.calls_used):
            st.session_state.validation_error = (
                f"Scan requires {total_calls} calls, but only {max_session_calls - st.session_state.calls_used} "
                f"remain in your session quota ({max_session_calls} max)."
            )
        elif not brand_clean:
            st.session_state.validation_error = "Brand name is required. Add it in the sidebar."
        elif not deduped_prompts:
            st.session_state.validation_error = "Add at least one prompt in the sidebar."
        elif len(selected_models) < 2:
            st.session_state.validation_error = "Select at least 2 models to compare."
        elif not is_analyzer_valid:
            st.session_state.validation_error = analyzer_err
        else:
            st.session_state.validation_error = None
            st.session_state.scan_brand = brand_clean
            st.session_state.is_sample = False

            aliases = [a.strip() for a in aliases_input.split(",") if a.strip()]
            known_comps = [c.strip() for c in competitors_input.split(",") if c.strip()]

            scan_id = create_scan(
                brand_name=brand_clean,
                industry=industry_input.strip() if industry_input.strip() else None,
                project_id=current_proj_id
            )
            st.session_state.scan_id = scan_id

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            query_units = []
            for prompt in deduped_prompts:
                for pm in selected_models:
                    for r_idx in range(1, runs_per_prompt + 1):
                        query_units.append((prompt, pm, r_idx))

            total_units = len(query_units)
            results = []
            completed = 0
            uncached_calls = 0

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        execute_query_unit,
                        p, pm, r, brand_clean, aliases, known_comps, analyzer_model, use_cache
                    ): (p, pm, r)
                    for (p, pm, r) in query_units
                }

                for future in as_completed(futures):
                    res = future.result()
                    results.append(res)
                    completed += 1
                    if not res.get("cached"):
                        uncached_calls += 1
                    progress_bar.progress(completed / total_units)
                    status_text.text(
                        f"{completed} of {total_units} complete: {res['provider']}:{res['model']}"
                    )

                    save_result(
                        scan_id=scan_id,
                        prompt=res["prompt"],
                        ai_response=res["ai_response"],
                        brand_mentioned=res["brand_mentioned"],
                        mention_count=res["mention_count"],
                        sentiment=res["sentiment"],
                        competitors=res["competitors"],
                        provider=res["provider"],
                        model=res["model"],
                        run_index=res["run_index"],
                        status=res["status"],
                        error=res["error"],
                        position=res["position"]
                    )

            # Record non-cached calls against session quota
            st.session_state.calls_used += uncached_calls

            overall_score = calculate_visibility_score(results)
            update_scan_score(scan_id, overall_score if overall_score is not None else 0.0)

            progress_bar.empty()
            status_text.empty()
            st.session_state.scan_results = results
            st.rerun()

    if st.session_state.validation_error:
        st.error(st.session_state.validation_error)

    st.divider()
    st.markdown("#### Provider status & self-check")

    providers = (("openrouter", "OpenRouter"), ("groq", "Groq"), ("gemini", "Gemini"))
    configured_any = False

    for provider, label in providers:
        key_val = get_setting(f"{provider.upper()}_API_KEY")
        key_present = bool(key_val and str(key_val).strip())
        model_count = sum(1 for m in available_models if m.startswith(f"{provider}:"))
        is_analyzer = bool(analyzer_model and analyzer_model.startswith(f"{provider}:"))

        if key_present:
            configured_any = True
            key_status = "Key configured"
            bits = [f"{model_count} model{'s' if model_count != 1 else ''}"]
            if is_analyzer:
                bits.append("analyzer")
            detail = f"{key_status} ({', '.join(bits)})"
        else:
            detail = "Key missing"

        st.markdown(
            f'<div class="provider-line"><span>{escape_text(label)}</span>'
            f'<span>{escape_text(detail)}</span></div>',
            unsafe_allow_html=True
        )

    # Analyzer model self-check
    st.markdown("#### Analyzer check")
    if not analyzer_model:
        st.markdown(
            '<div class="provider-line"><span>ANALYZER_MODEL</span>'
            '<span style="color:#B3402F">Missing</span></div>',
            unsafe_allow_html=True
        )
    else:
        prov, mod = (analyzer_model.split(":", 1) if ":" in analyzer_model else (analyzer_model, ""))
        supported = prov.lower() in ("openrouter", "groq", "gemini")
        in_tested = analyzer_model in selected_models
        if not supported:
            astatus = "Unsupported provider"
        elif in_tested:
            astatus = "Blocked: in tested models"
        else:
            astatus = "Configured"
            configured_any = True

        st.markdown(
            f'<div class="provider-line"><span>{escape_text(analyzer_model)}</span>'
            f'<span>{escape_text(astatus)}</span></div>',
            unsafe_allow_html=True
        )

    # If nothing is configured at all, show TOML guidance
    if not configured_any and not available_models:
        st.info(
            "**No API keys or models configured.**\n\n"
            "Add your credentials in Streamlit Cloud (App Settings -> Secrets) or in `.streamlit/secrets.toml` locally.\n\n"
            "**Format example (TOML strings, # comments):**\n"
            "```toml\n"
            "# Groq Cloud free tier\n"
            'GROQ_API_KEY = "gsk_..."\n'
            'GROQ_MODELS = "llama-3.1-8b-instant"\n\n'
            "# Dedicated analyzer\n"
            'ANALYZER_MODEL = "groq:llama-3.1-8b-instant"\n'
            "```"
        )


# ==========================================
# Main Header & Navigation Tabs
# ==========================================
st.title("AI Visibility Tracker")
st.markdown(
    '<p class="page-sub">See how language models answer the same buyer questions, '
    'and where your brand appears.</p>',
    unsafe_allow_html=True
)

tab_scan, tab_report, tab_library, tab_history, tab_compare = st.tabs([
    "Current scan",
    "Brand report",
    "Prompt library",
    "Scan history",
    "Trends"
])


# ==========================================
# TAB 1: CURRENT SCAN & DASHBOARD
# ==========================================
with tab_scan:
    if not st.session_state.scan_results:
        st.markdown(
            '<p class="start-line">Start with the brand you want to track and the '
            'questions your buyers ask.</p>',
            unsafe_allow_html=True
        )

        col_sample, _ = st.columns([1, 3])
        with col_sample:
            if st.button("Load sample dashboard", use_container_width=True):
                st.session_state.scan_results = SAMPLE_RESULTS
                st.session_state.scan_brand = "HubSpot"
                st.session_state.is_sample = True
                st.rerun()

        st.markdown("### Example prompts")
        st.markdown(
            '<p class="section-note">Select a prompt to add it to the scan.</p>',
            unsafe_allow_html=True
        )
        cols = st.columns(2)
        for i, prompt_text in enumerate(EXAMPLE_PROMPTS):
            with cols[i % 2]:
                if st.button(prompt_text, key=f"ex_btn_{i}", use_container_width=True):
                    current_prompts = [
                        p.strip() for p in st.session_state.prompts_text.split("\n") if p.strip()
                    ]
                    if prompt_text not in current_prompts:
                        current_prompts.append(prompt_text)
                    st.session_state.prompts_text = "\n".join(current_prompts)
                    st.rerun()

        with st.expander("Methodology"):
            st.markdown(
                "- Each prompt is sent to every selected model without a system wrapper or brand bias.\n"
                "- Mentions are matched on word boundaries, so substrings and aliases stay consistent.\n"
                "- A separate analyzer model grades sentiment and extracts competitors. It never grades its own answers.\n"
                "- Failed or retired calls are isolated, excluded from scoring, and reported with the successful calls.\n"
                "- Identical prompts and runs are cached locally when caching is enabled."
            )

    else:
        results = st.session_state.scan_results
        brand = st.session_state.scan_brand

        model_scores = calculate_model_scores(results)
        unique_prompts = list(dict.fromkeys([r["prompt"] for r in results]))
        colors = model_color_map(model_scores.keys())

        analyzer_failures = sum(
            1 for r in results
            if r.get("analyzer_status") == "failed" and r.get("status") == "ok"
        )
        total_failures = sum(1 for r in results if r.get("status") == "failed")

        st.markdown(
            f'<div class="lede">{build_lede(model_scores, brand)}</div>',
            unsafe_allow_html=True
        )

        notes = []
        if st.session_state.is_sample:
            notes.append("Sample data. No model calls were made.")
        if total_failures == 1:
            notes.append("1 call failed and is excluded from the ranking.")
        elif total_failures > 1:
            notes.append(
                f"{total_failures} calls failed and are excluded from the ranking."
            )
        if analyzer_failures:
            notes.append(
                f"Sentiment analysis failed for {analyzer_failures} answer"
                f"{'s' if analyzer_failures != 1 else ''}. "
                "Check the analyzer key and run the scan again."
            )
        if notes:
            st.markdown(
                f'<p class="lede-note">{escape_text(" ".join(notes))}</p>',
                unsafe_allow_html=True
            )

        st.markdown("### Model ranking")
        st.markdown(
            '<p class="section-note">Share of successful answers that name the brand.</p>',
            unsafe_allow_html=True
        )
        st.markdown(build_ranking_html(model_scores, colors), unsafe_allow_html=True)

        st.markdown("### Answers side by side")
        st.markdown(
            '<p class="section-note">The same prompt, read across models. '
            "Your brand is marked in each answer.</p>",
            unsafe_allow_html=True
        )

        chosen_prompt = st.selectbox("Prompt", unique_prompts)
        prompt_results = [r for r in results if r["prompt"] == chosen_prompt]

        unique_prompt_models = []
        seen_pm = set()
        for r in prompt_results:
            key = f"{r['provider']}:{r['model']}"
            if key not in seen_pm:
                seen_pm.add(key)
                unique_prompt_models.append(r)

        answer_cols = st.columns(
            len(unique_prompt_models) if unique_prompt_models else 1,
            gap="large"
        )
        for idx, item in enumerate(unique_prompt_models):
            pm_name = f"{item['provider']}:{item['model']}"
            color = colors.get(pm_name, MODEL_COLORS[0])
            with answer_cols[idx]:
                head = (
                    f'<div class="model-head">'
                    f'<div class="model-name">{escape_text(model_label(pm_name))}</div>'
                    f'<div class="model-provider">{escape_text(provider_label(pm_name))}</div>'
                    f'</div>'
                )
                if item["status"] == "failed":
                    body = (
                        f'<div class="failure">The call failed. '
                        f'{escape_text(item.get("error") or "No error detail was returned.")}</div>'
                        f'<div class="failure-fix">Remove this model or run the scan again.</div>'
                    )
                else:
                    highlighted = highlight_brand(item.get("ai_response") or "", brand)
                    highlighted = highlighted.replace("\n", "<br>")
                    position = f"#{item['position']}" if item.get("position") else "N/A"
                    mentioned = "Yes" if item["brand_mentioned"] else "No"
                    sentiment = (item.get("sentiment") or "neutral").capitalize()
                    body = (
                        f'<div class="answer-body">{highlighted}</div>'
                        f'<div class="answer-meta">'
                        f'<span>Mentioned <strong>{mentioned}</strong></span>'
                        f'<span>Position <strong>{position}</strong></span>'
                        f'<span>Sentiment <strong>{sentiment}</strong></span>'
                        f'</div>'
                    )
                    competitors = item.get("competitors") or []
                    if competitors:
                        body += (
                            f'<div class="answer-competitors">Competitors '
                            f'{escape_text(", ".join(competitors))}</div>'
                        )
                    if item.get("analyzer_status") == "failed":
                        body += (
                            '<div class="failure-fix">Sentiment analysis failed for '
                            'this answer.</div>'
                        )
                st.markdown(
                    f'<div class="model-column" style="--identity:{color}">'
                    f"{head}{body}</div>",
                    unsafe_allow_html=True
                )

        st.markdown("### Where models disagree")
        disagreements = find_model_disagreements(results)
        if not disagreements:
            st.markdown(
                '<p class="section-note">Every model agreed on this run.</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p class="section-note">Prompts where at least one model named the '
                "brand and another did not.</p>",
                unsafe_allow_html=True
            )
            for item in disagreements:
                with st.expander(item["prompt"]):
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        st.markdown("**Mentioned**")
                        for model_name in item["mentioned_by"]:
                            st.markdown(f"- `{model_name}`")
                    with col_no:
                        st.markdown("**Not mentioned**")
                        for model_name in item["not_mentioned_by"]:
                            st.markdown(f"- `{model_name}`")
                    if item["failed_models"]:
                        st.caption(
                            "Failed on this prompt: " + ", ".join(item["failed_models"])
                        )

        st.markdown("### Prompt coverage")
        st.markdown(
            '<p class="section-note">Mention status for every prompt and model.</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            build_coverage_html(
                unique_prompts,
                list(model_scores.keys()),
                results,
                colors
            ),
            unsafe_allow_html=True
        )

        st.markdown("### Share of voice")
        st.markdown(
            '<p class="section-note">How often your brand and competitors appear '
            "in the answers.</p>",
            unsafe_allow_html=True
        )

        all_comps = []
        brand_mentions_total = sum(
            1 for r in results if r["brand_mentioned"] and r["status"] == "ok"
        )
        for r in results:
            if r.get("competitors") and isinstance(r["competitors"], list):
                all_comps.extend(r["competitors"])

        comp_counts = Counter(all_comps)
        sov_items = [(f"{brand} (you)", brand_mentions_total, True)]
        for comp_name, count in comp_counts.most_common(10):
            sov_items.append((comp_name, count, False))

        max_mentions = max([count for _, count, _ in sov_items] + [1])
        sov_rows = []
        for name, count, is_brand in sov_items:
            width = (count / max_mentions) * 100
            row_class = "sov-row brand" if is_brand else "sov-row"
            sov_rows.append(
                f'<div class="{row_class}">'
                f'<div class="sov-name">{escape_text(name)}</div>'
                f'<div class="sov-track"><div class="sov-fill" '
                f'style="width:{width:.1f}%"></div></div>'
                f'<div class="sov-count">{count}</div>'
                f'</div>'
            )
        st.markdown('<div class="sov">' + "".join(sov_rows) + "</div>", unsafe_allow_html=True)

        st.markdown("### Export")
        export_rows = []
        for r in results:
            export_rows.append({
                "Brand": brand,
                "Provider": r["provider"],
                "Model": r["model"],
                "Run_Index": r.get("run_index", 1),
                "Prompt": r["prompt"],
                "Status": r["status"],
                "Error": r.get("error") or "",
                "Analyzer_Status": r.get("analyzer_status", "ok"),
                "Analyzer_Error": r.get("analyzer_error") or "",
                "Brand_Mentioned": r["brand_mentioned"],
                "Mention_Count": r["mention_count"],
                "Mention_Position": r.get("position") or "",
                "Sentiment": r["sentiment"],
                "Competitors": ", ".join(r["competitors"]) if isinstance(r.get("competitors"), list) else "",
                "AI_Response": r.get("ai_response") or ""
            })

        export_df = pd.DataFrame(export_rows)
        csv_data = export_df.to_csv(index=False)
        st.caption("One row per model call, including failed calls and raw answers.")
        st.download_button(
            label="Download results as CSV",
            data=csv_data,
            file_name=f"visibility_scan_{brand.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )



# ==========================================
# TAB 2: BRAND REPORT
# ==========================================
with tab_report:
    st.markdown("### Brand visibility report")
    st.markdown(
        '<p class="page-sub">Benchmark your brand visibility, competitor coverage, and share of voice '
        'across language models over time.</p>',
        unsafe_allow_html=True
    )

    # 1. Project / Brand selector
    available_projects = get_projects()
    all_known_scans = get_all_scans()
    all_scan_brands = list(dict.fromkeys([s["brand_name"] for s in all_known_scans if s.get("brand_name")]))
    
    col_rep_proj, col_rep_brand = st.columns([1, 1])
    with col_rep_proj:
        rep_proj_map = {"(All / Ad-hoc)": None}
        for p in available_projects:
            rep_proj_map[f"{p['brand_name']} ({p.get('industry') or 'General'})"] = p["id"]
        
        rep_p_labels = list(rep_proj_map.keys())
        default_p_idx = 0
        if st.session_state.selected_project_id:
            matching_p = [i for i, (lbl, pid) in enumerate(rep_proj_map.items()) if pid == st.session_state.selected_project_id]
            if matching_p:
                default_p_idx = matching_p[0]

        rep_proj_selected = st.selectbox("Filter by project", options=rep_p_labels, index=default_p_idx, key="rep_proj_select")
        active_rep_pid = rep_proj_map[rep_proj_selected]

    with col_rep_brand:
        if active_rep_pid:
            p_obj = get_project(active_rep_pid)
            brand_choices = [p_obj["brand_name"]] if p_obj else [st.session_state.scan_brand]
        else:
            brand_choices = all_scan_brands if all_scan_brands else [st.session_state.scan_brand]
        
        target_report_brand = st.selectbox(
            "Target brand",
            options=brand_choices,
            index=0 if brand_choices else 0,
            key="rep_brand_select"
        )

    # 2. Filter Bar
    st.markdown("#### Filter range")
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.2, 1.8, 0.8])
    with f_col1:
        date_preset = st.selectbox(
            "Date range",
            options=["All time", "Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", "Custom range"],
            index=0,
            key="rep_date_preset"
        )
    with f_col2:
        tag_choice = st.selectbox(
            "Prompt tag",
            options=["All tags", "branded", "non-branded", "top-of-funnel", "bottom-of-funnel"],
            index=0,
            key="rep_tag_choice"
        )
    
    # Calculate dates
    today_dt = date.today()
    start_dt_str = None
    end_dt_str = None
    if date_preset == "Last 7 days":
        start_dt_str = (today_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    elif date_preset == "Last 14 days":
        start_dt_str = (today_dt - timedelta(days=14)).strftime("%Y-%m-%d")
    elif date_preset == "Last 30 days":
        start_dt_str = (today_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    elif date_preset == "Last 90 days":
        start_dt_str = (today_dt - timedelta(days=90)).strftime("%Y-%m-%d")
    elif date_preset == "Custom range":
        custom_cols = st.columns(2)
        with custom_cols[0]:
            c_start = st.date_input("Start date", value=today_dt - timedelta(days=30), key="rep_c_start")
            start_dt_str = c_start.strftime("%Y-%m-%d") if c_start else None
        with custom_cols[1]:
            c_end = st.date_input("End date", value=today_dt, key="rep_c_end")
            end_dt_str = (c_end + timedelta(days=1)).strftime("%Y-%m-%d") if c_end else None

    # Fetch raw data from DB
    raw_report_results = get_scans_and_results_in_range(
        brand_name=target_report_brand if not active_rep_pid else None,
        project_id=active_rep_pid,
        start_date=start_dt_str,
        end_date=end_dt_str
    )

    # If no stored results in DB (e.g. running on sample dashboard state), use current session scan results
    using_session_fallback = False
    if not raw_report_results and st.session_state.scan_results:
        raw_report_results = [
            dict(r, scan_brand=target_report_brand, scan_created_at=datetime.now().strftime("%Y-%m-%d"))
            for r in st.session_state.scan_results
        ]
        using_session_fallback = True

    # Language model options in filtered rows
    present_models = list(dict.fromkeys([
        f"{r.get('provider')}:{r.get('model')}" for r in raw_report_results if r.get("provider") and r.get("model")
    ]))

    with f_col3:
        chosen_models = st.multiselect(
            "Language models",
            options=present_models,
            default=present_models,
            key="rep_models_multiselect",
            help="Filter answers by specific language models."
        )

    with f_col4:
        st.write("")
        st.write("")
        if st.button("Reset filters", key="rep_btn_reset"):
            st.session_state.rep_date_preset = "All time"
            st.session_state.rep_tag_choice = "All tags"
            st.rerun()

    # Filter rows based on tag and chosen models
    filtered_report_results = []
    for r in raw_report_results:
        pm_key = f"{r.get('provider')}:{r.get('model')}"
        if chosen_models and pm_key not in chosen_models:
            continue
        if tag_choice != "All tags":
            # Compare prompt_tag if joined from prompts table
            ptag = r.get("prompt_tag")
            if ptag:
                if ptag != tag_choice:
                    continue
            else:
                # Basic heuristic fallback for non-linked prompts
                p_text_lower = r.get("prompt", "").lower()
                b_name_lower = target_report_brand.lower()
                is_br = b_name_lower in p_text_lower
                if tag_choice == "branded" and not is_br:
                    continue
                elif tag_choice == "non-branded" and is_br:
                    continue
                elif tag_choice in ["top-of-funnel", "bottom-of-funnel"]:
                    # Cannot strictly determine without project tag
                    continue

        filtered_report_results.append(r)

    # 3. Header Summary Metric
    num_unique_prompts = len(set([r.get("prompt") for r in filtered_report_results if r.get("prompt")]))
    num_answers = len(filtered_report_results)
    
    st.markdown(
        f'<div class="section-note" style="margin-top: 8px; font-weight: 500;">'
        f'Based on <strong>{num_unique_prompts}</strong> prompts, <strong>{num_answers}</strong> language model answers.'
        f'</div>',
        unsafe_allow_html=True
    )

    if num_answers == 0:
        st.info("No answers match the selected filters. Adjust your date range, tag, or model selection.")
    else:
        # Compute metrics
        rep_metrics = compute_brand_report_metrics(filtered_report_results, target_brand=target_report_brand)

        # Exclusion alert
        if rep_metrics["failed_calls"] > 0:
            st.caption(
                f"ℹ️ {rep_metrics['failed_calls']} failed language model call"
                f"{'s' if rep_metrics['failed_calls'] != 1 else ''} excluded from report scoring."
            )

        # 4. Top KPI Cards: Share of Voice & Coverage & Sentiment
        st.markdown("#### Visibility overview")
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.markdown(
                f'<div class="brand-panel">'
                f'<div class="sov-name" style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pencil);">Share of voice</div>'
                f'<div style="font-size: 32px; font-weight: 700; color: var(--graphite); margin: 6px 0;">{rep_metrics["share_of_voice_pct"]:.1f}%</div>'
                f'<div style="font-size: 13px; color: var(--pencil);">'
                f'{rep_metrics["target_brand_mentions"]} mentions out of {rep_metrics["total_all_mentions"]} total brand mentions in range.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with kpi_col2:
            target_row = rep_metrics["comparison_table"][0]
            st.markdown(
                f'<div class="brand-panel">'
                f'<div class="sov-name" style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pencil);">Coverage rate</div>'
                f'<div style="font-size: 32px; font-weight: 700; color: var(--graphite); margin: 6px 0;">{target_row["coverage_pct"]:.1f}%</div>'
                f'<div style="font-size: 13px; color: var(--pencil);">'
                f'Named in {target_row["answers_with_mention"]} of {target_row["total_answers"]} language model answers.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with kpi_col3:
            s_counts = rep_metrics["sentiment_counts"]
            total_sent = sum(s_counts.values())
            pos_pct = (s_counts["positive"] / total_sent * 100.0) if total_sent > 0 else 0.0
            st.markdown(
                f'<div class="brand-panel">'
                f'<div class="sov-name" style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pencil);">Sentiment tone</div>'
                f'<div style="font-size: 32px; font-weight: 700; color: var(--graphite); margin: 6px 0;">{pos_pct:.0f}% Positive</div>'
                f'<div style="font-size: 13px; color: var(--pencil);">'
                f'{s_counts["positive"]} positive, {s_counts["neutral"]} neutral, {s_counts["negative"]} negative ({total_sent} evaluated).'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # 5. Comparison Panels: Chosen Brand vs. Top 5 Competitors
        st.markdown(f"#### {escape_text(target_report_brand)} vs. Top competitors")
        st.markdown(
            '<p class="section-note">Comparison across language model completions. Competitors are identified by the analyzer model.</p>',
            unsafe_allow_html=True
        )

        comp_table_data = rep_metrics["comparison_table"]
        
        # Render Comparison Cards
        for row in comp_table_data:
            b_name = row["brand"]
            is_tgt = row["is_target"]
            c_pct = row["coverage_pct"]
            m_pct = row["share_of_mentions_pct"]
            n_ans = row["answers_with_mention"]
            tot_ans = row["total_answers"]
            m_count = row["mentions"]
            tot_m = rep_metrics["total_all_mentions"]
            avg_p = f"#{row['avg_position']:.1f}" if row.get("avg_position") is not None else "—"
            tied_badge = ' <span style="font-size: 11px; background: var(--rule); padding: 2px 5px; border-radius: 3px;">(Tied)</span>' if row.get("is_tied_coverage") else ""
            target_badge = ' <span style="font-size: 11px; background: var(--highlighter); color: var(--graphite); font-weight: 600; padding: 2px 6px; border-radius: 3px;">YOU</span>' if is_tgt else ""

            st.markdown(
                f'<div class="brand-panel" style="margin-bottom: 12px; {"border-left: 4px solid var(--graphite);" if is_tgt else ""}">'
                f'<div style="display: flex; justify-content: space-between; align-items: baseline;">'
                f'<div style="font-size: 18px; font-weight: 600;">{escape_text(b_name)}{target_badge}</div>'
                f'<div style="font-size: 14px; color: var(--pencil);">Avg mention position: <strong>{avg_p}</strong></div>'
                f'</div>'
                f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px;">'
                f'<div>'
                f'<div style="font-size: 13px; color: var(--pencil);">Coverage: <strong>{c_pct:.1f}%</strong>{tied_badge} <small>({n_ans}/{tot_ans} answers)</small></div>'
                f'<div class="sov-track"><div class="sov-fill" style="width: {c_pct:.1f}%;"></div></div>'
                f'</div>'
                f'<div>'
                f'<div style="font-size: 13px; color: var(--pencil);">Share of mentions: <strong>{m_pct:.1f}%</strong> <small>({m_count}/{tot_m} mentions)</small></div>'
                f'<div class="sov-track"><div class="sov-fill" style="width: {m_pct:.1f}%; background: var(--model-sage);"></div></div>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # 6. Coverage Over Time Line Chart
        st.markdown("#### Coverage over time")
        time_data = rep_metrics["coverage_over_time"]
        
        if rep_metrics["scans_count"] < 2:
            st.info(
                f"At least 2 scans in the selected range are required to show coverage over time. "
                f"Currently, there {'is 1 scan' if rep_metrics['scans_count'] == 1 else 'are 0 scans'} in this range."
            )
        else:
            time_toggle = st.radio(
                "Display competitors:",
                options=["You + top 5", "You + all"],
                horizontal=True,
                key="rep_time_toggle"
            )

            df_time = pd.DataFrame(time_data)
            if time_toggle == "You + top 5":
                allowed_brands = set([target_report_brand] + rep_metrics["top_competitors"][:5])
                df_time = df_time[df_time["brand"].isin(allowed_brands)]

            # Altair line chart
            chart_brands = list(df_time["brand"].unique())
            time_colors = model_color_map(chart_brands)

            time_chart = (
                alt.Chart(df_time)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X(
                        "scan_date:N",
                        title="Scan Date",
                        axis=alt.Axis(
                            labelColor="#5F666D",
                            titleColor="#5F666D",
                            gridColor="#D5D9DC",
                            domainColor="#D5D9DC",
                            labelFont=FONT_FAMILY,
                            titleFont=FONT_FAMILY
                        )
                    ),
                    y=alt.Y(
                        "coverage_pct:Q",
                        title="Coverage (%)",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(
                            labelColor="#5F666D",
                            titleColor="#5F666D",
                            gridColor="#D5D9DC",
                            domainColor="#D5D9DC",
                            labelFont=FONT_FAMILY,
                            titleFont=FONT_FAMILY
                        )
                    ),
                    color=alt.Color(
                        "brand:N",
                        scale=alt.Scale(
                            domain=chart_brands,
                            range=[time_colors[b] for b in chart_brands]
                        ),
                        legend=alt.Legend(
                            orient="bottom",
                            title=None,
                            labelFont=FONT_FAMILY
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("brand:N", title="Brand"),
                        alt.Tooltip("scan_date:N", title="Date"),
                        alt.Tooltip("coverage_pct:Q", title="Coverage %", format=".1f"),
                        alt.Tooltip("answers_naming_brand:Q", title="Answers naming brand"),
                        alt.Tooltip("total_answers:Q", title="Total answers in scan")
                    ]
                )
                .properties(height=340)
                .configure_view(stroke=None)
            )

            st.altair_chart(time_chart, use_container_width=True)
            st.caption("Coverage % measures the percentage of successful language model answers that named each brand.")

        st.divider()

        # 7. "Where you're missing" (Deterministic Rule-Based Gap Analysis)
        st.markdown("#### Where you're missing")
        st.markdown(
            '<p class="section-note">Rule-based analysis identifying questions where competitors appear in '
            'most language model answers (&ge; 50%) but your brand appears in few or none (< 50%). '
            'Sorted by gap size.</p>',
            unsafe_allow_html=True
        )

        detected_gaps = find_visibility_gaps(filtered_report_results, target_brand=target_report_brand)

        if not detected_gaps:
            st.success(
                f"🎉 No major visibility gaps found in this range! Whenever competitors appeared, "
                f"**{target_report_brand}** also appeared consistently across language models."
            )
        else:
            for g_idx, gap in enumerate(detected_gaps):
                gap_prompt = gap["prompt"]
                gap_size = gap["gap_size"]
                c_cov = gap["competitor_coverage_pct"]
                b_cov = gap["brand_coverage_pct"]
                comps_list = gap["competitors"]
                models_omit = gap["models_omitting_brand"]
                models_comp = gap["models_naming_competitors"]

                rec_text = generate_gap_recommendation(gap, target_brand=target_report_brand)

                st.markdown(
                    f'<div class="brand-panel" style="margin-bottom: 14px; border-left: 4px solid var(--brick);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;">'
                    f'<div style="font-size: 16px; font-weight: 600; color: var(--graphite);">'
                    f'#{g_idx + 1} "{escape_text(gap_prompt)}"'
                    f'</div>'
                    f'<div style="font-size: 13px; font-weight: 600; color: var(--brick);">'
                    f'Gap size: +{gap_size:.1f}%'
                    f'</div>'
                    f'</div>'
                    f'<div style="display: flex; gap: 24px; font-size: 13px; color: var(--pencil); margin-top: 6px;">'
                    f'<span>Competitor coverage: <strong style="color: var(--graphite);">{c_cov:.1f}%</strong></span>'
                    f'<span>Your coverage: <strong style="color: var(--graphite);">{b_cov:.1f}%</strong></span>'
                    f'<span>Answers: {gap["answers_count"]}</span>'
                    f'</div>'
                    f'<div style="margin-top: 10px; font-size: 13px;">'
                    f'<strong style="color: var(--pencil);">Competitors appearing:</strong> '
                    f'{" ".join([f"`{escape_text(c["name"])} ({c["count"]})`" for c in comps_list])}'
                    f'</div>'
                    f'<div style="margin-top: 6px; font-size: 12px; color: var(--pencil);">'
                    f'Language models omitting your brand: '
                    f'{" ".join([f"<span class=\'model-tag\'>{escape_text(m)}</span>" for m in models_omit])}'
                    f'</div>'
                    f'<div style="margin-top: 12px; padding: 10px 14px; background: rgba(0,0,0,0.03); border: 1px dashed var(--rule); border-radius: 4px; font-size: 13px; color: var(--graphite);">'
                    f'<strong>📋 Suggested next step:</strong> {escape_text(rec_text)}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )



# ==========================================
# TAB: PROMPT LIBRARY & PROJECTS
# ==========================================
with tab_library:
    st.markdown("### Prompt library")
    st.markdown(
        '<p class="page-sub">Organize your buyer prompts by project and funnel stage, '
        'and run batch scans across language models.</p>',
        unsafe_allow_html=True
    )

    all_projs = get_projects()

    # Project selection / creation bar
    col_sel_p, col_new_p = st.columns([2, 1])
    with col_sel_p:
        proj_names = {f"{p['brand_name']} ({p.get('industry') or 'General'})": p["id"] for p in all_projs}
        if proj_names:
            current_idx = 0
            if st.session_state.selected_project_id:
                ids = list(proj_names.values())
                if st.session_state.selected_project_id in ids:
                    current_idx = ids.index(st.session_state.selected_project_id)
            lib_proj_label = st.selectbox("Select project", options=list(proj_names.keys()), index=current_idx)
            lib_proj_id = proj_names[lib_proj_label]
        else:
            lib_proj_id = None
            st.info("No projects yet. Create your first project on the right to organize prompts.")

    with col_new_p:
        with st.expander("➕ Create new project"):
            new_p_brand = st.text_input("Brand name", key="new_proj_brand", placeholder="e.g. Asana")
            new_p_aliases = st.text_input("Aliases (comma-separated)", key="new_proj_aliases", placeholder="e.g. Asana Inc")
            new_p_ind = st.text_input("Industry", key="new_proj_ind", placeholder="e.g. Project Management")
            if st.button("Create project", key="btn_create_proj"):
                if new_p_brand.strip():
                    created_pid = create_project(
                        brand_name=new_p_brand.strip(),
                        aliases=[a.strip() for a in new_p_aliases.split(",") if a.strip()],
                        industry=new_p_ind.strip() if new_p_ind.strip() else None
                    )
                    st.session_state.selected_project_id = created_pid
                    st.session_state.scan_brand = new_p_brand.strip()
                    st.success(f"Created project '{new_p_brand.strip()}'.")
                    st.rerun()
                else:
                    st.warning("Please provide a brand name.")

    if lib_proj_id:
        active_project = get_project(lib_proj_id)
        st.session_state.selected_project_id = lib_proj_id

        # Quick project actions & summary
        col_info, col_run = st.columns([3, 1])
        with col_info:
            p_prompts = get_prompts(lib_proj_id)
            active_count = sum(1 for p in p_prompts if p["active"])
            st.caption(
                f"**Project:** {active_project['brand_name']} | "
                f"**Industry:** {active_project.get('industry') or 'General'} | "
                f"**Total prompts:** {len(p_prompts)} ({active_count} active)"
            )
        with col_run:
            if st.button("🚀 Run all active prompts", key="btn_run_active_lib", use_container_width=True):
                active_list = [p["text"] for p in p_prompts if p["active"]]
                if not active_list:
                    st.warning("No active prompts in this project to run.")
                else:
                    st.session_state.prompts_text = "\n".join(active_list)
                    st.session_state.scan_brand = active_project["brand_name"]
                    st.info(f"Loaded {len(active_list)} active prompts into scan configuration. Click 'Run scan' in the sidebar.")

        st.divider()

        # Add single prompt or bulk paste
        col_single, col_bulk = st.columns([1, 1])
        with col_single:
            with st.expander("Add single prompt", expanded=False):
                single_text = st.text_input("Prompt text", key="lib_single_text", placeholder="What are the top CRM tools?")
                single_tag = st.selectbox(
                    "Tag",
                    options=["non-branded", "branded", "top-of-funnel", "bottom-of-funnel"],
                    key="lib_single_tag"
                )
                if st.button("Add prompt", key="btn_lib_add_single"):
                    if single_text.strip():
                        create_prompt(lib_proj_id, single_text.strip(), tag=single_tag, active=True)
                        st.success("Prompt added to library.")
                        st.rerun()
                    else:
                        st.warning("Prompt text cannot be empty.")

        with col_bulk:
            with st.expander("Bulk paste prompts", expanded=False):
                bulk_text = st.text_area(
                    "Paste prompts (one per line)",
                    key="lib_bulk_text",
                    height=100,
                    placeholder="1. Best project management software\n2. Jira vs Asana comparison\n* Top tools for sprint planning"
                )
                bulk_tag = st.selectbox(
                    "Assign tag to all",
                    options=["top-of-funnel", "bottom-of-funnel", "non-branded", "branded"],
                    key="lib_bulk_tag"
                )
                if st.button("Bulk add", key="btn_lib_add_bulk"):
                    if bulk_text.strip():
                        added = bulk_add_prompts(lib_proj_id, bulk_text, tag=bulk_tag)
                        st.success(f"Added {added} prompts with tag '{bulk_tag}'.")
                        st.rerun()
                    else:
                        st.warning("Paste at least one line.")

        # Prompt Ideas via Analyzer Language Model
        st.markdown("#### ✨ Prompt ideas generator")
        st.markdown(
            '<p class="section-note">Ask the analyzer language model to suggest 10 buyer-style prompts '
            'for your industry, categorized across funnel tags. Accept individual ideas directly into your library.</p>',
            unsafe_allow_html=True
        )

        col_gen_btn, col_gen_ind = st.columns([1, 2])
        with col_gen_ind:
            idea_industry = st.text_input(
                "Target industry / category",
                value=active_project.get("industry") or "",
                placeholder="e.g. CRM Software, Project Management",
                key="lib_idea_industry"
            )
        with col_gen_btn:
            st.write("")
            st.write("")
            generate_ideas_btn = st.button("💡 Generate 10 prompt ideas", key="btn_generate_ideas", use_container_width=True)

        if generate_ideas_btn:
            analyzer_m = get_analyzer_model()
            if not analyzer_m:
                st.error("ANALYZER_MODEL is not configured. Please set ANALYZER_MODEL in your secrets or .env.")
            elif not idea_industry.strip():
                st.warning("Please specify an industry/category to tailor the prompt ideas.")
            else:
                with st.spinner(f"Asking analyzer language model ({analyzer_m}) for 10 buyer prompts..."):
                    try:
                        ideas = generate_prompt_ideas(
                            industry=idea_industry.strip(),
                            brand_name=active_project["brand_name"],
                            analyzer_model=analyzer_m,
                            count=10
                        )
                        st.session_state.prompt_ideas = ideas
                        st.success(f"Generated {len(ideas)} buyer prompts!")
                    except Exception as e:
                        logger.exception("Error generating prompt ideas")
                        st.error(f"Failed to generate prompt ideas: {str(e)}")

        if st.session_state.prompt_ideas:
            with st.expander(f"Review and accept suggested ideas ({len(st.session_state.prompt_ideas)} available)", expanded=True):
                for idx, idea in enumerate(list(st.session_state.prompt_ideas)):
                    c_itxt, c_itag, c_iadd = st.columns([5, 2, 1.2])
                    with c_itxt:
                        st.write(f"**{idea['text']}**")
                    with c_itag:
                        st.caption(f"🏷️ `{idea['tag']}`")
                    with c_iadd:
                        if st.button("Accept", key=f"btn_accept_idea_{idx}"):
                            create_prompt(lib_proj_id, idea["text"], tag=idea["tag"], active=True)
                            st.session_state.prompt_ideas = [
                                item for item in st.session_state.prompt_ideas if item["text"] != idea["text"]
                            ]
                            st.success(f"Added to library!")
                            st.rerun()

                if st.button("Accept all remaining ideas", key="btn_accept_all_ideas"):
                    accepted_count = 0
                    for item in st.session_state.prompt_ideas:
                        create_prompt(lib_proj_id, item["text"], tag=item["tag"], active=True)
                        accepted_count += 1
                    st.session_state.prompt_ideas = []
                    st.success(f"Accepted {accepted_count} prompts into your project library!")
                    st.rerun()

        # Prompts Table and Controls
        st.markdown("#### Library items")
        filter_tag = st.selectbox(
            "Filter by tag",
            options=["All", "branded", "non-branded", "top-of-funnel", "bottom-of-funnel"],
            index=0,
            key="lib_filter_tag"
        )

        display_prompts = get_prompts(
            lib_proj_id,
            tag=None if filter_tag == "All" else filter_tag
        )

        if not display_prompts:
            st.info("No prompts match the current filter.")
        else:
            for p_item in display_prompts:
                p_id = p_item["id"]
                c_chk, c_text, c_tag, c_del = st.columns([0.6, 5, 2, 0.8])
                with c_chk:
                    is_active = st.checkbox(
                        "Active",
                        value=bool(p_item["active"]),
                        key=f"p_act_{p_id}",
                        label_visibility="collapsed"
                    )
                    if is_active != bool(p_item["active"]):
                        toggle_prompt_active(p_id, is_active)
                        st.rerun()
                with c_text:
                    st.write(p_item["text"])
                with c_tag:
                    st.caption(f"🏷️ `{p_item['tag']}`")
                with c_del:
                    if st.button("🗑️", key=f"p_del_{p_id}", help="Delete prompt"):
                        delete_prompt(p_id)
                        st.rerun()

        # Delete / manage project
        with st.expander("⚙️ Project settings"):
            col_edit_b, col_edit_i = st.columns(2)
            with col_edit_b:
                edit_brand_val = st.text_input("Brand", value=active_project["brand_name"], key="edit_proj_brand")
            with col_edit_i:
                edit_ind_val = st.text_input("Industry", value=active_project.get("industry") or "", key="edit_proj_ind")
            if st.button("Save project details", key="btn_save_proj_meta"):
                update_project(
                    project_id=lib_proj_id,
                    brand_name=edit_brand_val.strip(),
                    aliases=active_project.get("aliases"),
                    industry=edit_ind_val.strip() if edit_ind_val.strip() else None
                )
                st.success("Project updated.")
                st.rerun()

            st.markdown("---")
            if st.button("🚨 Delete project and prompts", key="btn_delete_proj"):
                delete_project(lib_proj_id)
                st.session_state.selected_project_id = None
                st.success("Project deleted.")
                st.rerun()


# ==========================================
# TAB 3: HISTORY
# ==========================================
with tab_history:
    st.markdown("### Scan history")
    st.caption("ℹ️ History is stored on the server and can reset when the app restarts. Download a CSV or backup to keep it.")

    col_bk1, col_bk2 = st.columns([1, 1])
    with col_bk1:
        db_bytes = get_database_bytes()
        st.download_button(
            label="Download database backup (.db)",
            data=db_bytes,
            file_name="visibility_tracker_backup.db",
            mime="application/x-sqlite3",
            help="Download the full SQLite database file to your local computer."
        )
    with col_bk2:
        uploaded_backup = st.file_uploader(
            "Restore from backup (.db)",
            type=["db", "sqlite", "sqlite3"],
            help="Upload a valid database backup to restore past scans.",
            label_visibility="collapsed"
        )
        if uploaded_backup is not None:
            if st.button("Apply database restore"):
                try:
                    restore_database_from_bytes(uploaded_backup.getvalue())
                    st.success("Database restored successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to restore database: {e}")

    scans = get_all_scans()

    if not scans:
        st.markdown(
            '<p class="section-note">No scans recorded. Run a scan to start a history.</p>',
            unsafe_allow_html=True
        )
    else:
        history_rows = []
        for scan in scans:
            score = (
                f"{scan['visibility_score']:.1f}%"
                if scan["visibility_score"] is not None else "N/A"
            )
            history_rows.append(
                "<tr>"
                f'<td class="num">{scan["id"]}</td>'
                f'<td>{escape_text(scan["brand_name"])}</td>'
                f'<td>{escape_text(scan["industry"] or "N/A")}</td>'
                f'<td class="num">{score}</td>'
                f'<td>{escape_text(scan["created_at"])}</td>'
                "</tr>"
            )

        st.markdown(
            '<div class="table-scroll"><table class="editorial-table">'
            "<thead><tr>"
            '<th class="num">Scan</th><th>Brand</th><th>Industry</th>'
            '<th class="num">Score</th><th>Date</th>'
            "</tr></thead><tbody>"
            + "".join(history_rows) +
            "</tbody></table></div>",
            unsafe_allow_html=True
        )

        selected_scan_id = st.selectbox(
            "Past scan",
            options=[s["id"] for s in scans],
            format_func=lambda x: f"Scan #{x} - {next((s['brand_name'] for s in scans if s['id'] == x), '')}"
        )

        if st.button("Load scan into current view"):
            past_results = get_scan_results(selected_scan_id)
            selected_scan_record = next((s for s in scans if s["id"] == selected_scan_id), None)
            if selected_scan_record and past_results:
                st.session_state.scan_results = past_results
                st.session_state.scan_brand = selected_scan_record["brand_name"]
                st.session_state.scan_id = selected_scan_id
                st.session_state.is_sample = False
                st.success(
                    f"Loaded {len(past_results)} records from scan #{selected_scan_id} "
                    "into the current view."
                )
                st.rerun()


# ==========================================
# TAB 3: COMPARE OVER TIME
# ==========================================
with tab_compare:
    st.markdown("### Visibility over time")
    st.caption("ℹ️ History is stored on the server and can reset when the app restarts. Download a CSV or backup to keep it.")
    scans = get_all_scans()

    if not scans:
        st.markdown(
            '<p class="section-note">Run scans over time to see visibility trends.</p>',
            unsafe_allow_html=True
        )
    else:
        unique_brands = sorted(list(set(s["brand_name"] for s in scans)))
        chosen_brand = st.selectbox("Brand", unique_brands)

        brand_scans = [s for s in scans if s["brand_name"].lower() == chosen_brand.lower()]
        trend_rows = []

        for scan in brand_scans:
            scan_res = get_scan_results(scan["id"])
            if not scan_res:
                continue
            model_metrics = calculate_model_scores(scan_res)
            for pm, stats in model_metrics.items():
                if stats["score"] is not None:
                    created_at = str(scan["created_at"])
                    trend_rows.append({
                        "Date": created_at,
                        "When": created_at[:16] if len(created_at) >= 16 else created_at,
                        "Scan": f"#{scan['id']}",
                        "Model": model_label(pm),
                        "Score": stats["score"],
                    })

        if trend_rows:
            trend_df = pd.DataFrame(trend_rows)
            trend_models = sorted(trend_df["Model"].unique().tolist())
            trend_colors = {
                name: MODEL_COLORS[i % len(MODEL_COLORS)]
                for i, name in enumerate(trend_models)
            }

            chart = (
                alt.Chart(trend_df)
                .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=52, filled=True))
                .encode(
                    x=alt.X(
                        "When:N",
                        title=None,
                        axis=alt.Axis(
                            labelAngle=-30,
                            labelLimit=160,
                            grid=False,
                            labelColor="#5F666D",
                            labelFontSize=12,
                            domainColor="#D5D9DC",
                            tickColor="#D5D9DC",
                            labelFont=FONT_FAMILY,
                        ),
                    ),
                    y=alt.Y(
                        "Score:Q",
                        title="Visibility score (%)",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(
                            labelColor="#5F666D",
                            titleColor="#5F666D",
                            labelFontSize=12,
                            titleFontSize=12,
                            gridColor="#D5D9DC",
                            domainColor="#D5D9DC",
                            tickColor="#D5D9DC",
                            labelFont=FONT_FAMILY,
                            titleFont=FONT_FAMILY,
                        ),
                    ),
                    color=alt.Color(
                        "Model:N",
                        scale=alt.Scale(
                            domain=trend_models,
                            range=[trend_colors[name] for name in trend_models],
                        ),
                        legend=alt.Legend(
                            orient="bottom",
                            title=None,
                            labelColor="#5F666D",
                            labelFontSize=12,
                            symbolType="stroke",
                            symbolStrokeWidth=3,
                            labelFont=FONT_FAMILY,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Model:N", title="Model"),
                        alt.Tooltip("Date:N", title="Date"),
                        alt.Tooltip("Score:Q", title="Score", format=".1f"),
                    ],
                )
                .properties(height=320)
                .configure_view(stroke=None)
                .configure_legend(padding=8)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("A score of 100% means every successful answer named the brand.")

            with st.expander("View data"):
                st.dataframe(
                    trend_df.rename(columns={"Score": "Visibility Score (%)"}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.markdown(
                f'<p class="section-note">Not enough data points yet for '
                f'{escape_text(chosen_brand)}.</p>',
                unsafe_allow_html=True
            )
