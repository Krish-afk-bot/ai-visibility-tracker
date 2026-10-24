import os
import json
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import altair as alt
from dotenv import load_dotenv

# Load environment variables. override=True makes "streamlit run" pick up
# keys and model IDs edited in .env without restarting the server.
load_dotenv(override=True)

from database import (
    init_db,
    create_scan,
    save_result,
    update_scan_score,
    get_scan_results,
    get_cached_response,
    set_cached_response,
    get_all_scans,
)
from llm_client import (
    query_ai,
    parse_provider_model,
    get_available_models,
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
)

logger = logging.getLogger(__name__)

# Initialize database schema and migrations
init_db()

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
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "scan_brand" not in st.session_state:
    st.session_state.scan_brand = ""
if "scan_id" not in st.session_state:
    st.session_state.scan_id = None
if "prompts_text" not in st.session_state:
    st.session_state.prompts_text = ""
if "validation_error" not in st.session_state:
    st.session_state.validation_error = None
if "is_sample" not in st.session_state:
    st.session_state.is_sample = False


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
    st.markdown("#### Scan configuration")

    brand_input = st.text_input(
        "Brand name",
        value=st.session_state.scan_brand if st.session_state.scan_brand else "",
        placeholder="e.g. HubSpot, Nike, Monday.com",
        help="The primary brand or product name you want to track."
    )

    aliases_input = st.text_input(
        "Brand aliases",
        placeholder="e.g. Monday, monday.com",
        help="Alternative spellings, separated by commas."
    )

    industry_input = st.text_input(
        "Industry",
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

    run_btn = st.button("Run scan", type="primary", use_container_width=True)

    # Validation checks
    if run_btn:
        brand_clean = brand_input.strip()

        is_analyzer_valid, analyzer_err = validate_analyzer_config(analyzer_model, selected_models)

        if not brand_clean:
            st.session_state.validation_error = "Brand name is required. Add it in the sidebar."
        elif not deduped_prompts:
            st.session_state.validation_error = "Add at least one prompt in the sidebar."
        elif len(selected_models) < 2:
            st.session_state.validation_error = "Select at least two models."
        elif not is_analyzer_valid:
            st.session_state.validation_error = analyzer_err
        else:
            st.session_state.validation_error = None
            st.session_state.scan_brand = brand_clean
            st.session_state.is_sample = False

            aliases = [a.strip() for a in aliases_input.split(",") if a.strip()]
            known_comps = [c.strip() for c in competitors_input.split(",") if c.strip()]

            scan_id = create_scan(brand_clean, industry_input.strip() if industry_input.strip() else None)
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

            overall_score = calculate_visibility_score(results)
            update_scan_score(scan_id, overall_score if overall_score is not None else 0.0)

            progress_bar.empty()
            status_text.empty()
            st.session_state.scan_results = results
            st.rerun()

    if st.session_state.validation_error:
        st.error(st.session_state.validation_error)

    st.divider()
    st.markdown("#### Provider access")
    for provider, label in (("openrouter", "OpenRouter"), ("groq", "Groq"), ("gemini", "Gemini")):
        key_present = bool(os.getenv(f"{provider.upper()}_API_KEY", "").strip())
        model_count = sum(1 for m in available_models if m.startswith(f"{provider}:"))
        is_analyzer = bool(analyzer_model and analyzer_model.startswith(f"{provider}:"))
        if not key_present:
            detail = "No key"
        else:
            bits = ["Key detected"]
            if model_count:
                bits.append(f"{model_count} model{'s' if model_count != 1 else ''}")
            if is_analyzer:
                bits.append("analyzer")
            detail = ", ".join(bits)
        st.markdown(
            f'<div class="provider-line"><span>{escape_text(label)}</span>'
            f'<span>{escape_text(detail)}</span></div>',
            unsafe_allow_html=True
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

tab_scan, tab_history, tab_compare = st.tabs(["Current scan", "Scan history", "Trends"])


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
# TAB 2: HISTORY
# ==========================================
with tab_history:
    st.markdown("### Scan history")
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
