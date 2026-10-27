import re
import json
import logging
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any

from llm_client import query_ai, parse_provider_model, LLMClientError

logger = logging.getLogger(__name__)


def detect_brand_mention_deterministic(
    text: str,
    brand_name: str,
    aliases: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Deterministically detect brand and alias mentions in text using word boundaries.
    
    Prevents false substrings like 'apple' matching 'pineapple'.
    Also extracts position/rank of the first mention if items are listed.
    """
    if not text or not brand_name or not brand_name.strip():
        return {
            "brand_mentioned": False,
            "mention_count": 0,
            "position": None,
            "matched_terms": []
        }

    terms = [brand_name.strip()]
    if aliases:
        for a in aliases:
            if a and a.strip() and a.strip().lower() not in [t.lower() for t in terms]:
                terms.append(a.strip())

    total_mentions = 0
    earliest_char_index = None
    matched_terms = []

    # Collect all boundary-respecting matches, then keep only non-overlapping
    # ones, preferring the longest term at a given position. This avoids
    # double-counting "Monday.com" as both the alias and the brand "Monday".
    all_matches = []
    for term in terms:
        escaped_term = re.escape(term)
        prefix_boundary = r"\b" if re.match(r"^\w", term) else r"(?:^|\s)"
        suffix_boundary = r"\b" if re.search(r"\w$", term) else r"(?:$|\s)"
        pattern = re.compile(rf"{prefix_boundary}{escaped_term}{suffix_boundary}", re.IGNORECASE)

        for match in pattern.finditer(text):
            all_matches.append((match.start(), match.end(), term))

    all_matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))

    last_end = -1
    for start, end, term in all_matches:
        if start < last_end:
            continue  # Overlaps a longer/earlier match already counted
        total_mentions += 1
        last_end = end
        if term not in matched_terms:
            matched_terms.append(term)
        if earliest_char_index is None or start < earliest_char_index:
            earliest_char_index = start

    brand_mentioned = total_mentions > 0
    position = None

    if brand_mentioned and earliest_char_index is not None:
        lines = text.splitlines()
        char_count = 0
        list_rank = 1
        found_in_list = False

        for line in lines:
            line_len = len(line) + 1
            num_match = re.match(r"^\s*(?:#?\s*(\d+)[\.\)]|\[(\d+)\])", line)
            if num_match:
                rank_str = num_match.group(1) or num_match.group(2)
                current_rank = int(rank_str)
                for term in terms:
                    esc = re.escape(term)
                    pref = r"\b" if re.match(r"^\w", term) else r"(?:^|\s)"
                    suff = r"\b" if re.search(r"\w$", term) else r"(?:$|\s)"
                    if re.search(rf"{pref}{esc}{suff}", line, re.IGNORECASE):
                        position = current_rank
                        found_in_list = True
                        break
                if found_in_list:
                    break
            elif re.match(r"^\s*[\*\-•]\s+", line):
                for term in terms:
                    esc = re.escape(term)
                    pref = r"\b" if re.match(r"^\w", term) else r"(?:^|\s)"
                    suff = r"\b" if re.search(r"\w$", term) else r"(?:$|\s)"
                    if re.search(rf"{pref}{esc}{suff}", line, re.IGNORECASE):
                        position = list_rank
                        found_in_list = True
                        break
                list_rank += 1
                if found_in_list:
                    break

            char_count += line_len

        if position is None:
            position = 1

    return {
        "brand_mentioned": brand_mentioned,
        "mention_count": total_mentions,
        "position": position,
        "matched_terms": matched_terms
    }


def parse_json_from_llm(raw_text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract and parse JSON object from LLM response without requiring JSON mode."""
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start_idx = cleaned.find('{')
    if start_idx != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(cleaned)):
            ch = cleaned[i]
            if ch == '"' and not escape:
                in_string = not in_string
            elif ch == '\\' and in_string:
                escape = not escape
                continue
            elif not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start_idx:i+1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, dict):
                                return data
                        except Exception:
                            pass
                        break
            escape = False

    return None


def analyze_sentiment_and_competitors(
    ai_response: str,
    brand_name: str,
    analyzer_model: str,
    brand_mentioned: bool = True,
    known_competitors: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Single LLM call to extract sentiment and competitor brands, saving free-tier quota.
    
    Returns:
        {
            "sentiment": "positive"|"neutral"|"negative"|"unknown",
            "competitors": list,
            "analyzer_status": "ok"|"failed",
            "analyzer_error": Optional[str]
        }
    """
    if not ai_response or not ai_response.strip():
        return {
            "sentiment": "unknown",
            "competitors": [],
            "analyzer_status": "failed",
            "analyzer_error": "Empty AI response to analyze"
        }

    known_list = [k.strip() for k in known_competitors if k and k.strip()] if known_competitors else []
    known_hint = f"Known competitors to consider: {', '.join(known_list)}." if known_list else ""

    prompt = f"""You are an unbiased data extractor. Analyze the AI response text below.

Target Brand: "{brand_name}"
{known_hint}

AI Response:
\"\"\"
{ai_response}
\"\"\"

Instructions:
1. Determine the sentiment toward "{brand_name}": "positive", "neutral", or "negative". If "{brand_name}" is not mentioned or tone is unclear, respond with "neutral".
2. Extract all competitor brand names or alternative product/company names mentioned in the text.
3. Exclude "{brand_name}" from the competitor list.
4. Return ONLY a valid JSON object in this exact format:
{{
    "sentiment": "positive" | "neutral" | "negative",
    "competitors": ["BrandA", "BrandB"]
}}"""

    provider, model = parse_provider_model(analyzer_model)

    try:
        raw_output = query_ai(prompt, provider, model, temperature=0.1)
        parsed = parse_json_from_llm(raw_output)
        if parsed:
            sentiment_raw = str(parsed.get("sentiment", "neutral")).strip().lower()
            if sentiment_raw not in ["positive", "neutral", "negative"]:
                sentiment = "neutral"
            else:
                sentiment = sentiment_raw

            raw_comps = parsed.get("competitors", [])
            if not isinstance(raw_comps, list):
                raw_comps = []

            cleaned_comps = []
            blocked = {brand_name.lower().strip()}
            if aliases:
                blocked.update(a.strip().lower() for a in aliases if a and a.strip())
            for c in raw_comps:
                if isinstance(c, str):
                    c_clean = c.strip()
                    if c_clean and c_clean.lower() not in blocked and c_clean.lower() not in [x.lower() for x in cleaned_comps]:
                        cleaned_comps.append(c_clean)

            # A valid JSON reply can still name no competitors. Fall back to
            # deterministic extraction so a successful answer never silently
            # reports "no competitors" when the text clearly names some.
            if not cleaned_comps:
                cleaned_comps = fallback_extract_competitors(
                    ai_response, brand_name, known_competitors, aliases
                )

            return {
                "sentiment": sentiment if brand_mentioned else "neutral",
                "competitors": cleaned_comps,
                "analyzer_status": "ok",
                "analyzer_error": None
            }
        else:
            err_msg = f"Failed to parse JSON from analyzer model {analyzer_model}"
            logger.warning(f"{err_msg}. Raw: {raw_output[:120]}")
            return {
                "sentiment": "unknown",
                "competitors": fallback_extract_competitors(ai_response, brand_name, known_competitors, aliases),
                "analyzer_status": "failed",
                "analyzer_error": err_msg
            }
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error querying analyzer model {analyzer_model}: {err_msg}")
        return {
            "sentiment": "unknown",
            "competitors": fallback_extract_competitors(ai_response, brand_name, known_competitors, aliases),
            "analyzer_status": "failed",
            "analyzer_error": err_msg
        }


# Generic table headers and filler that must never be reported as a competitor.
GENERIC_COMPETITOR_TERMS = {
    "brand", "brands", "company", "companies", "product", "products",
    "tool", "tools", "platform", "platforms", "software", "service",
    "services", "vendor", "vendors", "option", "options", "alternative",
    "alternatives", "competitor", "competitors", "feature", "features",
    "price", "pricing", "cost", "costs", "pros", "cons", "rating",
    "ratings", "score", "scores", "best", "top", "name", "summary",
    "details", "description", "overview", "category", "type", "website",
    "link", "notes", "note", "verdict", "recommendation", "reason",
    "reasoning", "strength", "strengths", "weakness", "weaknesses",
    "use case", "use cases", "free tier", "trial", "yes", "no", "n/a",
    "na", "none", "other", "others", "model", "models", "tool name",
    "product name", "company name", "platform name", "key features",
    "bottom line", "why it stands out", "best for", "starting price",
}

MAX_FALLBACK_COMPETITORS = 20


def _clean_extracted_name(raw_name: str) -> str:
    """Strip markdown emphasis, links, and table punctuation from a candidate name."""
    name = (raw_name or "").strip()
    if not name:
        return ""
    # Links: [Name](https://...)
    name = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", name)
    name = name.replace("**", "").replace("__", "").replace("`", "")
    name = name.strip(" \t|:;,.(-)")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _is_valid_competitor_name(
    name: str,
    brand_name: str,
    aliases: Optional[List[str]] = None
) -> bool:
    """Reject generic labels, the target brand, links, and non-name fragments."""
    if not name:
        return False
    if name.lower() in GENERIC_COMPETITOR_TERMS:
        return False
    if "http://" in name.lower() or "https://" in name.lower() or "www." in name.lower():
        return False
    if re.fullmatch(r"[\d\s.,%#/-]+", name):
        return False
    if len(name) < 2 or len(name) > 60:
        return False
    if len(name.split()) > 4:
        return False

    blocked = {brand_name.strip().lower()} if brand_name and brand_name.strip() else set()
    if aliases:
        blocked.update(a.strip().lower() for a in aliases if a and a.strip())
    return name.lower() not in blocked


def extract_markdown_table_first_column(text: str) -> List[str]:
    """Return the first column of every markdown table data row.

    Header rows (the row directly above a separator) and separator rows are
    skipped, so table labels like "Brand" or "Tool" never become competitors.
    """
    if not text:
        return []

    names: List[str] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped.strip("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0]
        if re.fullmatch(r":?-{2,}:?", first):
            continue

        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if re.fullmatch(r"\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*", next_line):
            continue  # This row is the table header.

        cleaned = _clean_extracted_name(first)
        if cleaned:
            names.append(cleaned)
    return names


def extract_bold_competitor_names(text: str) -> List[str]:
    """Return names emphasized in bold, e.g. "**Asana** offers ..."."""
    if not text:
        return []
    return [
        cleaned
        for cleaned in (_clean_extracted_name(m) for m in re.findall(r"\*\*(.+?)\*\*", text))
        if cleaned and cleaned.lower() not in GENERIC_COMPETITOR_TERMS
    ]


def fallback_extract_competitors(
    ai_response: str,
    brand_name: str,
    known_competitors: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None
) -> List[str]:
    """Deterministic competitor extraction used when the analyzer fails or is empty.

    1. Every configured "Known competitor" found with word boundaries.
    2. First columns of markdown tables (header/separator rows skipped).
    3. Bold names.
    Generic labels such as "Brand", "Tool", or "Best for" are dropped, and the
    target brand and its aliases are never reported as competitors.
    """
    if not ai_response or not ai_response.strip():
        return []

    detected: List[str] = []
    seen = set()

    def add(candidate: str, require_boundary: bool = False) -> None:
        cleaned = _clean_extracted_name(candidate)
        if not _is_valid_competitor_name(cleaned, brand_name, aliases):
            return
        key = cleaned.lower()
        if key in seen:
            return
        if require_boundary:
            esc = re.escape(cleaned)
            pref = r"\b" if re.match(r"^\w", cleaned) else r"(?:^|\s)"
            suff = r"\b" if re.search(r"\w$", cleaned) else r"(?:$|\s)"
            if not re.search(rf"{pref}{esc}{suff}", ai_response, re.IGNORECASE):
                return
        if len(detected) >= MAX_FALLBACK_COMPETITORS:
            return
        seen.add(key)
        detected.append(cleaned)

    brand_lower = brand_name.strip().lower() if brand_name else ""
    for comp in known_competitors or []:
        comp_clean = (comp or "").strip()
        if not comp_clean or comp_clean.lower() == brand_lower:
            continue
        add(comp_clean, require_boundary=True)

    for table_name in extract_markdown_table_first_column(ai_response):
        add(table_name)

    for bold_name in extract_bold_competitor_names(ai_response):
        add(bold_name)

    return detected


def analyze_brand_mention(
    ai_response: str,
    brand_name: str,
    aliases: Optional[List[str]] = None,
    analyzer_model: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze brand mention deterministically and fetch sentiment."""
    mention_info = detect_brand_mention_deterministic(ai_response, brand_name, aliases)
    
    sentiment = "neutral"
    analyzer_status = "ok"
    analyzer_error = None

    if mention_info["brand_mentioned"] and analyzer_model:
        analysis = analyze_sentiment_and_competitors(
            ai_response=ai_response,
            brand_name=brand_name,
            analyzer_model=analyzer_model,
            brand_mentioned=True
        )
        sentiment = analysis.get("sentiment", "unknown")
        analyzer_status = analysis.get("analyzer_status", "ok")
        analyzer_error = analysis.get("analyzer_error")

    return {
        "brand_mentioned": mention_info["brand_mentioned"],
        "mention_count": mention_info["mention_count"],
        "position": mention_info["position"],
        "sentiment": sentiment,
        "analyzer_status": analyzer_status,
        "analyzer_error": analyzer_error
    }


def extract_competitors(
    ai_response: str,
    brand_name: str,
    known_competitors: Optional[List[str]] = None,
    analyzer_model: Optional[str] = None
) -> List[str]:
    """Extract competitor brands using analyzer model or fallback."""
    if analyzer_model:
        res = analyze_sentiment_and_competitors(
            ai_response=ai_response,
            brand_name=brand_name,
            analyzer_model=analyzer_model,
            known_competitors=known_competitors
        )
        return res.get("competitors", [])
    return fallback_extract_competitors(ai_response, brand_name, known_competitors)


def calculate_visibility_score(
    results: List[Dict[str, Any]],
    model_filter: Optional[str] = None
) -> Optional[float]:
    """Calculate visibility score (% of successful prompts where brand was mentioned).
    
    Failed calls (status == 'failed') are excluded from the denominator.
    Returns None if there are zero successful prompts.
    """
    filtered = results
    if model_filter:
        filtered = [r for r in results if r.get("model") == model_filter or f"{r.get('provider')}:{r.get('model')}" == model_filter]

    successful = [r for r in filtered if r.get("status") == "ok"]
    if not successful:
        return None

    mentioned_count = sum(1 for r in successful if r.get("brand_mentioned"))
    return (mentioned_count / len(successful)) * 100.0


def calculate_model_scores(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-model metrics including visibility score range, avg position, and sentiments."""
    models: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        key = f"{r.get('provider')}:{r.get('model')}"
        if key not in models:
            models[key] = []
        models[key].append(r)

    summary = {}
    for key, model_results in models.items():
        successful = [r for r in model_results if r.get("status") == "ok"]
        failed = [r for r in model_results if r.get("status") == "failed"]

        if not successful:
            summary[key] = {
                "score": None,
                "score_min": None,
                "score_max": None,
                "successful_calls": 0,
                "failed_calls": len(failed),
                "total_calls": len(model_results),
                "mentions": 0,
                "avg_position": None,
                "positive_pct": 0.0,
                "sentiments": {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0}
            }
            continue

        total_ok = len(successful)
        mentioned_ok = sum(1 for r in successful if r.get("brand_mentioned"))
        score = (mentioned_ok / total_ok) * 100.0

        runs: Dict[int, List[Dict[str, Any]]] = {}
        for r in successful:
            idx = r.get("run_index", 1)
            runs.setdefault(idx, []).append(r)

        run_scores = []
        for r_list in runs.values():
            if r_list:
                m_count = sum(1 for r in r_list if r.get("brand_mentioned"))
                run_scores.append((m_count / len(r_list)) * 100.0)

        score_min = min(run_scores) if run_scores else score
        score_max = max(run_scores) if run_scores else score

        positions = [r["position"] for r in successful if r.get("brand_mentioned") and r.get("position") is not None]
        avg_pos = sum(positions) / len(positions) if positions else None

        sentiments = {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0}
        for r in successful:
            if r.get("brand_mentioned"):
                s = r.get("sentiment", "neutral").lower()
                if s in sentiments:
                    sentiments[s] += 1
                else:
                    sentiments["unknown"] += 1

        # Positive-sentiment % out of mentioned answers
        pos_pct = (sentiments["positive"] / mentioned_ok * 100.0) if mentioned_ok > 0 else 0.0

        summary[key] = {
            "score": score,
            "score_min": score_min,
            "score_max": score_max,
            "successful_calls": total_ok,
            "failed_calls": len(failed),
            "total_calls": len(model_results),
            "mentions": mentioned_ok,
            "avg_position": avg_pos,
            "positive_pct": pos_pct,
            "sentiments": sentiments
        }

    return summary


def find_model_disagreements(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify prompts where some models mentioned the brand and other models did not.
    
    Returns a list of dicts:
        [
            {
                "prompt": str,
                "mentioned_by": [model_keys],
                "not_mentioned_by": [model_keys],
                "failed_models": [model_keys]
            },
            ...
        ]
    """
    prompts = list(dict.fromkeys([r["prompt"] for r in results]))
    disagreements = []

    for prompt in prompts:
        prompt_results = [r for r in results if r["prompt"] == prompt]
        mentioned_by = []
        not_mentioned_by = []
        failed_models = []

        # Track per distinct model
        model_seen = set()
        for r in prompt_results:
            key = f"{r.get('provider')}:{r.get('model')}"
            if key in model_seen:
                continue
            model_seen.add(key)

            if r.get("status") == "failed":
                failed_models.append(key)
            elif r.get("brand_mentioned"):
                mentioned_by.append(key)
            else:
                not_mentioned_by.append(key)

        # Disagreement occurs if at least one model mentioned AND at least one model did NOT mention
        if mentioned_by and not_mentioned_by:
            disagreements.append({
                "prompt": prompt,
                "mentioned_by": mentioned_by,
                "not_mentioned_by": not_mentioned_by,
                "failed_models": failed_models
            })

    return disagreements


def generate_comparison_summary(model_scores: Dict[str, Dict[str, Any]]) -> str:
    """Generate a one-line auto summary comparing models, e.g.:
    'groq:llama-3 mentions you in 3/4 prompts; gemini:gemini-1.5 in 1/4 (openrouter:m1 failed).'
    """
    parts = []
    failed_models = []
    
    for pm, stats in model_scores.items():
        if stats["score"] is None:
            failed_models.append(pm)
        else:
            mentions = stats.get("mentions", 0)
            total = stats.get("successful_calls", 0)
            parts.append(f"**{pm}** mentions you in {mentions}/{total} prompts ({stats['score']:.0f}%)")

    summary_str = "; ".join(parts) if parts else "No successful model queries."
    if failed_models:
        summary_str += f" ({', '.join(failed_models)} failed calls excluded)"

    return summary_str


ANALYZER_INCOMPLETE_THRESHOLD = 0.30


def coerce_competitor_list(raw: Any) -> List[str]:
    """Return a clean list of competitor names from a stored value.

    Accepts lists or JSON-encoded strings (the database stores JSON text).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except Exception:
            return []
    if not isinstance(raw, (list, tuple)):
        return []

    names: List[str] = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if name:
                names.append(name)
    return names


def is_analysis_incomplete(
    failed_answers: int,
    total_answers: int,
    threshold: float = ANALYZER_INCOMPLETE_THRESHOLD
) -> bool:
    """True when the analyzer failed for more than `threshold` of the answers."""
    if total_answers <= 0:
        return False
    return (failed_answers / total_answers) > threshold


def incomplete_analysis_message(failed_answers: int, total_answers: int) -> str:
    """Exact banner text used when competitor/sentiment data cannot be trusted."""
    return (
        f"Competitor and sentiment data is incomplete "
        f"({failed_answers} of {total_answers} answers). Fix the analyzer and rerun."
    )


def compute_brand_report_metrics(
    results: List[Dict[str, Any]],
    target_brand: str,
    top_competitors_count: int = 5
) -> Dict[str, Any]:
    """Compute aggregated Brand Report metrics for a chosen brand vs. competitors.

    Units are answers, not word occurrences: a brand is counted once per
    successful language model answer that names it. Failed language model calls
    are excluded and reported separately.

    When the analyzer failed for more than 30% of successful answers, the
    competitor-derived metrics (share of voice, competitor shares) are marked
    unavailable instead of silently reporting a 100% share.

    Returns:
        {
            "total_answers": int,
            "successful_answers": int,
            "failed_calls": int,
            "unique_prompts_count": int,
            "analyzer_failures": int,
            "analyzed_answers": int,
            "analysis_incomplete": bool,
            "analysis_incomplete_message": Optional[str],
            "share_of_voice_pct": Optional[float],
            "sov_available": bool,
            "sov_n": int,                      # answers naming any tracked brand
            "target_brand_mentions": int,      # answers naming the target brand
            "answers_naming_any_tracked_brand": int,
            "total_all_mentions": int,         # legacy alias for the union above
            "sentiment_counts": {"positive": int, "neutral": int, "negative": int, "unknown": int},
            "sentiment_evaluated": int,
            "sentiment_available": bool,
            "comparison_table": List[Dict[str, Any]],
            "coverage_over_time": List[Dict[str, Any]],
            "top_competitors": List[str],
            "scans_count": int
        }
    """
    failed_calls = sum(1 for r in results if r.get("status") == "failed")
    successful = [r for r in results if r.get("status") == "ok"]
    total_successful = len(successful)

    unique_prompts = list(dict.fromkeys([r.get("prompt") for r in results if r.get("prompt")]))

    # ---- Analyzer coverage (answer unit) ----
    analyzer_failures = sum(
        1 for r in successful if (r.get("analyzer_status") or "ok") == "failed"
    )
    analyzed_answers = total_successful - analyzer_failures
    analysis_incomplete = is_analysis_incomplete(analyzer_failures, total_successful)
    analysis_message = (
        incomplete_analysis_message(analyzer_failures, total_successful)
        if analysis_incomplete else None
    )

    brand_norm = target_brand.strip().lower()

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0}
    sentiment_evaluated = 0

    target_answers_with_mention = 0
    target_positions: List[float] = []
    answers_naming_any = set()

    competitor_answers: Counter = Counter()

    for idx, r in enumerate(successful):
        competitors = coerce_competitor_list(r.get("competitors"))
        unique_in_answer = {}
        for c in competitors:
            key = c.strip().lower()
            if not key or key == brand_norm:
                continue
            unique_in_answer.setdefault(key, c.strip())

        if unique_in_answer:
            answers_naming_any.add(idx)
            for name in unique_in_answer.values():
                competitor_answers[name] += 1

        if r.get("brand_mentioned"):
            answers_naming_any.add(idx)
            target_answers_with_mention += 1

            if r.get("position") is not None:
                try:
                    target_positions.append(float(r["position"]))
                except (ValueError, TypeError):
                    pass

            if (r.get("analyzer_status") or "ok") != "failed":
                sent = str(r.get("sentiment") or "neutral").lower()
                if sent not in sentiment_counts:
                    sent = "unknown"
                sentiment_counts[sent] += 1
                sentiment_evaluated += 1

    sov_n = len(answers_naming_any)
    sov_available = (not analysis_incomplete) and sov_n > 0
    sov_pct = (
        round(target_answers_with_mention / sov_n * 100.0, 1)
        if sov_available else None
    )

    top_comps = [c for c, _ in competitor_answers.most_common(top_competitors_count)]

    # Build comparison rows for chosen brand + top competitors (same answer unit).
    comparison_table = []

    target_cov_pct = (
        target_answers_with_mention / total_successful * 100.0
        if total_successful > 0 else 0.0
    )
    target_share_answers = (
        round(target_answers_with_mention / sov_n * 100.0, 1)
        if sov_n > 0 and not analysis_incomplete else None
    )
    target_avg_pos = (sum(target_positions) / len(target_positions)) if target_positions else None

    comparison_table.append({
        "brand": target_brand,
        "is_target": True,
        "mentions": target_answers_with_mention,
        "share_of_mentions_pct": target_share_answers,
        "coverage_pct": target_cov_pct,
        "answers_with_mention": target_answers_with_mention,
        "total_answers": total_successful,
        "avg_position": target_avg_pos,
    })

    for comp in top_comps:
        c_answers = competitor_answers[comp]
        c_cov_pct = (c_answers / total_successful * 100.0) if total_successful > 0 else 0.0
        c_share_answers = (
            round(c_answers / sov_n * 100.0, 1)
            if sov_n > 0 and not analysis_incomplete else None
        )

        comparison_table.append({
            "brand": comp,
            "is_target": False,
            "mentions": c_answers,
            "share_of_mentions_pct": c_share_answers,
            "coverage_pct": c_cov_pct,
            "answers_with_mention": c_answers,
            "total_answers": total_successful,
            "avg_position": None,  # Not derivable from free-text competitor names
        })

    # Mark ties in coverage %
    cov_values = [round(row["coverage_pct"], 1) for row in comparison_table]
    cov_counts = Counter(cov_values)
    for row in comparison_table:
        val = round(row["coverage_pct"], 1)
        row["is_tied_coverage"] = cov_counts[val] > 1

    # Coverage over time (per scan_id or scan date)
    scans_grouped = {}
    for r in successful:
        s_id = r.get("scan_id") or "scan"
        s_date = str(r.get("scan_created_at") or r.get("created_at") or "")[:10]
        key = (s_id, s_date)
        if key not in scans_grouped:
            scans_grouped[key] = []
        scans_grouped[key].append(r)

    coverage_over_time = []
    all_tracked_brands = [target_brand] + top_comps

    for (s_id, s_date), s_rows in sorted(scans_grouped.items(), key=lambda x: str(x[0][1])):
        s_total = len(s_rows)
        if s_total == 0:
            continue

        for b_name in all_tracked_brands:
            b_is_target = (b_name.lower().strip() == brand_norm)
            if b_is_target:
                b_mentions_count = sum(1 for row in s_rows if row.get("brand_mentioned"))
            else:
                b_mentions_count = 0
                for row in s_rows:
                    comps = coerce_competitor_list(row.get("competitors"))
                    if any(c.lower().strip() == b_name.lower().strip() for c in comps):
                        b_mentions_count += 1

            cov = (b_mentions_count / s_total * 100.0)
            coverage_over_time.append({
                "scan_id": s_id,
                "scan_date": s_date or "Scan",
                "brand": b_name,
                "is_target": b_is_target,
                "coverage_pct": round(cov, 1),
                "answers_naming_brand": b_mentions_count,
                "total_answers": s_total
            })

    return {
        "total_answers": len(results),
        "successful_answers": total_successful,
        "failed_calls": failed_calls,
        "unique_prompts_count": len(unique_prompts),
        "analyzer_failures": analyzer_failures,
        "analyzed_answers": analyzed_answers,
        "analysis_incomplete": analysis_incomplete,
        "analysis_incomplete_message": analysis_message,
        "share_of_voice_pct": sov_pct,
        "sov_available": sov_available,
        "sov_n": sov_n,
        "target_brand_mentions": target_answers_with_mention,
        "answers_naming_any_tracked_brand": sov_n,
        "total_all_mentions": sov_n,
        "sentiment_counts": sentiment_counts,
        "sentiment_evaluated": sentiment_evaluated,
        "sentiment_available": (not analysis_incomplete) and sentiment_evaluated > 0,
        "comparison_table": comparison_table,
        "coverage_over_time": coverage_over_time,
        "top_competitors": top_comps,
        "scans_count": len(scans_grouped)
    }


def count_result_statuses(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count completed and failed calls in a scan result list."""
    completed = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") == "failed")
    return {"completed": completed, "failed": failed, "total": len(results)}


def compute_call_accounting(
    expected_calls: Optional[int],
    completed_calls: int,
    failed_calls: int
) -> Dict[str, Any]:
    """Reconcile expected scan calls (prompts x models x runs) with recorded rows.

    Returns expected/completed/failed/missing counts and a warning string when
    completed + failed does not equal expected.
    """
    expected = max(0, int(expected_calls or 0))
    completed = max(0, int(completed_calls or 0))
    failed = max(0, int(failed_calls or 0))
    accounted = completed + failed
    missing = max(0, expected - accounted)
    extra = max(0, accounted - expected)
    reconciles = expected == accounted

    warning = None
    if not reconciles:
        warning = (
            f"Call accounting does not reconcile: expected {expected} calls "
            f"(prompts x models x runs), but {accounted} were recorded "
            f"({completed} completed, {failed} failed, {missing} missing"
            f"{f', {extra} unexpected' if extra else ''})."
        )

    return {
        "expected": expected,
        "completed": completed,
        "failed": failed,
        "missing": missing,
        "extra": extra,
        "accounted": accounted,
        "reconciles": reconciles,
        "warning": warning,
    }


def find_visibility_gaps(
    results: List[Dict[str, Any]],
    target_brand: str
) -> List[Dict[str, Any]]:
    """Rule-based, deterministic gap identification (NO LLM).
    
    Identifies prompts where competitors appear in most answers (>= 50%)
    and the target brand appears in few or none (< 50%).
    
    Returns a list of gaps sorted by gap size descending.
    """
    brand_norm = target_brand.strip().lower()
    successful = [r for r in results if r.get("status") == "ok"]

    # Group by prompt
    prompt_groups = {}
    for r in successful:
        p = r.get("prompt")
        if not p:
            continue
        if p not in prompt_groups:
            prompt_groups[p] = []
        prompt_groups[p].append(r)

    gaps = []

    for prompt_text, p_rows in prompt_groups.items():
        total_answers = len(p_rows)
        if total_answers == 0:
            continue

        brand_answers = sum(1 for r in p_rows if r.get("brand_mentioned"))
        brand_cov_pct = (brand_answers / total_answers * 100.0)

        # Track competitors mentioned for this prompt
        comp_mentions = Counter()
        models_naming_competitors = set()
        models_naming_brand = set()
        models_omitting_brand = set()

        for r in p_rows:
            pm = f"{r.get('provider')}:{r.get('model')}"
            if r.get("brand_mentioned"):
                models_naming_brand.add(pm)
            else:
                models_omitting_brand.add(pm)

            comps = coerce_competitor_list(r.get("competitors"))

            for c in comps:
                if isinstance(c, str):
                    c_clean = c.strip()
                    if c_clean and c_clean.lower() != brand_norm:
                        comp_mentions[c_clean] += 1
                        models_naming_competitors.add(pm)

        answers_with_any_competitor = sum(
            1 for r in p_rows if any(
                c.strip().lower() != brand_norm
                for c in coerce_competitor_list(r.get("competitors"))
            )
        )
        comp_cov_pct = (answers_with_any_competitor / total_answers * 100.0)

        # Gap condition: Competitors appear in most answers (>= 50%) and brand in few or none (< 50%)
        if comp_cov_pct >= 50.0 and brand_cov_pct < 50.0:
            gap_size = comp_cov_pct - brand_cov_pct
            top_present_comps = [
                {"name": name, "count": count}
                for name, count in comp_mentions.most_common(5)
            ]
            gaps.append({
                "prompt": prompt_text,
                "gap_size": round(gap_size, 1),
                "competitor_coverage_pct": round(comp_cov_pct, 1),
                "brand_coverage_pct": round(brand_cov_pct, 1),
                "competitors": top_present_comps,
                "models_naming_competitors": sorted(list(models_naming_competitors)),
                "models_omitting_brand": sorted(list(models_omitting_brand)),
                "models_naming_brand": sorted(list(models_naming_brand)),
                "answers_count": total_answers,
                "brand_answers_count": brand_answers
            })

    # Sort by gap size descending
    gaps.sort(key=lambda g: g["gap_size"], reverse=True)
    return gaps


def generate_prompt_ideas(
    industry: str,
    brand_name: str,
    analyzer_model: str,
    count: int = 10
) -> List[Dict[str, str]]:
    """Ask the analyzer language model for buyer-style prompts grouped by funnel tag.
    
    Returns:
        [
            {"text": "...", "tag": "top-of-funnel" | "bottom-of-funnel" | "non-branded" | "branded"},
            ...
        ]
    """
    if not analyzer_model:
        raise ValueError("ANALYZER_MODEL is required to generate prompt ideas.")

    prompt_req = f"""You are an expert in buyer search behavior. Generate exactly {count} realistic, buyer-style questions and queries for the "{industry}" industry.
The user is tracking the brand "{brand_name}".

Categorize each prompt into one of these exact 4 funnel tags:
- "top-of-funnel": Discovery or problem-aware queries (e.g., "how to streamline team workflows")
- "bottom-of-funnel": Decision or high-intent buying queries (e.g., "best enterprise crm with sales automation")
- "non-branded": Category queries comparing general solutions without brand names (e.g., "top alternatives to spreadsheets for task tracking")
- "branded": Queries evaluating or comparing "{brand_name}" directly (e.g., "is {brand_name} worth it for small teams")

Return ONLY a valid JSON object in this exact format, with no markdown or extra commentary:
{{
  "prompts": [
    {{"text": "...", "tag": "top-of-funnel"}},
    {{"text": "...", "tag": "bottom-of-funnel"}},
    {{"text": "...", "tag": "non-branded"}},
    {{"text": "...", "tag": "branded"}}
  ]
}}"""

    provider, model = parse_provider_model(analyzer_model)
    raw_output = query_ai(prompt_req, provider, model, temperature=0.3)
    parsed = parse_json_from_llm(raw_output)

    if not parsed or "prompts" not in parsed or not isinstance(parsed["prompts"], list):
        raise ValueError(f"Failed to parse prompt ideas from analyzer model response: {raw_output[:120]}")

    valid_tags = {"top-of-funnel", "bottom-of-funnel", "non-branded", "branded"}
    cleaned_ideas = []
    for item in parsed["prompts"]:
        if isinstance(item, dict):
            p_text = str(item.get("text") or "").strip()
            p_tag = str(item.get("tag") or "non-branded").strip().lower()
            if p_tag not in valid_tags:
                p_tag = "non-branded"
            if p_text:
                cleaned_ideas.append({"text": p_text, "tag": p_tag})

    return cleaned_ideas


def generate_gap_recommendation(gap: Dict[str, Any], target_brand: str) -> str:
    """Generate a plain-language, purely data-driven next step for a visibility gap.
    
    Strictly factual based on observed completions. Makes NO claims or ranking guarantees.
    """
    total_answers = gap.get("answers_count", 0)
    brand_answers = gap.get("brand_answers_count", 0)
    comps = gap.get("competitors") or []
    models_omitting = gap.get("models_omitting_brand") or []
    omit_count = len(models_omitting)

    top_comp_name = comps[0]["name"] if comps else "competitors"
    top_comp_count = comps[0]["count"] if comps else 0

    if brand_answers == 0:
        if top_comp_count > 0:
            return (
                f"{top_comp_name} was named in {top_comp_count} of {total_answers} language model answers here, "
                f"while {target_brand} was omitted across all {total_answers} answers. "
                f"Check whether your public documentation or website has an informative resource, guide, or comparison page "
                f"specifically addressing this question."
            )
        else:
            return (
                f"{omit_count} of {total_answers} language model answers omitted {target_brand} for this query. "
                f"Consider verifying whether your product's positioning or features for this use case are clearly described publicly."
            )
    else:
        return (
            f"{target_brand} appeared in {brand_answers} of {total_answers} answers, but {top_comp_name} appeared "
            f"more frequently ({top_comp_count} times). "
            f"Review whether your public product materials clearly highlight your specific advantages for this topic."
        )



