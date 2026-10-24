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
    known_competitors: Optional[List[str]] = None
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
            brand_lower = brand_name.lower().strip()
            for c in raw_comps:
                if isinstance(c, str):
                    c_clean = c.strip()
                    if c_clean and c_clean.lower() != brand_lower and c_clean.lower() not in [x.lower() for x in cleaned_comps]:
                        cleaned_comps.append(c_clean)

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
                "competitors": fallback_extract_competitors(ai_response, brand_name, known_competitors),
                "analyzer_status": "failed",
                "analyzer_error": err_msg
            }
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error querying analyzer model {analyzer_model}: {err_msg}")
        return {
            "sentiment": "unknown",
            "competitors": fallback_extract_competitors(ai_response, brand_name, known_competitors),
            "analyzer_status": "failed",
            "analyzer_error": err_msg
        }


def fallback_extract_competitors(
    ai_response: str,
    brand_name: str,
    known_competitors: Optional[List[str]] = None
) -> List[str]:
    """Deterministic fallback to find known competitors in text if LLM call fails."""
    detected = []
    brand_lower = brand_name.lower().strip()
    if known_competitors:
        for comp in known_competitors:
            comp_clean = comp.strip()
            if not comp_clean or comp_clean.lower() == brand_lower:
                continue
            esc = re.escape(comp_clean)
            pref = r"\b" if re.match(r"^\w", comp_clean) else r"(?:^|\s)"
            suff = r"\b" if re.search(r"\w$", comp_clean) else r"(?:$|\s)"
            if re.search(rf"{pref}{esc}{suff}", ai_response, re.IGNORECASE):
                if comp_clean.lower() not in [d.lower() for d in detected]:
                    detected.append(comp_clean)
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


def compute_brand_report_metrics(
    results: List[Dict[str, Any]],
    target_brand: str,
    top_competitors_count: int = 5
) -> Dict[str, Any]:
    """Compute aggregated Brand Report metrics for a chosen brand vs. competitors.
    
    All metrics strictly exclude failed language model calls and report failed count.
    
    Returns:
        {
            "total_answers": int,
            "successful_answers": int,
            "failed_calls": int,
            "unique_prompts_count": int,
            "share_of_voice_pct": float,
            "target_brand_mentions": int,
            "total_all_mentions": int,
            "sentiment_counts": {"positive": int, "neutral": int, "negative": int, "unknown": int},
            "comparison_table": List[Dict[str, Any]], # [target_brand, comp1, comp2, ...]
            "coverage_over_time": List[Dict[str, Any]], # [{scan_id, scan_date, brand, coverage_pct, answers_n}]
            "top_competitors": List[str]
        }
    """
    failed_calls = sum(1 for r in results if r.get("status") == "failed")
    successful = [r for r in results if r.get("status") == "ok"]
    total_successful = len(successful)

    unique_prompts = list(dict.fromkeys([r.get("prompt") for r in results if r.get("prompt")]))

    # Target brand sentiment distribution over successful calls
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0}
    for r in successful:
        if r.get("brand_mentioned"):
            sent = str(r.get("sentiment") or "neutral").lower()
            if sent not in sentiment_counts:
                sent = "unknown"
            sentiment_counts[sent] += 1

    # Brand stats aggregation
    brand_norm = target_brand.strip().lower()
    
    # 1. Target brand mentions & coverage
    target_mentions = 0
    target_answers_with_mention = 0
    target_positions = []
    
    for r in successful:
        if r.get("brand_mentioned"):
            target_mentions += max(1, r.get("mention_count") or 1)
            target_answers_with_mention += 1
            if r.get("position") is not None:
                try:
                    target_positions.append(float(r["position"]))
                except (ValueError, TypeError):
                    pass

    # 2. Competitors aggregation from stored competitor lists
    competitor_mentions_counter = Counter()
    competitor_answers_counter = Counter()
    competitor_positions = {}
    
    for r in successful:
        comps = r.get("competitors") or []
        if isinstance(comps, str):
            try:
                comps = json.loads(comps)
            except Exception:
                comps = []
        
        seen_in_row = set()
        for c in comps:
            if not isinstance(c, str):
                continue
            c_clean = c.strip()
            if not c_clean or c_clean.lower() == brand_norm:
                continue
            
            competitor_mentions_counter[c_clean] += 1
            
            if c_clean.lower() not in seen_in_row:
                seen_in_row.add(c_clean.lower())
                competitor_answers_counter[c_clean] += 1

    # Top competitors by frequency
    top_comps = [c for c, _ in competitor_mentions_counter.most_common(top_competitors_count)]

    # Total brand mentions across ALL brands (target brand + all competitors)
    total_all_mentions = target_mentions + sum(competitor_mentions_counter.values())

    # Share of voice for target brand
    sov_pct = (target_mentions / total_all_mentions * 100.0) if total_all_mentions > 0 else 0.0

    # Build comparison rows for chosen brand + top competitors
    comparison_table = []

    # Target brand row
    target_cov_pct = (target_answers_with_mention / total_successful * 100.0) if total_successful > 0 else 0.0
    target_share_of_mentions = (target_mentions / total_all_mentions * 100.0) if total_all_mentions > 0 else 0.0
    target_avg_pos = (sum(target_positions) / len(target_positions)) if target_positions else None

    comparison_table.append({
        "brand": target_brand,
        "is_target": True,
        "mentions": target_mentions,
        "share_of_mentions_pct": target_share_of_mentions,
        "coverage_pct": target_cov_pct,
        "answers_with_mention": target_answers_with_mention,
        "total_answers": total_successful,
        "avg_position": target_avg_pos,
    })

    # Competitor rows
    for comp in top_comps:
        c_mentions = competitor_mentions_counter[comp]
        c_answers = competitor_answers_counter[comp]
        c_cov_pct = (c_answers / total_successful * 100.0) if total_successful > 0 else 0.0
        c_share_mentions = (c_mentions / total_all_mentions * 100.0) if total_all_mentions > 0 else 0.0
        
        comparison_table.append({
            "brand": comp,
            "is_target": False,
            "mentions": c_mentions,
            "share_of_mentions_pct": c_share_mentions,
            "coverage_pct": c_cov_pct,
            "answers_with_mention": c_answers,
            "total_answers": total_successful,
            "avg_position": None,  # Extracted from free text competitors
        })

    # Mark ties in coverage % and average position
    cov_values = [round(row["coverage_pct"], 1) for row in comparison_table]
    cov_counts = Counter(cov_values)
    for row in comparison_table:
        val = round(row["coverage_pct"], 1)
        row["is_tied_coverage"] = cov_counts[val] > 1

    # Coverage over time (per scan_id or scan_created_at)
    scans_grouped = {}
    for r in successful:
        s_id = r.get("scan_id") or "scan"
        s_date = str(r.get("scan_created_at") or r.get("created_at") or "")[:10]
        key = (s_id, s_date)
        if key not in scans_grouped:
            scans_grouped[key] = []
        scans_grouped[key].append(r)

    coverage_over_time = []
    # All brands to track over time
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
                    comps = row.get("competitors") or []
                    if any(isinstance(c, str) and c.lower().strip() == b_name.lower().strip() for c in comps):
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
        "share_of_voice_pct": round(sov_pct, 1),
        "target_brand_mentions": target_mentions,
        "total_all_mentions": total_all_mentions,
        "sentiment_counts": sentiment_counts,
        "comparison_table": comparison_table,
        "coverage_over_time": coverage_over_time,
        "top_competitors": top_comps,
        "scans_count": len(scans_grouped)
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

            comps = r.get("competitors") or []
            if isinstance(comps, str):
                try:
                    comps = json.loads(comps)
                except Exception:
                    comps = []

            for c in comps:
                if isinstance(c, str):
                    c_clean = c.strip()
                    if c_clean and c_clean.lower() != brand_norm:
                        comp_mentions[c_clean] += 1
                        models_naming_competitors.add(pm)

        answers_with_any_competitor = sum(
            1 for r in p_rows if any(
                isinstance(c, str) and c.strip().lower() != brand_norm
                for c in (r.get("competitors") or [])
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



