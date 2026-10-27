#!/usr/bin/env python3
"""Headless Scheduled Scan Runner for AI Visibility Tracker.

Executes all active prompts for a project across language models,
respecting per-provider rate limits and a configurable maximum call cap.
Stores all results in the database and updates visibility scores.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

from config import get_setting, get_int_setting
from database import (
    init_db,
    get_projects,
    get_project,
    get_prompts,
    create_scan,
    save_result,
    update_scan_score,
    get_cached_response,
    set_cached_response,
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
    generate_comparison_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("scheduled_scan")


def run_scheduled_scan(
    project_id: Optional[int] = None,
    max_calls: Optional[int] = None,
    models: Optional[List[str]] = None,
    use_cache: bool = True
) -> int:
    """Run all active prompts for a project headlessly."""
    init_db()

    # 1. Resolve project
    projects = get_projects()
    if not projects:
        logger.error("No projects found in database. Create a project in the UI first.")
        return 1

    target_proj = None
    if project_id is not None:
        target_proj = get_project(project_id)
        if not target_proj:
            logger.error(f"Project ID {project_id} not found.")
            return 1
    else:
        # Default to first available project
        target_proj = projects[0]

    brand_name = target_proj["brand_name"]
    aliases = target_proj.get("aliases") or []
    industry = target_proj.get("industry")
    proj_id = target_proj["id"]

    logger.info(f"Target Brand: {brand_name} (Project ID: {proj_id}, Industry: {industry or 'N/A'})")

    # 2. Resolve prompts
    active_prompts = get_prompts(proj_id, active_only=True)
    if not active_prompts:
        logger.warning(f"No active prompts found for project '{brand_name}'.")
        return 0

    logger.info(f"Loaded {len(active_prompts)} active prompt(s).")

    # 3. Resolve test models
    available_models = get_available_models()
    if not available_models:
        logger.error("No language models available. Please set API keys and model lists in environment variables.")
        return 1

    selected_models = models if models else available_models[:min(3, len(available_models))]
    if len(selected_models) < 2:
        logger.error(f"Need at least 2 language models to compare, but only got {selected_models}.")
        return 1

    logger.info(f"Selected language models: {', '.join(selected_models)}")

    # 4. Resolve analyzer model
    analyzer_model = get_analyzer_model()
    is_valid_analyzer, err = validate_analyzer_config(analyzer_model, selected_models)
    if not is_valid_analyzer:
        logger.error(f"Analyzer model configuration error: {err}")
        return 1

    logger.info(f"Analyzer language model: {analyzer_model}")

    # 5. Check call limits
    total_planned = len(active_prompts) * len(selected_models)
    call_cap = max_calls if max_calls is not None else get_int_setting("MAX_CALLS_PER_SESSION", 40)
    logger.info(f"Planned queries: {total_planned} calls (Cap: {call_cap})")

    # 6. Initialize scan in DB
    scan_id = create_scan(
        brand_name=brand_name,
        industry=industry,
        project_id=proj_id,
        expected_calls=total_planned,
    )
    logger.info(f"Created scan ID {scan_id} in database.")

    results = []
    calls_executed = 0

    for p_idx, prompt_obj in enumerate(active_prompts, 1):
        prompt_text = prompt_obj["text"]
        logger.info(f"[{p_idx}/{len(active_prompts)}] Prompt: \"{prompt_text}\"")

        for pm in selected_models:
            if calls_executed >= call_cap:
                logger.warning(f"Reached max call cap of {call_cap}. Halting further queries.")
                break

            provider, model = parse_provider_model(pm)
            ai_response = None
            status = "ok"
            error_msg = None
            cached_hit = False

            if use_cache:
                cached = get_cached_response(provider, model, prompt_text, run_index=1)
                if cached:
                    ai_response = cached
                    cached_hit = True

            if not cached_hit:
                try:
                    logger.info(f"  -> Querying {pm}...")
                    ai_response = query_ai(prompt_text, provider, model)
                    calls_executed += 1
                    if use_cache and ai_response:
                        set_cached_response(provider, model, prompt_text, ai_response, run_index=1)
                except (ConfigurationError, ModelUnavailableError, RateLimitExceededError, LLMClientError) as e:
                    status = "failed"
                    error_msg = str(e)
                    calls_executed += 1
                    logger.warning(f"  -> Call failed ({pm}): {e}")
                except Exception as e:
                    status = "failed"
                    error_msg = f"Unexpected failure: {str(e)}"
                    calls_executed += 1
                    logger.exception(f"  -> Unexpected error querying {pm}")

            # Analyze answer if successful
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
                    aliases=aliases,
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
                analyzer_status = "ok"
                analyzer_error = None

            # Save result row to DB
            save_result(
                scan_id=scan_id,
                prompt=prompt_text,
                ai_response=ai_response,
                brand_mentioned=brand_mentioned,
                mention_count=mention_count,
                sentiment=sentiment,
                competitors=competitors,
                provider=provider,
                model=model,
                run_index=1,
                status=status,
                error=error_msg,
                position=position,
                analyzer_status=analyzer_status,
                analyzer_error=analyzer_error,
            )

            results.append({
                "prompt": prompt_text,
                "provider": provider,
                "model": model,
                "status": status,
                "brand_mentioned": brand_mentioned,
                "sentiment": sentiment,
                "position": position,
                "competitors": competitors,
                "cached": cached_hit
            })

        if calls_executed >= call_cap:
            break

    # 7. Finalize scan score
    score = calculate_visibility_score(results)
    final_score = score if score is not None else 0.0
    update_scan_score(scan_id, final_score)

    model_scores = calculate_model_scores(results)
    summary_text = generate_comparison_summary(model_scores)

    logger.info("=" * 60)
    logger.info(f"SCAN COMPLETE: Scan ID {scan_id}")
    logger.info(f"Overall Visibility Score: {final_score:.1f}%")
    logger.info(f"Summary: {summary_text}")
    logger.info(f"Total calls executed: {calls_executed} (cached hits free)")
    logger.info("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Headless runner for AI Visibility Tracker scans across language models.")
    parser.add_argument("--project-id", type=int, help="Optional project ID to run active prompts for.")
    parser.add_argument("--max-calls", type=int, help="Maximum number of language model API calls allowed.")
    parser.add_argument("--models", nargs="+", help="Explicit list of provider:model to compare.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass response cache.")
    args = parser.parse_args()

    sys.exit(
        run_scheduled_scan(
            project_id=args.project_id,
            max_calls=args.max_calls,
            models=args.models,
            use_cache=not args.no_cache
        )
    )


if __name__ == "__main__":
    main()
