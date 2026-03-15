import streamlit as st
import pandas as pd
from collections import Counter
import json

from database import init_db, create_scan, save_result, update_scan_score
from gemini_client import query_ai, analyze_brand_mention, extract_competitors

# Initialize database
init_db()

st.set_page_config(
    page_title="AI Visibility Tracker",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Visibility Tracker")
st.markdown("Measure how visible your brand is in AI-generated responses")
def run_visibility_scan(brand_name: str, industry: str, prompts: list):
    scan_id = create_scan(brand_name, industry if industry else None)
    results = []
    all_competitors = []
    mentions_count = 0

    for prompt in prompts:
        ai_response = query_ai(prompt)

        analysis = analyze_brand_mention(ai_response, brand_name)

        competitors = extract_competitors(ai_response, brand_name)
        all_competitors.extend(competitors)

        if analysis["brand_mentioned"]:
            mentions_count += 1

        save_result(
            scan_id=scan_id,
            prompt=prompt,
            ai_response=ai_response,
            brand_mentioned=analysis["brand_mentioned"],
            mention_count=analysis["mention_count"],
            sentiment=analysis["sentiment"],
            competitors=competitors
        )

        results.append({
            "prompt": prompt,
            "ai_response": ai_response,
            "brand_mentioned": analysis["brand_mentioned"],
            "mention_count": analysis["mention_count"],
            "sentiment": analysis["sentiment"],
            "competitors": competitors
        })

    visibility_score = (mentions_count / len(prompts)) * 100 if prompts else 0
    update_scan_score(scan_id, visibility_score)

    return results, visibility_score, all_competitors