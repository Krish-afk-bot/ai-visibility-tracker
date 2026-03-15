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