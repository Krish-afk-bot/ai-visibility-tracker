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

# Initialize session state
if "scan_complete" not in st.session_state:
    st.session_state.scan_complete = False
if "results" not in st.session_state:
    st.session_state.results = []
if "visibility_score" not in st.session_state:
    st.session_state.visibility_score = 0
if "all_competitors" not in st.session_state:
    st.session_state.all_competitors = []


def run_visibility_scan(brand_name: str, industry: str, prompts: list):
    """Execute visibility scan for all prompts."""
    scan_id = create_scan(brand_name, industry if industry else None)
    results = []
    all_competitors = []
    mentions_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, prompt in enumerate(prompts):
        status_text.text(f"Processing prompt {i + 1} of {len(prompts)}...")
        
        # Query AI
        ai_response = query_ai(prompt)
        
        # Analyze brand mention
        analysis = analyze_brand_mention(ai_response, brand_name)
        
        # Extract competitors
        competitors = extract_competitors(ai_response, brand_name)
        all_competitors.extend(competitors)
        
        if analysis["brand_mentioned"]:
            mentions_count += 1
        
        # Save to database
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
        
        progress_bar.progress((i + 1) / len(prompts))
    
    # Calculate visibility score
    visibility_score = (mentions_count / len(prompts)) * 100 if prompts else 0
    update_scan_score(scan_id, visibility_score)
    
    status_text.empty()
    progress_bar.empty()
    
    return results, visibility_score, all_competitors


# Sidebar - Input Form
with st.sidebar:
    st.header("📝 Scan Configuration")
    
    brand_name = st.text_input(
        "Brand / Product Name *",
        placeholder="e.g., Nike, Apple, Tesla"
    )
    
    industry = st.text_input(
        "Industry (optional)",
        placeholder="e.g., Sportswear, Technology, Automotive"
    )
    
    prompts_input = st.text_area(
        "Prompts (one per line) *",
        placeholder="What are the best running shoes?\nRecommend a smartphone for photography\nWhich electric car should I buy?",
        height=200
    )
    
    run_button = st.button("🚀 Run Visibility Scan", type="primary", use_container_width=True)
    
    if run_button:
        if not brand_name:
            st.error("Please enter a brand name")
        elif not prompts_input.strip():
            st.error("Please enter at least one prompt")
        else:
            prompts = [p.strip() for p in prompts_input.strip().split("\n") if p.strip()]
            if prompts:
                with st.spinner("Running visibility scan..."):
                    results, score, competitors = run_visibility_scan(brand_name, industry, prompts)
                    st.session_state.results = results
                    st.session_state.visibility_score = score
                    st.session_state.all_competitors = competitors
                    st.session_state.scan_complete = True
                    st.session_state.brand_name = brand_name
                st.success("Scan complete!")
            else:
                st.error("Please enter valid prompts")

# Main content area
if st.session_state.scan_complete and st.session_state.results:
    results = st.session_state.results
    brand = st.session_state.brand_name
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Visibility Score",
            value=f"{st.session_state.visibility_score:.1f}%"
        )
    
    with col2:
        mentions = sum(1 for r in results if r["brand_mentioned"])
        st.metric(
            label="Prompts with Mention",
            value=f"{mentions} / {len(results)}"
        )
    
    with col3:
        total_mentions = sum(r["mention_count"] for r in results)
        st.metric(
            label="Total Mentions",
            value=total_mentions
        )
    
    with col4:
        sentiments = [r["sentiment"] for r in results if r["brand_mentioned"]]
        positive = sentiments.count("positive")
        st.metric(
            label="Positive Sentiment",
            value=f"{positive} / {len(sentiments)}" if sentiments else "N/A"
        )
    
    st.divider()
    
    # Results table
    st.subheader("📊 Detailed Results")
    
    table_data = []
    for r in results:
        table_data.append({
            "Prompt": r["prompt"][:80] + "..." if len(r["prompt"]) > 80 else r["prompt"],
            "Brand Mentioned": "✅ Yes" if r["brand_mentioned"] else "❌ No",
            "Mentions": r["mention_count"],
            "Sentiment": r["sentiment"].capitalize() if r["brand_mentioned"] else "-"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Competitor analysis
    st.subheader("🏆 Competitor Brands Mentioned")
    
    if st.session_state.all_competitors:
        competitor_counts = Counter(st.session_state.all_competitors)
        competitor_df = pd.DataFrame(
            competitor_counts.most_common(10),
            columns=["Competitor", "Frequency"]
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(competitor_df, use_container_width=True, hide_index=True)
        with col2:
            st.bar_chart(competitor_df.set_index("Competitor"))
    else:
        st.info("No competitor brands detected in responses")
    
    # Raw responses (expandable)
    st.subheader("📄 Raw AI Responses")
    
    for i, r in enumerate(results):
        with st.expander(f"Prompt {i + 1}: {r['prompt'][:60]}..."):
            st.markdown("**Prompt:**")
            st.write(r["prompt"])
            st.markdown("**AI Response:**")
            st.write(r["ai_response"])
            st.markdown("**Analysis:**")
            st.write(f"- Brand Mentioned: {'Yes' if r['brand_mentioned'] else 'No'}")
            st.write(f"- Mention Count: {r['mention_count']}")
            st.write(f"- Sentiment: {r['sentiment']}")
            if r["competitors"]:
                st.write(f"- Competitors: {', '.join(r['competitors'])}")
    
    # Export section
    st.divider()
    st.subheader("📥 Export Results")
    
    export_data = []
    for r in results:
        export_data.append({
            "Prompt": r["prompt"],
            "AI Response": r["ai_response"],
            "Brand Mentioned": r["brand_mentioned"],
            "Mention Count": r["mention_count"],
            "Sentiment": r["sentiment"],
            "Competitors": ", ".join(r["competitors"])
        })
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"visibility_scan_{brand.replace(' ', '_')}.csv",
        mime="text/csv"
    )

else:
    # Welcome state
    st.info("👈 Configure your scan in the sidebar and click 'Run Visibility Scan' to get started")
    
    st.markdown("""
    ### How it works
    
    1. **Enter your brand name** - The product or company you want to track
    2. **Add prompts** - Questions a user might ask an AI assistant
    3. **Run the scan** - We'll query Gemini and analyze the responses
    4. **View results** - See your visibility score, sentiment analysis, and competitor mentions
    
    ### Example prompts to try
    
    - "What are the best laptops for programming?"
    - "Recommend a project management tool for startups"
    - "Which cloud provider should I use for my app?"
    - "What's the best CRM software for small businesses?"
    """)
