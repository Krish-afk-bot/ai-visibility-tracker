import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)


def get_model():
    configure_gemini()
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"temperature": 0.3}
    )


def query_ai(prompt: str) -> str:
    """Query Gemini with a visibility prompt."""
    model = get_model()
    
    visibility_prompt = f"""You are an unbiased AI assistant.
Answer the following question naturally and honestly.
Do not favor or promote any brand unless it is genuinely relevant.

Question:
{prompt}"""
    
    try:
        response = model.generate_content(visibility_prompt)
        return response.text
    except Exception as e:
        return f"Error querying AI: {str(e)}"


def analyze_brand_mention(ai_response: str, brand_name: str) -> dict:
    """Check if brand is mentioned and analyze sentiment."""
    model = get_model()
    
    brand_check_prompt = f"""Given the following AI response:
{ai_response}

Check for brand: {brand_name}

Return JSON only:
{{
    "brand_mentioned": true/false,
    "mention_count": number,
    "sentiment": "positive" | "neutral" | "negative"
}}"""
    
    try:
        response = model.generate_content(brand_check_prompt)
        text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "brand_mentioned": bool(result.get("brand_mentioned", False)),
                "mention_count": int(result.get("mention_count", 0)),
                "sentiment": result.get("sentiment", "neutral")
            }
    except Exception as e:
        pass
    
    # Fallback: simple text search
    brand_lower = brand_name.lower()
    response_lower = ai_response.lower()
    mentioned = brand_lower in response_lower
    count = response_lower.count(brand_lower)
    
    return {
        "brand_mentioned": mentioned,
        "mention_count": count,
        "sentiment": "neutral"
    }


def extract_competitors(ai_response: str, brand_name: str) -> list:
    """Extract competitor brand names from AI response."""
    model = get_model()
    
    extraction_prompt = f"""Extract only product or company names mentioned in the text below.
Exclude generic terms and exclude "{brand_name}".
Return ONLY a JSON array of strings.

Text:
{ai_response}"""
    
    try:
        response = model.generate_content(extraction_prompt)
        text = response.text.strip()
        
        # Extract JSON array from response
        json_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if json_match:
            competitors = json.loads(json_match.group())
            # Filter out the target brand
            return [c for c in competitors if c.lower() != brand_name.lower()]
    except Exception as e:
        pass
    
    return []
