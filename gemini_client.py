import os
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
{prompt}
"""

    try:
        response = model.generate_content(visibility_prompt)
        return response.text
    except Exception as e:
        return f"Error querying AI: {str(e)}"