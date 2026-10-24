# 🔍 AI Visibility Tracker

A multi-model intelligence tool that queries leading Large Language Models (LLMs) in parallel to measure and benchmark brand visibility, mention rankings, sentiment, and competitor share of voice in AI-generated answers.

Designed to operate on a **$0 budget** using free models from **OpenRouter**, **Groq Cloud**, and **Google AI Studio (Gemini)**.

---

## 📸 Dashboard Preview

```
+----------------------------------------------------------------------------------------------------+
| 🔍 AI Visibility Tracker                                                                           |
| [ Current Scan & Comparison ]     [ Scan History ]     [ Trends ]                                  |
+----------------------------------------------------------------------------------------------------+
| 🏆 Model Comparison                                                                                |
| 💡 Summary: gemini:gemini-1.5-flash mentions you in 3/3 prompts (100%); groq:llama in 1/3 (33%)     |
| +-------------------------+-------+------------------+--------------+-------------+--------------+ |
| | Model                   | Rank  | Visibility Score | Avg Position | Pos Sent. % | Failed Calls | |
| | gemini:gemini-1.5-flash | #1    | 100.0%           | #1.7         | 67%         | 0            | |
| | groq:llama-3.1-8b-inst  | #2    | 33.3%            | #3.0         | 100%        | 0            | |
| | openrouter:mistral-7b   | N/A   | ⚠️ Failed        | N/A          | N/A         | 3            | |
| +-------------------------+-------+------------------+--------------+-------------+--------------+ |
| ⚡ Where Models Disagree:                                                                          |
| - Prompt: "Recommend top project management software for agile teams."                             |
|   Mentioned: gemini:gemini-1.5-flash | Not mentioned: groq:llama-3.1-8b-instant                     |
+----------------------------------------------------------------------------------------------------+
| 🔍 Side-by-Side Response Comparison                                                                |
| [ gemini:gemini-1.5-flash ]              [ groq:llama-3.1-8b-instant ]                             |
| "...we recommend HubSpot Projects..."    "...consider Jira, Asana, Monday..."                      |
| Mentioned: ✅ Yes | Rank: #1             | Mentioned: ❌ No | Rank: N/A                                |
+----------------------------------------------------------------------------------------------------+
```
*(Screenshot placeholder: `docs/dashboard_screenshot.png`)*

---

- **Brand Report & Benchmarking**:
  - Filter by date range (presets and custom pickers), prompt funnel tag, and language models.
  - Compares your brand vs. top 5 competitors across Share of Voice (SOV), Coverage %, and average rank position.
  - Interactive coverage-over-time trend lines with "You + top 5" vs "You + all" toggle.
  - Explicit reporting of failed calls excluded from scoring.
- **Rule-Based Visibility Gaps & Next Steps**:
  - Deterministically identifies "Where you're missing" on prompts where competitors appear frequently ($\ge 50\%$) while your brand is omitted.
  - Provides plain-language, data-driven next steps based strictly on observed results (no ranking guarantees).
- **Prompt Library & Ideas Generator**:
  - Organize prompts by project and funnel tags (`top-of-funnel`, `bottom-of-funnel`, `non-branded`, `branded`).
  - Generate 10 buyer-style prompt ideas via the analyzer language model with one-click individual acceptance.
- **Headless Scheduled Scans & Pluggable Database**:
  - `scripts/run_scan.py` runs all active prompts headlessly with configurable call caps.
  - GitHub Actions workflow (`.github/workflows/scheduled_scan.yml`) runs weekly scans automatically.
  - Pluggable `DATABASE_URL` supporting local SQLite or hosted persistent databases (Turso libSQL / Supabase).

---

## ☁️ Deploy on Streamlit Community Cloud

Deploying to Streamlit Community Cloud takes less than 2 minutes.

### 1. Repository Setup
1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your repository, branch, and set Main file path to:
   ```text
   app.py
   ```
4. **Choose Python Version (CRITICAL)**:
   - Click **Advanced settings...** at the bottom of the deployment setup modal.
   - Set **Python version** to **`3.12`** (recommended).
   - *Note*: Changing the Python version on an existing app requires a full redeploy (delete & redeploy or rebuild from the dashboard).

### 2. Configure Secrets in Streamlit Cloud
In Streamlit Cloud, open **App Settings -> Secrets** and paste your configuration in TOML format:

```toml
# -------------------------------------------------------------
# Streamlit Cloud Secrets (TOML format: quoted string values)
# -------------------------------------------------------------

# Google AI Studio (Gemini) Free Tier
GEMINI_API_KEY = "AIzaSy..."
GEMINI_MODELS = "gemini-1.5-flash,gemini-2.0-flash"

# Groq Cloud Free Tier
GROQ_API_KEY = "gsk_..."
GROQ_MODELS = "llama-3.1-8b-instant,mixtral-8x7b-32768"

# OpenRouter Free Tier (Optional)
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODELS = "meta-llama/llama-3.1-8b-instruct:free"

# Dedicated Analyzer Model (REQUIRED - format: provider:model-id)
ANALYZER_MODEL = "gemini:gemini-1.5-flash"

# Optional: Public Access Control & Quota Protection
# APP_PASSWORD = "secret_password"
MAX_CALLS_PER_SESSION = "30"
```

> **Reboot After Secrets Change**:
> Whenever you add or edit secrets in Streamlit Cloud, reboot the app (**Manage app -> ⋮ -> Reboot**) for changes to take effect immediately.

---

## 🔧 Troubleshooting Cloud Deployments

1. **Slow or Failed First Build (`Building dependencies...`)**:
   - Cause: Trying to compile dependencies from C source on newer or incompatible Python versions (e.g. Python 3.13 without binary wheels).
   - Fix: Ensure `requirements.txt` specifies modern ranges (`pandas>=2.2.3`, `openai>=1.55.3`, `streamlit>=1.40`) and set Python version to **3.12** in Cloud Advanced Settings.
2. **"Invalid format: please enter valid TOML"**:
   - Cause: Unquoted string values (e.g. `KEY = sk-1234` instead of `KEY = "sk-1234"`) or invalid comment syntax.
   - Fix: Wrap all strings in double quotes `""`, use `#` for every comment line, and verify against [.streamlit/secrets.toml.example](file:///.streamlit/secrets.toml.example).
3. **Analysis Showing "unknown" or "Sentiment analysis failed"**:
   - Cause: `ANALYZER_MODEL` is pointing to an invalid, retired, or rate-limited model ID, or its API key is missing.
   - Fix: Verify that the provider specified in `ANALYZER_MODEL` has its API key configured and that the model ID exists in your provider console (e.g., `gemini:gemini-1.5-flash` or `groq:llama-3.1-8b-instant`).

---

## 🛠️ Local Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Free API keys from [Google AI Studio](https://aistudio.google.com/app/apikey), [Groq Cloud](https://console.groq.com/keys), or [OpenRouter](https://openrouter.ai/keys)

### 2. Clone and Install Dependencies
```bash
git clone <repo-url>
cd ai_visibility_tracker

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install application dependencies
pip install -r requirements.txt

# For running tests locally:
pip install -r requirements-dev.txt
```

### 3. Configure Local Secrets or Environment
Create `.streamlit/secrets.toml` from the template:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Or copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests

The test suite runs with `pytest` using mocked API calls (no network or live keys required):

```bash
pytest tests/ -v
```

Test coverage includes:
- **`tests/test_stage1.py`**: Idempotent SQLite migrations, CRUD operations, response caching, and 429 retry backoff.
- **`tests/test_stage2.py`**: Word boundary matching ("apple" vs "pineapple"), brand aliases, rank position detection, robust JSON extraction, and score calculation.
- **`tests/test_stage3.py`**: Fault isolation (one failing provider does not crash scan) and cache hit verification.
- **`tests/test_validation_and_comparison.py`**: Multi-model disagreement detection, ranking tables, and model scoring.
- **`tests/test_config_and_cloud.py`**: `config.get_setting` precedence, safe RPM parsing, startup self-check logic, session quota caps, password verification, and SQLite backup validation.

---

## 🔬 Methodology & Limitations

1. **LLM Output Non-Determinism**: Generative models produce probabilistic responses. While temperature is kept low (`0.3` for generation, `0.1` for analysis), phrasing can vary between runs. Use the **Runs per prompt** slider (1–5) to measure variance and view visibility ranges.
2. **Small Sample Sizes**: Visibility percentages across 3–5 prompts provide indicative qualitative signals rather than statistical certainty. For robust benchmarks, include 10+ diverse user prompts representing various stages of the buyer journey.
3. **Fixed Analyzer Model**: A single configured model (`ANALYZER_MODEL`) evaluates all responses for sentiment and competitor extraction. This ensures consistent grading across models, eliminating evaluator bias.
4. **Rate Limit Throttling**: Free-tier rate limits (e.g. 15–30 requests per minute) mean large scans will automatically throttle and pace requests via exponential backoff.
