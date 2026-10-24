# 🔍 AI Visibility Tracker

A multi-model intelligence tool that queries leading Large Language Models (LLMs) in parallel to measure and benchmark brand visibility, mention rankings, sentiment, and competitor share of voice in AI-generated answers.

Designed to operate on a **$0 budget** using only free models from **OpenRouter** and the free tier of **Groq Cloud**.

---

## 📸 Dashboard Preview

```
+----------------------------------------------------------------------------------------------------+
| 🔍 AI Visibility Tracker                                                                           |
| [ Current Scan ]     [ Scan History ]     [ Compare Over Time ]                                    |
+----------------------------------------------------------------------------------------------------+
| Visibility Score per Model                                                                         |
|  groq:llama-3.1-8b-instant       openrouter:meta-llama/3-8b:free       openrouter:mistral-7b:free   |
|         85.0%                                 60.0%                              75.0%             |
|  [|||||||||||||||||||||]             [|||||||||||||]                   [|||||||||||||||||]         |
+----------------------------------------------------------------------------------------------------+
| Per-Prompt Visibility Grid                                                                         |
| Prompt                                   | groq:llama-3.1-8b-instant | openrouter:llama-3-8b:free   |
| What are the best CRM tools for startups?| ✅ Yes (#1)               | ✅ Yes (#3)                  |
| Recommend top agile PM software          | ❌ No                     | ✅ Yes (#1)                  |
+----------------------------------------------------------------------------------------------------+
| Competitor Share of Voice                | Sentiment Breakdown & Mention Ranking                   |
| 👉 Your Brand (6 mentions)                | Positive: 4 | Neutral: 2 | Negative: 0                  |
| Salesforce (5 mentions)                  | Average Position: #1.5                                  |
+----------------------------------------------------------------------------------------------------+
```
*(Screenshot placeholder: `docs/dashboard_screenshot.png`)*

---

## 🚀 Key Features

- **Multi-Model Parallel Scanning**: Compare outputs across multiple free models from OpenRouter and Groq in a single execution using concurrent worker threads.
- **Strict Word-Boundary Matching**: Deterministic Python regex prevents false substring positives (e.g., `apple` will never match `pineapple`). Also supports brand aliases (e.g., `Monday` / `Monday.com`).
- **Normalized Single-Model Grading**: A dedicated, independent analyzer LLM evaluates sentiment and competitor mentions across all answers. A model never grades its own response.
- **Combined Analysis for Quota Conservation**: Sentiment and competitor extraction are merged into a single LLM prompt per response, reducing free-tier token usage.
- **Free-Tier Reliability**:
  - Exponential backoff with jitter on HTTP 429, 5xx, and network timeouts.
  - Honors server `Retry-After` headers.
  - Per-provider rate limiting (token window) prevents rate limit collisions.
  - Automatic SQLite caching to avoid redundant queries (`Use cached results` toggle).
  - Fault-tolerant: failing or retired models do not interrupt or crash the scan.
- **Historical Analysis & Benchmarking**:
  - **Scan History**: Review past scan runs and reload granular prompt-level records.
  - **Compare Over Time**: Track longitudinal visibility trajectories across models for any brand.
- **Export**: Download full run results (including status, errors, sentiment, rank position, and competitors) as a formatted CSV.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Free API keys from:
  - [Groq Cloud Console](https://console.groq.com/keys)
  - [OpenRouter](https://openrouter.ai/keys)

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

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and configure your API keys and models:
```dotenv
# OpenRouter API Key and Models
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
# Copy current live free models ending in :free from https://openrouter.ai/models?max_price=0
OPENROUTER_MODELS=meta-llama/llama-3.1-8b-instruct:free,mistralai/mistral-7b-instruct:free

# Groq API Key and Models
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
# Live Groq model IDs from console.groq.com/docs/models
GROQ_MODELS=llama-3.1-8b-instant,mixtral-8x7b-32768

# Dedicated Analyzer Model
ANALYZER_MODEL=groq:llama-3.1-8b-instant
```

> **Important Note on Free-Tier Models & Limits**:
> Free models on OpenRouter frequently rotate and change availability. If a model returns 404 or retired errors, filter OpenRouter's model directory by **"Free"** (`max_price=0`) and update your `.env` with active model IDs.

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests

The test suite uses `pytest` with mocked network calls (no real API requests or API keys required):

```bash
pytest tests/ -v
```

Test coverage includes:
- **`tests/test_stage1.py`**: Idempotent SQLite migrations on legacy schemas, CRUD operations, response caching, client pooling, 429 exponential backoff retry with `Retry-After`, and 404 retired model handling.
- **`tests/test_stage2.py`**: Word-boundary regex matching, brand aliases with symbols, rank position extraction, robust markdown-fenced JSON extraction, single combined sentiment/competitors analysis, and score calculation excluding failures.
- **`tests/test_stage3.py`**: Parallel execution error isolation (one failing provider not crashing the scan) and SQLite cache hit verification.

---

## 🔬 Methodology & Limitations

1. **LLM Output Non-Determinism**: Generative models produce probabilistic responses. While temperature is kept low (`0.3` for generation, `0.1` for analysis), phrasing can vary between runs. Use the **Runs per prompt** slider (1–5) to measure variance and view visibility ranges.
2. **Small Sample Sizes**: Visibility percentages across 3–5 prompts provide indicative qualitative signals rather than statistical certainty. For robust benchmarks, include 10+ diverse user prompts representing various stages of the buyer journey.
3. **Fixed Analyzer Model**: A single configured model (`ANALYZER_MODEL`) evaluates all responses for sentiment and competitor extraction. This ensures consistent grading across models, eliminating evaluator bias.
4. **Rate Limit Throttling**: Free-tier rate limits (e.g. 20–30 requests per minute) mean large scans (over 30 total queries) will automatically throttle and pace requests via exponential backoff.
