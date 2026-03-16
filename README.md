<div align="center">

#  AI Visibility Tracker

**Measure how visible your brand is in AI-generated responses**

![Status](https://img.shields.io/badge/status-active-success.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

[🚀 Live Demo](https://ai-visibility-tracker-fjt2fxn73cargkaatc6rqr.streamlit.app/) • [📖 Documentation](#how-it-works) • [🛠️ Setup](#local-setup)

</div>

---

##  Mission

As AI assistants become the primary way people discover products and services, brand visibility in AI responses is the new SEO. AI Visibility Tracker helps businesses understand and measure their presence in AI-generated recommendations.

##  The Challenge

Traditional marketing metrics don't capture how often AI assistants recommend your brand. When users ask ChatGPT, Gemini, or Claude for product recommendations, is your brand mentioned? How does your visibility compare to competitors?

Without measurement, you can't optimize. Businesses need data on:
- How frequently AI mentions their brand
- Sentiment of those mentions (positive, neutral, negative)
- Which competitors appear alongside them
- How visibility changes across different prompts

##  Our Solution

AI Visibility Tracker automates brand visibility measurement by:
1. Querying AI models with custom prompts
2. Analyzing responses for brand mentions and sentiment
3. Identifying competitor brands in the same responses
4. Calculating a visibility score across all prompts
5. Providing exportable data for tracking over time

##  Key Features

| Feature | Description |
|---------|-------------|
|  **Visibility Score** | Percentage of prompts where your brand appears |
|  **Sentiment Analysis** | Tracks positive, neutral, or negative brand mentions |
|  **Competitor Tracking** | Identifies which competitors appear in AI responses |
|  **Batch Scanning** | Test multiple prompts in one scan |
|  **Data Export** | Download results as CSV for reporting |
|  **Scan History** | SQLite database stores all scan results |

##  How It Works

```
User Input → AI Query → Response Analysis → Visibility Metrics
```

| Step | Action | Output |
|------|--------|--------|
| 1️⃣ | User enters brand name and prompts | Configuration stored |
| 2️⃣ | System queries Gemini AI for each prompt | Raw AI responses |
| 3️⃣ | AI analyzes responses for brand mentions | Mention count + sentiment |
| 4️⃣ | System extracts competitor brand names | Competitor list |
| 5️⃣ | Calculate visibility score | Score = (mentions / total prompts) × 100 |
| 6️⃣ | Display results with charts and export | Dashboard + CSV |

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│                        (app.py)                             │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             ▼                                ▼
┌────────────────────────┐      ┌────────────────────────────┐
│   Gemini AI Client     │      │   SQLite Database          │
│  (gemini_client.py)    │      │    (database.py)           │
│                        │      │                            │
│ • query_ai()           │      │ • scans table              │
│ • analyze_brand()      │      │ • scan_results table       │
│ • extract_competitors()│      │ • get/save operations      │
└────────────┬───────────┘      └────────────────────────────┘
             │
             ▼
┌────────────────────────┐
│  Google Gemini API     │
│  (gemini-2.0-flash)    │
└────────────────────────┘
```

##  User Journey

**Scenario**: A startup wants to measure their visibility for "best project management tools"

1. User opens the app and enters brand name: "Asana"
2. User adds prompts:
   - "What are the best project management tools?"
   - "Recommend a tool for remote team collaboration"
   - "Which PM software integrates with Slack?"
3. User clicks "Run Visibility Scan"
4. System queries Gemini AI with each prompt
5. AI responses are analyzed:
   - Prompt 1: Asana mentioned (positive sentiment)
   - Prompt 2: Asana not mentioned
   - Prompt 3: Asana mentioned (neutral sentiment)
6. Dashboard shows:
   - Visibility Score: 66.7% (2/3 prompts)
   - Total Mentions: 2
   - Positive Sentiment: 1/2
   - Competitors: Trello, Monday.com, Jira
7. User downloads CSV for reporting

##  Technology Stack

### Frontend
- **Streamlit** - Web UI framework for rapid prototyping
- **Pandas** - Data manipulation and CSV export
- **Python 3.8+** - Core language

### AI & Analysis
- **Google Gemini API** - AI model for queries and analysis (gemini-2.0-flash)
- **JSON parsing** - Structured data extraction from AI responses
- **Regex** - Fallback text analysis

### Data Storage
- **SQLite** - Lightweight database for scan history
- **python-dotenv** - Environment variable management

### Deployment
- **Streamlit Cloud** - Hosted deployment platform

##  Local Setup

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd ai-visibility-tracker
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

4. Run the application
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Yes |

##  Contribution Guidelines

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Areas for Contribution
- Add support for additional AI models
- Improve sentiment analysis accuracy
- Build historical tracking features
- Enhance UI/UX design
- Write tests
- Improve documentation

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**Built with ❤️ for the future of AI-driven marketing**

[Report Bug](https://github.com/yourusername/ai-visibility-tracker/issues) • [Request Feature](https://github.com/yourusername/ai-visibility-tracker/issues)

</div>
