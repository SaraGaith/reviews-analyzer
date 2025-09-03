# 🧠 Reviews Analyzer (Booking / TripAdvisor)

Streamlit project to analyze hotel reviews (Booking / TripAdvisor) using LLM and connect the results with Airtable.  
The project allows you to upload a local file (Excel/CSV/JSON) or fetch a dataset from Apify, analyze the reviews, and send the results to Airtable.

---

## 📂 Project Structure

```
reviews-analyzer/
│
├── app_streamlit.py      # Main Streamlit UI
├── just_try.py           # EnhancedFeedbackAnalyzer class (core logic)
├── requirements.txt      # Dependencies
├── .env.example          # Example environment file (without real keys)
└── README.md             # Documentation (this file)
```

---

## ⚙️ Requirements

- Python 3.9 or higher
- Airtable account (Base ID, API Key, Table Name)
- Apify account (for datasets)
- Claude API Key (Anthropic) or any compatible LLM

---

## 🛠️ Installation & Local Run

1. **Clone the repo**:
   ```bash
   git clone https://github.com/YourUsername/reviews-analyzer.git
   cd reviews-analyzer
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows

   pip install -r requirements.txt
   ```

3. **Add `.env` file** (copy from `.env.example` and insert your real keys):
   ```
   AIRTABLE_API_KEY=your_airtable_api_key
   AIRTABLE_BASE_ID=your_base_id
   AIRTABLE_RESULTS_TABLE=LLM Analysis Results
   APIFY_TOKEN=your_apify_token
   CLAUDE_API_KEY=your_claude_key
   ```

4. **Run Streamlit app**:
   ```bash
   streamlit run app_streamlit.py
   ```

5. Open browser at:
   ```
   http://localhost:8501
   ```

---

## 🌐 Deployment on Streamlit Cloud

1. Link your GitHub repo with [Streamlit Cloud](https://streamlit.io/cloud).
2. Create **New App** → select this repo.
3. Branch: `main`, File: `app_streamlit.py`.
4. Go to Settings → Secrets → add your environment variables (.env values).
5. After build, you’ll get a public link like:
   ```
   https://your-app.streamlit.app
   ```

---

## ✨ Features

- Upload Excel/CSV/JSON files for review analysis.
- Enter Apify Dataset ID for online datasets.
- Automatic column mapping and source detection.
- Push analyzed results directly to Airtable.
- User-friendly Streamlit interface.

---

## 📌 Notes

- **API Keys** must be stored in `.env` or Streamlit Secrets (never hardcoded).  
- Do not upload real secrets to GitHub (only `.env.example`).  
- This project is for academic and research purposes.
