# 🧠 Reviews Analyzer 

Streamlit project to analyze hotel reviews using LLM and connect the results with Airtable.  
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
- Claude API Key (Anthropic) 

---

## ✨ Features

- Upload Excel/CSV/JSON files for review analysis.
- Enter Apify Dataset ID for online datasets.
- Automatic column mapping and source detection.
- Push analyzed results directly to Airtable.
- User-friendly Streamlit interface.

---

## 📌 Notes
 
- This project is for academic and research purposes.



