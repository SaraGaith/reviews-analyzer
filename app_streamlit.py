# app_streamlit.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from just_try import EnhancedFeedbackAnalyzer  # الكلاس تبعك

load_dotenv()

st.set_page_config(page_title="Review Analyzer", page_icon="🧠", layout="centered")
st.title("🧠 Reviews Analyzer (Booking / TripAdvisor)")

# ==== Sidebar: إعدادات ====
with st.sidebar:
    st.header("⚙️ Settings")

    # من .env
    env_api_key_airtable = os.getenv("AIRTABLE_API_KEY", "")
    env_base_id          = os.getenv("AIRTABLE_BASE_ID", "")
    env_table            = os.getenv("AIRTABLE_RESULTS_TABLE", "LLM Analysis Results")
    env_anthropic        = os.getenv("ANTHROPIC_API_KEY", "")
    env_apify            = os.getenv("APIFY_TOKEN", "")

    # إدخال المستخدم (يسمح يغيّر)
    ui_api_key_airtable = st.text_input("Airtable API Key", value=env_api_key_airtable, type="password")
    ui_base_id          = st.text_input("Airtable Base ID", value=env_base_id)
    ui_table            = st.text_input("Table name", value=env_table)
    ui_anthropic        = st.text_input("Anthropic API Key", value=env_anthropic, type="password")

    st.caption(f"Current Base: `{ui_base_id or '—'}`")
    st.caption(f"Current Table: `{ui_table or '—'}`")

    if not ui_api_key_airtable or not ui_base_id:
        st.warning("Airtable credentials are missing. Add them here or in .env")
    if not ui_anthropic:
        st.warning("Anthropic API key is missing. Add it here or in .env")

# وضع التشغيل
mode = st.radio("Choose mode:", ["Analyze local file", "Analyze Apify dataset"], horizontal=True)

# حضّري الأنالايزر مع مفاتيحك
analyzer = EnhancedFeedbackAnalyzer(api_key=ui_anthropic, advanced_analysis=True)
analyzer.set_airtable(api_key=ui_api_key_airtable, base_id=ui_base_id, table=ui_table)

def process_local_file(uploaded_file):
    if not ui_api_key_airtable or not ui_base_id:
        st.error("Airtable API Key و Base ID مطلوبين.")
        return
    if not ui_anthropic:
        st.error("Anthropic API Key مطلوب.")
        return

    # حفظ مؤقت لنعطي الدالة مسار فعلي
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        df = analyzer.read_file_enhanced(tmp_path)
        if df is None or df.empty:
            st.error("File is empty or cannot be read.")
            return

        df = analyzer._ensure_source_column(df)

        column_mapping = analyzer.smart_column_mapping_enhanced(df)
        with st.expander("Column mapping preview", expanded=False):
            st.json(column_mapping)

        with st.spinner("Processing reviews with LLM…"):
            # نفس البايبلاين تبعك (ترجع 'airtable://pushed' عند النجاح)
            result = analyzer._process_apify_data(df, column_mapping, tmp_path)

        if result:
            st.success("✅ Done! Pushed results to Airtable.")
        else:
            st.error("❌ Processing failed.")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def process_apify(dataset_id, apify_token):
    if not ui_api_key_airtable or not ui_base_id:
        st.error("Airtable API Key و Base ID مطلوبين.")
        return
    if not ui_anthropic:
        st.error("Anthropic API Key مطلوب.")
        return
    if not dataset_id.strip():
        st.error("Dataset ID is required.")
        return
    if not apify_token.strip():
        st.error("APIFY_TOKEN is required.")
        return

    with st.spinner(f"Downloading & analyzing dataset {dataset_id}…"):
        result = analyzer.analyze_apify_dataset(dataset_id, apify_token)

    if result:
        st.success("✅ Done! Pushed results to Airtable.")
    else:
        st.error("❌ Processing failed.")

if mode == "Analyze local file":
    st.subheader("📁 Upload a file (Excel / CSV / JSON)")
    file = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv", "json"])
    st.button("Run analysis", type="primary", disabled=(file is None), on_click=lambda: process_local_file(file) if file else None)
else:
    st.subheader("🕷️ Analyze Apify Dataset")
    dataset_id = st.text_input("Dataset ID")
    apify_token = st.text_input("APIFY_TOKEN", value=os.getenv("APIFY_TOKEN", ""), type="password")
    st.button("Run analysis", type="primary", on_click=lambda: process_apify(dataset_id, apify_token))

st.caption("Results are pushed to Airtable table set in the sidebar (or your .env)")
