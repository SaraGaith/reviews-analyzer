# app_streamlit.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from just_try import EnhancedFeedbackAnalyzer  # الكلاس تبعك

load_dotenv()

st.set_page_config(page_title="Review Analyzer", page_icon="🧠", layout="centered")
st.title("🧠 Reviews Analyzer (Booking / TripAdvisor)")

# ==== Sidebar: إدخال إعدادات Airtable من المستخدم ====
with st.sidebar:
    st.header("⚙️ Settings")

    # القيَم الافتراضية من .env
    env_api_key = os.getenv("AIRTABLE_API_KEY", "")
    env_base_id = os.getenv("AIRTABLE_BASE_ID", "")
    env_table   = os.getenv("AIRTABLE_RESULTS_TABLE", "LLM Analysis Results")

    ui_api_key = st.text_input("Airtable API Key", value=env_api_key, type="password")
    ui_base_id = st.text_input("Airtable Base ID", value=env_base_id)
    ui_table   = st.text_input("Table name", value=env_table)

    st.caption(f"Current Base: `{ui_base_id or '—'}`")
    st.caption(f"Current Table: `{ui_table or '—'}`")

    if not ui_api_key or not ui_base_id:
        st.warning("Airtable credentials are missing. Add them here or in .env")

# وضع الاختيار
mode = st.radio("Choose mode:", ["Analyze local file", "Analyze Apify dataset"], horizontal=True)

# نحضّر الأنالايزر ونمرر إعدادات Airtable التي أدخلها المستخدم (مرة واحدة فقط)
analyzer = EnhancedFeedbackAnalyzer(advanced_analysis=True)
analyzer.set_airtable(api_key=ui_api_key, base_id=ui_base_id, table=ui_table)


def process_local_file(uploaded_file):
    """
    نشغّل نفس البايبلاين تبعك لكن بدون Tkinter:
    - نقرأ الملف
    - نضيف عمود source (مهم لعامل ×2 لترب أدفايزر)
    - نعمل mapping
    - نستخدم _process_apify_data نفسها لتكملة المعالجة والدفع إلى Airtable
    """
    if not ui_api_key or not ui_base_id:
        st.error("Airtable API Key و Base ID مطلوبين.")
        return

    # احفظ الملف المؤقت (Streamlit بيحتاج مسار)
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # نفس خطواتك الأساسية:
        df = analyzer.read_file_enhanced(tmp_path)
        if df is None or df.empty:
            st.error("File is empty or cannot be read.")
            return

        # 👇 مهم: حتى بمسار الملف المحلي نحدد المصدر (booking/tripadvisor)
        df = analyzer._ensure_source_column(df)

        # mapping ذكي للأعمدة
        column_mapping = analyzer.smart_column_mapping_enhanced(df)

        # عرض بسيط للمابنج
        with st.expander("Column mapping preview", expanded=False):
            st.json(column_mapping)

        # تشغيل المعالجة؛ هاي الدالة عندك ترجع "airtable://pushed" لما تنجح
        with st.spinner("Processing reviews with LLM…"):
            result = analyzer._process_apify_data(df, column_mapping, tmp_path)

        if result:
            st.success("✅ Done! Pushed results to Airtable.")
        else:
            st.error("❌ Processing failed.")

    finally:
        # تنظيف الملف المؤقت
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def process_apify(dataset_id, apify_token):
    if not ui_api_key or not ui_base_id:
        st.error("Airtable API Key و Base ID مطلوبين.")
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
    if st.button("Run analysis", type="primary", disabled=(file is None)):
        process_local_file(file)
else:
    st.subheader("🕷️ Analyze Apify Dataset")
    dataset_id = st.text_input("Dataset ID")
    apify_default = os.getenv("APIFY_TOKEN", "")
    apify_token = st.text_input("APIFY_TOKEN", value=apify_default, type="password")
    if st.button("Run analysis", type="primary"):
        process_apify(dataset_id, apify_token)

st.caption("Results are pushed to Airtable table set in the sidebar (or your .env)")
