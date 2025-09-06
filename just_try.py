import csv
import re
import pandas as pd
from datetime import datetime
import os
import requests
import json
from typing import Optional, List, Dict, Any, Tuple, Union
import logging
from collections import Counter
import hashlib
import base64
import time
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_RESULTS_TABLE = os.getenv("AIRTABLE_RESULTS_TABLE", "LLM Analysis Results")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# إضافة مكتبة لكشف اللغة
try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect غير متوفرة. سيتم استخدام الكشف البسيط للغة.")
    print("💡 لتحسين دقة كشف اللغة، قم بتنفيذ: pip install langdetect")

# إضافة مكتبة للتحليل المتقدم
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy غير متوفرة. سيتم استخدام التحليل البسيط.")
    print("💡 لتحسين دقة التحليل، قم بتنفيذ: pip install spacy")


class ApifyDatasetDownloader:
    """Simple Apify dataset downloader"""
    def __init__(self, apify_token: str):
        self.apify_token = apify_token
        self.base_url = "https://api.apify.com/v2"

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_dataset(self, dataset_id: str) -> Optional[str]:
        """Download dataset from Apify and save as JSON"""

        try:
            self.logger.info(f"📥 Downloading Apify dataset: {dataset_id}")

            # Download from Apify API
            url = f"{self.base_url}/datasets/{dataset_id}/items"
            headers = {'Authorization': f'Bearer {self.apify_token}'}

            response = requests.get(url, headers=headers, timeout=120)

            if response.status_code != 200:
                self.logger.error(f"❌ Apify download failed: {response.status_code}")
                return None

            data = response.json()

            if not data:
                self.logger.error("❌ No data in dataset")
                return None

            # Save to JSON file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = f"apify_reviews_{dataset_id}_{timestamp}.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"✅ Downloaded {len(data)} reviews to: {json_file}")
            return json_file

        except Exception as e:
            self.logger.error(f"❌ Download error: {e}")
            return None


class AirtableClient:
    def __init__(self, api_key: str, base_id: str):
        self.base_url = f"https://api.airtable.com/v0/{base_id}"
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        self.logger = logging.getLogger(__name__)

    def create_batch(self, table: str, rows: list[dict]) -> None:
        BATCH_SIZE = 10  # <-- أهم سطر
        MAX_RETRIES = 3

        total = len(rows)
        for start in range(0, total, BATCH_SIZE):
            chunk = rows[start:start + BATCH_SIZE]
            payload = {"records": [{"fields": r} for r in chunk]}

            attempt = 0
            while True:
                attempt += 1
                r = self.s.post(f"{self.base_url}/{table}", data=json.dumps(payload), timeout=60)

                if r.status_code < 300:
                    self.logger.info("✅ Batch %d/%d sent (%d records)",
                                     start // BATCH_SIZE + 1,
                                     (total + BATCH_SIZE - 1) // BATCH_SIZE,
                                     len(chunk))
                    break

                # لو Rate limit أو خطأ مؤقت: جرّبي تاني مع backoff
                if r.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                    self.logger.warning("⏳ Retry batch after %ss (status %s): %s", wait, r.status_code, r.text)
                    time.sleep(wait)
                    continue

                # 422 أو أي خطأ دائم: اطبعي التفاصيل وارفع الاستثناء
                self.logger.error("Airtable create error %s: %s", r.status_code, r.text)
                r.raise_for_status()

            # اختيارياً: هدّئي الإيقاع نقطة صغيرة
            time.sleep(0.2)


class EnhancedFeedbackAnalyzer:
    # Constants - Updated for Claude API
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Using Claude 3.5 Sonnet
    MAX_TOKENS = 2000  # Increased for better processing
    TEMPERATURE = 0.1  # Lower for more consistent outputs
    TIMEOUT = 60  # Increased timeout for complex processing

    # Language detection thresholds
    HEBREW_THRESHOLD = 0.25
    ENGLISH_THRESHOLD = 0.75

    # Text quality thresholds - simplified since LLM will handle quality
    MIN_TEXT_LENGTH = 3  # Minimal check, let LLM decide quality
    MIN_WORDS = 1

    # Analysis quality settings
    MAX_RETRIES = 3
    CONFIDENCE_THRESHOLD = 0.7


    def _ensure_source_column(self, df: pd.DataFrame) -> pd.DataFrame:
        # لو العمود موجود أصلاً، خلّيه
        if 'source' in df.columns:
            return df

        cols = {c.lower() for c in df.columns}

        # دلائل Booking و TripAdvisor
        looks_booking = any(k in cols for k in ['reviewername', 'reviewdate', 'positivetext', 'negativetext'])
        looks_tripad = any(k in cols for k in ['username', 'publisheddate', 'bubblerating', 'text'])

        default_source = 'booking' if looks_booking and not looks_tripad else \
            'tripadvisor' if looks_tripad and not looks_booking else 'unknown'

        def guess_row(row):
            # إشارات صفية أوضح
            if 'bubbleRating' in df.columns or 'publishedDate' in df.columns:
                if pd.notna(row.get('bubbleRating')) or pd.notna(row.get('publishedDate')):
                    return 'tripadvisor'
            if 'reviewerName' in df.columns or 'reviewDate' in df.columns:
                if pd.notna(row.get('reviewerName')) or pd.notna(row.get('reviewDate')):
                    return 'booking'
            return default_source

        df['source'] = df.apply(guess_row, axis=1)
        return df

    def _gen_fallback_name(self, source: str = "") -> str:
        self.name_seq += 1  # يبدأ من 0 في __init__ → أول اسم = Review_1
        return f"Review_{self.name_seq}"

    def __init__(self, api_key: Optional[str] = None, use_session: bool = True, advanced_analysis: bool = True):
        """
        Enhanced Feedback Analyzer with LLM-Powered Analysis

        Args:
            api_key: Claude API key
            use_session: Whether to use requests.Session for better performance
            advanced_analysis: Enable advanced NLP features
        """


        self.name_seq = 0
        self.airtable_overrides = {"api_key": None, "base_id": None, "table": None}


        # Setup logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        # API Configuration - Updated for Claude
        self.claude_api_key = api_key
        self.use_llm_parsing = True
        self.advanced_analysis = advanced_analysis

        # Setup requests session for Claude API
        if use_session:
            self.session = requests.Session()
            self.session.headers.update({
                'x-api-key': self.claude_api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            })
        else:
            self.session = None

        # Initialize spaCy if available and advanced analysis is enabled
        self.nlp = None
        if advanced_analysis and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("🧠 spaCy NLP model loaded successfully")
            except OSError:
                self.logger.warning(
                    "spaCy English model not found. Install with: python -m spacy download en_core_web_sm")

        # for repeated operations
        self.language_cache = {}
        self.translation_cache = {}
        self.topic_cache = {}

        self.logger.info("🤖 Enhanced Claude LLM-Powered Analysis Ready!")

        # Enhanced content patterns for column detection only
        self.content_patterns = {
            'name_patterns': [
                'name', 'author', 'customer', 'client', 'user', 'reviewer', 'guest', 'visitor',
                'שם', 'לקוח', 'כותב', 'סCache וקר', 'משתמש', 'אורח', 'מבקר'
            ],
            'date_patterns': [
                'date', 'time', 'created', 'submitted', 'posted', 'review_date', 'when', 'timestamp',
                'reviewdate', 'review_date', 'created_at', 'published',
                'תאריך', 'זמן', 'נוצר', 'נשלח', 'פורסם', 'מתי'
            ],
            'location_patterns': [
                'country', 'nation', 'location', 'region', 'place', 'origin', 'from', 'city', 'state',
                'מדינה', 'ארץ', 'מקום', 'אזור', 'מיקום', 'עיר', 'מחוז'
            ],
            'rating_patterns': [
                'rating', 'score', 'stars', 'grade', 'points', 'evaluation', 'rank', 'review_score',
                'דירוג', 'ציון', 'כוכבים', 'נקודות', 'הערכה'
            ],
            'feedback_patterns': [
                'review', 'comment', 'feedback', 'text', 'content', 'message', 'description',
                'opinion', 'thoughts', 'experience', 'positive', 'negative', 'good', 'bad',
                'liked', 'disliked', 'title', 'pros', 'cons', 'likedText',
                'ביקורת', 'תגובה', 'פידבק', 'טקסט', 'תוכן', 'הודעה', 'תיאור', 'דעה'
            ]
        }

        # Operational topics definitions for the LLM
        self.operational_topics = {
            'cleanliness': {
                'description': 'Feedback about room cleanliness, bathroom hygiene, housekeeping standards, tidiness, and sanitary conditions',
                'examples': ['bathroom was spotless', 'room needs better cleaning', 'housekeeping did excellent job'],
                # 'weight': 1.2
            },
            'staff': {
                'description': 'Comments about hotel staff service, reception quality, employee helpfulness, professionalism, and customer service interactions',
                'examples': ['receptionist was very helpful', 'staff ignored our requests',
                             'excellent customer service'],
                # 'weight': 1.3
            },
            'location': {
                'description': 'Opinions about hotel location, proximity to attractions, accessibility, transportation links, and surrounding area',
                'examples': ['great location near beach', 'too far from city center', 'convenient for airport'],
                # 'weight': 1.1
            },
            'food': {
                'description': 'Reviews of dining experience, breakfast quality, restaurant service, menu variety, and meal satisfaction',
                'examples': ['breakfast was delicious', 'limited dining options', 'restaurant food was cold'],
                # 'weight': 1.2
            },
            'room_size': {
                'description': 'Comments about room dimensions, space adequacy, cramped conditions, or spaciousness',
                'examples': ['room was very spacious', 'felt cramped and small', 'good size for family'],
                # 'weight': 1.0
            },
            'room_quality': {
                'description': 'Feedback about room amenities, furniture condition, bed comfort, bathroom facilities, and overall room standards',
                'examples': ['comfortable bed and pillows', 'outdated furniture', 'bathroom needs renovation'],
                # 'weight': 1.1
            },
            'noise': {
                'description': 'Comments about noise levels, sound insulation, peaceful environment, or disturbances',
                'examples': ['very quiet and peaceful', 'noisy air conditioning', 'could hear neighbors'],
                # 'weight': 1.0
            },
            'parking': {
                'description': 'Feedback about parking availability, convenience, cost, and parking facilities',
                'examples': ['convenient free parking', 'expensive valet service', 'no parking available'],
                # 'weight': 0.9
            },
            'facilities': {
                'description': 'Reviews of hotel amenities like pool, gym, WiFi, spa, elevators, and recreational facilities',
                'examples': ['great pool area', 'WiFi was unreliable', 'gym equipment was broken'],
                # 'weight': 1.0
            },
            'price': {
                'description': 'Comments about value for money, pricing fairness, cost effectiveness, and financial value',
                'examples': ['excellent value for money', 'overpriced for quality', 'reasonable rates'],
                # 'weight': 1.1
            },
            'check_in_out': {
                'description': 'Feedback about arrival and departure procedures, front desk efficiency, and check-in/out experience',
                'examples': ['smooth check-in process', 'long wait at reception', 'early check-out was easy'],
                # 'weight': 0.8
            }
        }

        # داخل الكلاس EnhancedFeedbackAnalyzer (على نفس مستوى بقية الميثودز)
    def _is_country_like(self, col_lower: str) -> bool:
        col_lower = (col_lower or "").lower()
        tokens = ['country', 'userlocation', 'location', 'nation', 'city', 'state', 'origin', 'region']
        return any(tok in col_lower for tok in tokens)

    def set_airtable(self, api_key: str | None = None, base_id: str | None = None, table: str | None = None):
        if api_key:
            self.airtable_overrides["api_key"] = api_key.strip()
        if base_id:
            self.airtable_overrides["base_id"] = base_id.strip()
        if table:
            self.airtable_overrides["table"] = table.strip()

    def analyze_apify_dataset(self, dataset_id: str, apify_token: str) -> Optional[str]:
        """
        Analyze reviews from Apify dataset ID and output Excel file

        Args:
            dataset_id: <YOUR_DATASET_ID>
            apify_token: <APIFY_TOKEN_FROM_ENV>

        Returns:
            Path to Excel analysis results file
        """

        self.logger.info(f"🕷️ APIFY INTEGRATION: Analyzing Dataset {dataset_id}")
        self.logger.info("=" * 60)

        try:
            # Download dataset from Apify
            downloader = ApifyDatasetDownloader(apify_token)
            json_file = downloader.download_dataset(dataset_id)

            if not json_file:
                self.logger.error("❌ Failed to download Apify dataset")
                return None

            # Load and process the JSON file
            df = self.read_file_enhanced(json_file)
            df = self._ensure_source_column(df)

            self.logger.info(f"📋 Loaded {len(df)} reviews from Apify")

            # Map Apify columns to your expected format
            column_mapping = self._detect_apify_columns(df)

            # Run analysis with your existing logic
            result = self._process_apify_data(df, column_mapping, json_file)

            # Clean up temporary JSON file
            try:
                os.remove(json_file)
                self.logger.info(f"🗑️ Cleaned up temporary file: {json_file}")
            except:
                pass

            return result

        except Exception as e:
            self.logger.error(f"❌ Apify analysis error: {e}")
            return None

    def _detect_apify_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Detect and map Apify column structure to expected format"""

        columns = df.columns.tolist()
        self.logger.info(f"🔍 Apify columns found: {columns}")

        mapping = {
            'name': None, 'date': None, 'country': None, 'rating': None,
            'review_title': None, 'liked_text': None, 'disliked_text': None,
            'general_feedback': None, 'trip_type': None
        }

        # Common Apify field mappings for different scrapers
        apify_field_map = {
            # Booking.com
            'reviewerName': 'name',
            'userName': 'name',  # <— جديد
            'user_name': 'name',  # احتياط
            'user.name': 'name',  # لو اتسطّح بـ flatten
            'reviewDate': 'date',
            'reviewerCountry': 'country',
            'userLocation': 'country',  # <— أحياناً Apify ترجعها كذا
            'user_location': 'country',
            'rating': 'rating',
            'reviewTitle': 'review_title',
            'positiveText': 'liked_text',
            'negativeText': 'disliked_text',
            'reviewText': 'general_feedback',
            'tripType': 'trip_type',

            # TripAdvisor
            'username': 'name',
            'travelDate': 'date',
            'bubbleRating': 'rating',
            'title': 'review_title',
            'text': 'general_feedback',

            # Generic
            'author': 'name',
            'review_date': 'date',
            'location': 'country',
            'review_rating': 'rating',
            'review_score': 'rating',
            'score': 'rating',
            'liked': 'liked_text',
            'disliked': 'disliked_text',
            'positive': 'liked_text',
            'negative': 'disliked_text',
            'content': 'general_feedback',
            'review': 'general_feedback',
            'comment': 'general_feedback',
        }

        # Direct mapping first
        for col in columns:
            if col in apify_field_map:
                target = apify_field_map[col]
                mapping[target] = col
                self.logger.info(f"✅ Direct mapped: {col} → {target}")

        # Smart mapping for unmapped fields
        unmapped = [k for k, v in mapping.items() if v is None]
        if unmapped:
            self.logger.info(f"🧠 Running smart mapping for: {unmapped}")
            smart_mapping = self.smart_column_mapping_enhanced(df)

            for field in unmapped:
                if field in smart_mapping and smart_mapping[field]:
                    mapping[field] = smart_mapping[field]
                    self.logger.info(f"🎯 Smart mapped: {smart_mapping[field]} → {field}")

        return mapping


    def _process_apify_data(self, df: pd.DataFrame, column_mapping: Dict[str, Optional[str]],
                            file_path: str) -> Optional[str]:
        """Process Apify data with existing analysis pipeline"""

        # Your existing processing logic
        all_processed = []
        skipped_rows = 0
        processing_stats = {
            'total_topics': 0,
            'positive_topics': 0,
            'negative_topics': 0,
            'languages_detected': set(),
            'api_calls': 0,
            'cache_hits': 0
        }

        self.logger.info(f"🔄 Processing {len(df)} scraped reviews with Claude...")

        for index, row in df.iterrows():
            if (index + 1) % 10 == 1:
                self.logger.info(f"📝 Processing rows {index + 1}-{min(index + 10, len(df))}/{len(df)}")

            try:
                processed_rows = self._process_row_enhanced(row, column_mapping, processing_stats)
                if processed_rows:
                    all_processed.extend(processed_rows)
                    processing_stats['total_topics'] += len(processed_rows)

                    for pr in processed_rows:
                        if pr['Positive_text']:
                            processing_stats['positive_topics'] += 1
                        elif pr['Negative_text']:
                            processing_stats['negative_topics'] += 1
                else:
                    skipped_rows += 1

            except Exception as e:
                self.logger.error(f"⚠️ Error processing row {index + 1}: {e}")
                skipped_rows += 1

            # Progress update every 25 rows
            if (index + 1) % 25 == 0:
                self._show_progress_stats(index + 1, len(df), processing_stats, skipped_rows)

        # Use existing finalization logic
        return self._finalize_enhanced_analysis(all_processed, file_path, processing_stats, skipped_rows, len(df))

    def get_text_hash(self, text: str) -> str:
        """Create hash for caching purposes"""
        return hashlib.md5(text.encode()).hexdigest()

    def detect_language(self, text: str) -> str:
        """Enhanced language detection with caching"""
        if not text or len(str(text).strip()) < 3:
            return 'unknown'

        text = str(text).strip()
        text_hash = self.get_text_hash(text)

        # Check cache first
        if text_hash in self.language_cache:
            return self.language_cache[text_hash]

        detected_lang = 'unknown'

        # Try langdetect first if available
        if LANGDETECT_AVAILABLE:
            try:
                detected = detect(text)
                language_map = {
                    'he': 'hebrew', 'en': 'english', 'ar': 'arabic',
                    'es': 'spanish', 'fr': 'french', 'de': 'german',
                    'it': 'italian', 'pt': 'portuguese', 'ru': 'russian',
                    'zh-cn': 'chinese', 'zh': 'chinese'
                }
                detected_lang = language_map.get(detected, 'other')
            except Exception as e:
                self.logger.debug(f"langdetect failed: {e}")
                detected_lang = self._simple_language_detection(text)
        else:
            detected_lang = self._simple_language_detection(text)

        # Cache the result
        self.language_cache[text_hash] = detected_lang
        return detected_lang

    def _simple_language_detection(self, text: str) -> str:
        """Enhanced simple language detection with Arabic support"""
        text_lower = text.lower()

        # Hebrew detection
        hebrew_chars = re.findall(r'[\u0590-\u05FF]', text)
        if hebrew_chars:
            total_chars = len(re.findall(r'[a-zA-Z\u0590-\u05FF\u0600-\u06FF]', text))
            hebrew_ratio = len(hebrew_chars) / max(total_chars, 1)
            if hebrew_ratio > self.HEBREW_THRESHOLD:
                return 'hebrew'

        # Arabic detection
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        if arabic_chars:
            total_chars = len(re.findall(r'[a-zA-Z\u0590-\u05FF\u0600-\u06FF]', text))
            arabic_ratio = len(arabic_chars) / max(total_chars, 1)
            if arabic_ratio > 0.3:
                return 'arabic'

        # English detection
        english_chars = re.findall(r'[a-zA-Z]', text)
        if english_chars:
            total_chars = len(re.findall(r'[a-zA-Z\u0590-\u05FF\u0600-\u06FF]', text))
            english_ratio = len(english_chars) / max(total_chars, 1)
            if english_ratio > self.ENGLISH_THRESHOLD:
                return 'english'

        # Language-specific keyword detection
        language_keywords = {
            'spanish': ['muy', 'buena', 'excelente', 'ubicación', 'desayuno', 'habitación'],
            'french': ['très', 'bon', 'excellent', 'petit', 'déjeuner', 'chambre'],
            'german': ['sehr', 'gut', 'ausgezeichnet', 'frühstück', 'zimmer'],
            'italian': ['molto', 'buono', 'eccellente', 'colazione', 'camera'],
            'portuguese': ['muito', 'bom', 'excelente', 'café', 'quarto'],
            'russian': ['очень', 'хороший', 'отличный', 'завтрак', 'комната']
        }

        for lang, keywords in language_keywords.items():
            if any(word in text_lower for word in keywords):
                return lang

        # Unicode range detection
        if re.search(r'[\u0400-\u04FF]', text):
            return 'russian'
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'chinese'

        return 'other'

    def _create_llm_powered_prompt(self, text: str, sentiment_hint: Optional[str] = None) -> str:
        """Create LLM-powered prompt with few-shot examples and clear instructions"""
        detected_lang = self.detect_language(text)
        language_instruction = self._get_language_instruction(detected_lang)

        # Handle combined content format
        if sentiment_hint == 'mixed':
            sentiment_instruction = "CONTEXT: This text contains both positive and negative feedback from the same review, separated by ' | '. The format is 'POSITIVE: [positive content] | NEGATIVE: [negative content]'. Extract topics from both sections and assign appropriate sentiments."
        elif sentiment_hint:
            sentiment_instruction = f"CONTEXT: This text comes from a {sentiment_hint} feedback section."
        else:
            sentiment_instruction = ""

        # Create operational topics description for the prompt
        topics_description = "\n".join([
            f"• **{topic}**: {info['description']}"
            for topic, info in self.operational_topics.items()
        ])

        prompt = f"""You are an expert hotel operations analyzer. Your task is to extract ONLY operationally actionable feedback that hotel management can use to improve their services.

    **DETECTION LANGUAGE**: {detected_lang.upper()}
    {language_instruction}

    **OPERATIONAL FOCUS AREAS**:
    {topics_description}

    **EXTRACTION PRINCIPLES**:
    1. **Focus on Specificity**: Extract feedback that mentions specific operational aspects, not generic sentiments
    2. **Actionable Insights**: Prioritize comments that suggest concrete actions or improvements
    3. **Ignore Generic Praise/Complaints**: Skip vague statements like "great stay" or "terrible experience" without specific context
    4. **Context-Aware Cleaning**: Automatically ignore logistics info (arrival times, booking details) while preserving meaningful operational feedback
    5. **Each employee's name should be on a separate line, while maintaining sentence consistency. That is, if two employees are praised, the same praise should be added to each employee on a separate line.
    6. **Handle Combined Content**: If the input contains 'POSITIVE:' and 'NEGATIVE:' sections, extract topics from both and assign appropriate sentiments.
    7. **Translation rule**: If the input is HEBREW, keep extracted text in HEBREW, Otherwise, translate extracted text to ENGLISH.
    8. **TOPIC CONSOLIDATION**:
            - Each operational topic = separate output row (cleanliness, staff, parking, food, location, noise, room size)
            - Merge repetitive sentences about same topic with "..." connector
            - Example: "The staff is exceptional ... goes above and beyond"
    9. - No emojis.        
    10. **PRESERVE GUEST ADVICE**:
        - Keep guest suggestions/recommendations exactly as written
        - Maintain original tone and phrasing of constructive suggestions       

    **FEW-SHOT EXAMPLES**:

    **Example 1:**
    Input: "Amazing stay! We arrived at 10 PM and the receptionist Maria was incredibly helpful with late check-in. The room was a bit small but very clean. Overall fantastic experience, will definitely come back next year!"

    Output:
    [
      {{"topic": "staff", "text": "receptionist Maria was incredibly helpful with late check-in", "sentiment": "positive"}},
      {{"topic": "room_size", "text": "room was a bit small", "sentiment": "negative"}},
      {{"topic": "cleanliness", "text": "room was very clean", "sentiment": "positive"}}
    ]

    **Example 2:**
    Input: "POSITIVE: The location near the beach was perfect and breakfast had good variety | NEGATIVE: WiFi kept disconnecting during video calls and breakfast was cold most mornings"

    Output:
    [
      {{"topic": "location", "text": "location near the beach was perfect", "sentiment": "positive"}},
      {{"topic": "food", "text": "breakfast had good variety", "sentiment": "positive"}},
      {{"topic": "facilities", "text": "WiFi kept disconnecting during video calls", "sentiment": "negative"}},
      {{"topic": "food", "text": "breakfast was cold most mornings", "sentiment": "negative"}}
    ]

    **Example 3 (Hebrew):**
    Input: "השהייה הייתה נפלאה! הצוות במלון היה מאוד מקצועי ועזר לנו בכל בקשה. החדר היה נקי אבל קצת רועש בלילה מהכביש. המיקום מעולה, קרוב למרכז העיר."

    Output:
    [
      {{"topic": "staff", "text": "הצוות במלון היה מאוד מקצועי ועזר לנו בכל בקשה", "sentiment": "positive"}},
      {{"topic": "cleanliness", "text": "החדר היה נקי", "sentiment": "positive"}},
      {{"topic": "noise", "text": "קצת רועש בלילה מהכביש", "sentiment": "negative"}},
      {{"topic": "location", "text": "המיקום מעולה, קרוב למרכז העיר", "sentiment": "positive"}} 
    ]

    **Example 4 (Combined Content):**
    Input: "POSITIVE: השף שם מושלם האוכל היה מדהים ארוחת בוקר. מיקום. והצוות מדהים במיוחד מוחמד | NEGATIVE: החדר היה קטן מדי והרעש מהכביש הפריע"

    Output:
    [
      {{"topic": "staff", "text": "הצוות מדהים", "sentiment": "positive"}},
      {{"topic": "staff", "text": "מוחמד", "sentiment": "positive"}},
      {{"topic": "staff", "text": "השף שם מושלם", "sentiment": "positive"}},
      {{"topic": "food", "text": "האוכל היה מדהים", "sentiment": "positive"}},
      {{"topic": "food", "text": "ארוחת בוקר", "sentiment": "positive"}},
      {{"topic": "location", "text": "מיקום", "sentiment": "positive"}},
      {{"topic": "room_size", "text": "החדר היה קטן מדי", "sentiment": "negative"}},
      {{"topic": "noise", "text": "הרעש מהכביש הפריע", "sentiment": "negative"}}
    ]

    {sentiment_instruction}

    **ANALYZE THIS TEXT**:
    "{text}"

    **INSTRUCTIONS**:
    - If the text contains 'POSITIVE:' and 'NEGATIVE:' sections, process both parts
    - Extract ONLY specific, actionable operational feedback
    - Each extracted text should be substantial enough to guide improvements
    - Ignore generic expressions, booking logistics, and purely emotional statements
    - Maintain the original language as specified
    - Return valid JSON array format only

    **OUTPUT**:"""
        return prompt

    def _parse_with_claude_api_enhanced(self, text: str, sentiment_hint: Optional[str] = None) -> Optional[
        List[Dict[str, str]]]:
        """Enhanced Claude API parsing focusing on LLM-powered cleaning and extraction"""

        for attempt in range(self.MAX_RETRIES):
            try:
                prompt = self._create_llm_powered_prompt(text, sentiment_hint)

                # Claude API request format
                data = {
                    'model': self.CLAUDE_MODEL,
                    'max_tokens': self.MAX_TOKENS,
                    'temperature': self.TEMPERATURE,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                }

                if self.session:
                    response = self.session.post(
                        'https://api.anthropic.com/v1/messages',
                        json=data,
                        timeout=self.TIMEOUT
                    )
                else:
                    headers = {
                        'x-api-key': self.claude_api_key,
                        'Content-Type': 'application/json',
                        'anthropic-version': '2023-06-01'
                    }
                    response = requests.post(
                        'https://api.anthropic.com/v1/messages',
                        headers=headers,
                        json=data,
                        timeout=self.TIMEOUT
                    )

                if response.status_code == 200:
                    result = response.json()
                    # Claude API response format
                    content = result['content'][0]['text']

                    # Enhanced parsing with validation
                    parsed_topics = self._parse_llm_response_enhanced(content)
                    if parsed_topics and len(parsed_topics) > 0:
                        self.logger.info(
                            f"✅ Claude extracted {len(parsed_topics)} actionable topics (attempt {attempt + 1})")
                        return parsed_topics
                    else:
                        self.logger.warning(f"❌ No actionable topics found in attempt {attempt + 1}")
                else:
                    self._log_api_error(response.status_code)
                    if response.status_code == 429:  # Rate limit
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff

            except requests.exceptions.Timeout:
                self.logger.warning(f"⏰ Timeout in attempt {attempt + 1}")
            except Exception as e:
                self.logger.error(f"❌ Error in attempt {attempt + 1}: {e}")

        return None

    def _parse_llm_response_enhanced(self, response_text: str) -> Optional[List[Dict[str, str]]]:
        """Enhanced LLM response parsing with better validation"""
        try:
            # Multiple JSON extraction attempts
            json_patterns = [
                r'\[.*?\]',  # Basic array
                r'```json\s*(\[.*?\])\s*```',  # Markdown code block
                r'```\s*(\[.*?\])\s*```',  # Generic code block
                r'(?s)\[.*?\]'  # Multiline array
            ]

            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    json_str = match.group(1) if match.groups() else match.group(0)
                    break

            if not json_str:
                self.logger.warning("No JSON structure found in response")
                return None

            try:
                topics = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON: {e}")
                # Try to fix common JSON issues
                json_str = self._fix_common_json_issues(json_str)
                try:
                    topics = json.loads(json_str)
                except json.JSONDecodeError:
                    return None

            if not isinstance(topics, list):
                self.logger.warning("Response is not a list")
                return None

            # Lightweight validation since LLM handles quality
            return self._validate_llm_topics(topics)

        except Exception as e:
            self.logger.error(f"Error parsing enhanced LLM response: {e}")
            return None

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix unescaped quotes
        json_str = re.sub(r'(?<!\\)"(?![,:\]\}])', '\\"', json_str)

        # Ensure proper quote consistency
        json_str = re.sub(r'[""]', '"', json_str)

        return json_str

    def _validate_llm_topics(self, topics: List[Dict]) -> List[Dict[str, str]]:
        """Lightweight topic validation since LLM handles quality control"""
        valid_topics = []
        seen_combinations = set()

        for topic in topics:
            if not isinstance(topic, dict):
                continue

            if 'topic' not in topic or 'text' not in topic:
                continue

            # Trust LLM topic assignment but normalize names
            topic_name = self._normalize_topic_name(topic.get('topic', ''))
            if not topic_name:
                continue

            text = topic.get('text', '').strip()
            sentiment = topic.get('sentiment', 'neutral')

            # Basic length check - trust LLM for quality
            if len(text) < self.MIN_TEXT_LENGTH:
                continue

            # Avoid exact duplicates
            content_signature = f"{topic_name}:{text.lower()}"
            if content_signature in seen_combinations:
                continue
            seen_combinations.add(content_signature)

            # Validate sentiment
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'

            valid_topics.append({
                'topic': topic_name,
                'text': text,
                'sentiment': sentiment
            })

        # Sort by topic importance (weight)
        # valid_topics.sort(key=lambda x: self.operational_topics.get(x['topic'], {}).get('weight', 1.0), reverse=True)

        return valid_topics

    def _normalize_topic_name(self, topic_name: str) -> Optional[str]:
        """Normalize topic names to match operational categories"""
        topic_lower = topic_name.lower().strip()

        # Direct match
        if topic_lower in self.operational_topics:
            return topic_lower

        # Synonym mapping
        topic_synonyms = {
            'rooms': 'room_quality',
            'accommodation': 'room_quality',
            'service': 'staff',
            'employees': 'staff',
            'workers': 'staff',
            'dining': 'food',
            'restaurant': 'food',
            'meals': 'food',
            'internet': 'facilities',
            'wi-fi': 'facilities',
            'wifi': 'facilities',
            'gym': 'facilities',
            'cost': 'price',
            'value': 'price',
            'money': 'price',
            'arrival': 'check_in_out',
            'departure': 'check_in_out',
            'checkin': 'check_in_out',
            'checkout': 'check_in_out',
            'location': 'The_view'
        }

        if topic_lower in topic_synonyms:
            return topic_synonyms[topic_lower]

        # Return the original if it's reasonable (let LLM decide)
        if len(topic_lower) > 2 and topic_lower.replace('_', '').isalpha():
            return topic_lower

        return None

    def read_file_enhanced(self, file_path: str) -> Optional[pd.DataFrame]:
        """Enhanced file reading with JSON support and better error handling"""
        try:
            file_extension = file_path.lower().split('.')[-1]

            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
                self.logger.info(f"📊 Excel file loaded: {len(df)} rows")

            elif file_extension == 'csv':
                # Try multiple encodings
                encodings = ['utf-8', 'utf-8-sig', 'windows-1255', 'cp1252', 'iso-8859-8', 'windows-1256']
                df = None

                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        self.logger.info(f"📊 CSV file loaded with {encoding}: {len(df)} rows")
                        break
                    except UnicodeDecodeError:
                        continue

                if df is None:
                    df = pd.read_csv(file_path)  # Last resort with default encoding

            elif file_extension == 'json':
                # Enhanced JSON handling
                df = self._read_json_file(file_path)
                if df is not None:
                    self.logger.info(f"📊 JSON file loaded: {len(df)} rows")

            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Basic data validation
            if df is None or df.empty:
                self.logger.error("File is empty or couldn't be loaded")
                return None

            # Remove completely empty rows
            df = df.dropna(how='all')

            self.logger.info(f"✅ File processed successfully: {len(df)} valid rows")
            return df

        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
            return None

    def _read_json_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read and process JSON files with various structures"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                df = pd.json_normalize(data)

            elif isinstance(data, dict):
                if 'reviews' in data:
                    # Structure like {"reviews": [...]}
                    df = pd.json_normalize(data['reviews'])
                elif 'data' in data:
                    # Structure like {"data": [...]}
                    df = pd.json_normalize(data['data'])
                elif any(isinstance(v, list) for v in data.values()):
                    # Find the main data array
                    main_key = next(k for k, v in data.items() if isinstance(v, list))
                    df = pd.json_normalize(data[main_key])
                else:
                    # Single object - convert to single row DataFrame
                    df = pd.json_normalize([data])
            else:
                self.logger.error("Unsupported JSON structure")
                return None

            # Flatten nested s if any
            df = self._flatten_json_columns(df)

            return df

        except Exception as e:
            self.logger.error(f"Error reading JSON file: {e}")
            return None

    def _flatten_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested JSON columns"""
        for col in df.columns:
            # Check if column contains nested data
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, (dict, list)):
                    try:
                        # Try to normalize nested data
                        nested_df = pd.json_normalize(df[col].dropna())
                        # Prefix column names to avoid conflicts
                        nested_df.columns = [f"{col}.{subcol}" for subcol in nested_df.columns]

                        # Merge back with original DataFrame
                        df = df.drop(columns=[col])
                        df = pd.concat([df.reset_index(drop=True), nested_df.reset_index(drop=True)], axis=1)
                    except Exception:
                        # If normalization fails, keep original column
                        pass

        return df

    def smart_column_mapping_enhanced(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Enhanced column mapping with better pattern recognition"""
        columns = df.columns.tolist()
        mapping = {
            'name': None, 'date': None, 'country': None, 'rating': None,
            'review_title': None, 'liked_text': None, 'disliked_text': None,
            'general_feedback': None, 'trip_type': None
        }


        self.logger.info("🔍 Enhanced column analysis...")
        self.logger.info(f"📋 Available columns: {columns}")

        # Enhanced pattern matching
        enhanced_patterns = {
            'name': r'(^|_)(name|author|customer|guest|user(?!location)|reviewer)($|_)|^اسم$|^שם$|^לקוח$',
            'date': r'(date|time|created|posted|review.*date|timestamp|تاريخ|זמן|תאריך)',
            'country': r'(country|nation|location|region|origin|userlocation|מדינה|ארץ)',
            'rating': r'(rating|score|stars|grade|points|تقييم|דירוג|ציון)',
            'review_title': r'(^|_)(title|headline|subject)($|_)|عنوان|כותרת',
            'liked_text': r'(^|_)(liked|likedtext|pros|positive|advantages)($|_)|إيجابي|חיובי|טוב',
            'disliked_text': r'(^|_)(disliked|dislikedtext|cons|negative|disadvantages)($|_)|سلبي|שלילי|רע',
            'general_feedback': r'(^|_)(review|comment|feedback|text|content|message|description)($|_)|تعليق|ביקורת|תגובה',
            'trip_type': r'(trip|travel|type|purpose|category|نوع|סוג)'
        }

        # Score-based matching
        column_scores = {col: {} for col in columns}

        for col in columns:
            col_lower = col.lower().replace(' ', '_').replace('-', '_')
            content_type = self.analyze_column_content_enhanced(df, col)

            self.logger.info(f"🔎 Analyzing: {col} → {content_type}")
            # لا تسمح بترشيح أعمدة تشبه الدولة كـ name
            skip_name_for_this_col = self._is_country_like(col_lower)

            for purpose, pattern in enhanced_patterns.items():
                if purpose == 'name' and skip_name_for_this_col:
                    continue
                if re.search(pattern, col_lower, re.IGNORECASE):
                    score = 1.0
                    if any(word in col_lower for word in pattern.replace('(', '').replace(')', '').split('|')):
                        score += 0.5
                    if purpose == 'date' and content_type in ['date', 'timestamp']:
                        score += 1.0
                    elif purpose == 'rating' and content_type == 'rating':
                        score += 1.0
                    elif purpose in ['liked_text', 'disliked_text', 'general_feedback'] and content_type in [
                        'long_text', 'medium_text']:
                        score += 0.5
                    column_scores[col][purpose] = score

            # Pattern matching with scoring
            for purpose, pattern in enhanced_patterns.items():
                if re.search(pattern, col_lower, re.IGNORECASE):
                    score = 1.0

                    # Boost score for exact matches
                    if any(word in col_lower for word in pattern.replace('(', '').replace(')', '').split('|')):
                        score += 0.5

                    # Content type validation
                    if purpose == 'date' and content_type in ['date', 'timestamp']:
                        score += 1.0
                    elif purpose == 'rating' and content_type == 'rating':
                        score += 1.0
                    elif purpose in ['liked_text', 'disliked_text', 'general_feedback'] and content_type in [
                        'long_text', 'medium_text']:
                        score += 0.5

                    column_scores[col][purpose] = score

        # Assign columns based on highest scores
        used_columns = set()
        for purpose in mapping.keys():
            best_col = None
            best_score = 0

            for col in columns:
                if col in used_columns:
                    continue
                score = column_scores[col].get(purpose, 0)
                if score > best_score:
                    best_score = score
                    best_col = col

            if best_col and best_score > 0.5:  # Minimum confidence threshold
                mapping[purpose] = best_col
                used_columns.add(best_col)
                self.logger.info(f"✅ {purpose}: {best_col} (confidence: {best_score:.2f})")

        # حارس نهائي: لو Name وقع على عمود يشبه الدولة رجّعيه None
        if mapping.get('name') and self._is_country_like(mapping['name'].lower()):
            self.logger.warning("⚠️ Name mapped to a location-like column; resetting to None (will use fallback).")
            mapping['name'] = None

        return mapping

    def analyze_column_content_enhanced(self, df: pd.DataFrame, col_name: str, sample_size: int = 50) -> str:
        """Enhanced column content analysis"""
        sample_values = df[col_name].dropna().head(sample_size)

        if len(sample_values) == 0:
            return 'empty'

        # Convert to string for analysis
        str_values = sample_values.astype(str)

        analysis = {
            'avg_length': str_values.str.len().mean(),
            'has_dates': self._detect_patterns_enhanced(str_values, 'date'),
            'has_numbers': self._detect_patterns_enhanced(str_values, 'number'),
            'has_ratings': self._detect_patterns_enhanced(str_values, 'rating'),
            'has_timestamps': self._detect_patterns_enhanced(str_values, 'timestamp'),
            'has_multilingual': self._detect_patterns_enhanced(str_values, 'multilingual'),
            'content_diversity': len(set(str_values)) / len(str_values)
        }

        return self._classify_column_type_enhanced(analysis)

    def _detect_patterns_enhanced(self, values: pd.Series, pattern_type: str) -> float:
        """Enhanced pattern detection with multiple formats"""
        patterns = {
            'date': [
                r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}',
                r'\d{1,2}/\d{1,2}/\d{2,4}', r'\d{1,2}-\d{1,2}-\d{2,4}'
            ],
            'timestamp': [
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
                r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}'
            ],
            'number': [r'^\d+\.?\d*$'],
            'rating': [r'^[0-5](\.\d)?$', r'^[0-9](\.\d)?$', r'^10(\.0)?$'],
            'multilingual': [r'[\u0590-\u05FF]', r'[\u0600-\u06FF]', r'[\u4e00-\u9fff]']
        }
        if pattern_type not in patterns:
            return 0.0

        match_count = 0
        for value in values:
            str_value = str(value)
            if any(re.search(pattern, str_value) for pattern in patterns[pattern_type]):
                match_count += 1

        return match_count / len(values)

    def _classify_column_type_enhanced(self, analysis: Dict[str, float]) -> str:
        """Enhanced column type classification"""
        if analysis['has_timestamps'] > 0.7:
            return 'timestamp'
        if analysis['has_dates'] > 0.7:
            return 'date'
        if analysis['has_ratings'] > 0.7:
            return 'rating'
        if analysis['has_numbers'] > 0.8:
            return 'numeric'
        if analysis['has_multilingual'] > 0.3:
            return 'multilingual_text'
        if analysis['avg_length'] > 100:
            return 'long_text'
        if analysis['avg_length'] > 20:
            return 'medium_text'
        if analysis['content_diversity'] < 0.1:
            return 'categorical'

        return 'short_text'

    def run_enhanced_analysis(self, file_path: Optional[str] = None) -> Optional[str]:
        """Enhanced main analysis function with LLM-powered processing.
        If file_path is provided (e.g., from Streamlit upload), use it.
        Otherwise fall back to local file picker (only works on desktop)."""
        self.logger.info("🎯 LLM-Powered Review Processor - Advanced AI-Driven Analysis")
        self.logger.info("🧠 STAGE 1: LLM Context Understanding & Smart Cleaning")
        self.logger.info("✂️ STAGE 2: AI-Powered Topic Extraction & Classification")
        self.logger.info("🌐 STAGE 3: Intelligent Quality Assurance")
        self.logger.info("=" * 70)

        # لو وصل ملف من Streamlit/الخارج، استخدمه. غير هيك جرّب الـ picker المحلي.
        if not file_path:
            file_path = self._select_file_enhanced()
            if not file_path:
                self.logger.error("No file selected (headless/server mode needs an uploaded file).")
                return None

        try:
            # Enhanced file reading
            df = self.read_file_enhanced(file_path)
            if df is None or df.empty:
                self.logger.error("File is empty or cannot be read")
                return None

            df = self._ensure_source_column(df)
            self.logger.info(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")

            # Enhanced column mapping
            column_mapping = self.smart_column_mapping_enhanced(df)
            if not any(column_mapping.values()):
                self.logger.error("❌ Could not identify any relevant columns")
                return None

            # Processing
            all_processed = []
            skipped_rows = 0
            processing_stats = {
                'total_topics': 0,
                'positive_topics': 0,
                'negative_topics': 0,
                'languages_detected': set(),
                'api_calls': 0,
                'cache_hits': 0
            }

            self.logger.info(f"\n🔄 Processing {len(df)} rows with LLM-powered analysis...")
            for index, row in df.iterrows():
                if (index + 1) % 10 == 1:
                    self.logger.info(f"\n{'=' * 60}")
                    self.logger.info(f"📝 Processing rows {index + 1}-{min(index + 10, len(df))}/{len(df)}")

                try:
                    processed_rows = self._process_row_enhanced(row, column_mapping, processing_stats)
                    if processed_rows:
                        all_processed.extend(processed_rows)
                        processing_stats['total_topics'] += len(processed_rows)
                        for pr in processed_rows:
                            if pr['Positive_text']:
                                processing_stats['positive_topics'] += 1
                            elif pr['Negative_text']:
                                processing_stats['negative_topics'] += 1
                    else:
                        skipped_rows += 1
                except Exception as e:
                    self.logger.error(f"⚠️ Error processing row {index + 1}: {str(e)}")
                    skipped_rows += 1

                if (index + 1) % 25 == 0:
                    self._show_progress_stats(index + 1, len(df), processing_stats, skipped_rows)

            # Finalize
            return self._finalize_enhanced_analysis(all_processed, file_path, processing_stats, skipped_rows, len(df))

        except Exception as e:
            self.logger.error(f"❌ Critical error: {str(e)}")
            return None

    def _select_file_enhanced(self) -> Optional[str]:
        """Local-only file picker. Returns None on servers/Streamlit."""
        import os, sys
        # سيرفر/Streamlit: لا تستخدمي Tkinter
        if 'streamlit' in sys.modules or os.environ.get('STREAMLIT_SERVER_ENABLED') == '1':
            return None
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk();
            root.withdraw();
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Select review file (Excel, CSV, or JSON)",
                filetypes=[("All supported", "*.xlsx *.xls *.csv *.json"),
                           ("Excel files", "*.xlsx *.xls"),
                           ("CSV files", "*.csv"),
                           ("JSON files", "*.json"),
                           ("All files", "*.*")]
            )
            root.destroy()
            return path if path and os.path.exists(path) else None
        except Exception:
            return None

    def _process_row_enhanced(self, row: pd.Series, column_mapping: Dict[str, Optional[str]],
                              stats: Dict[str, Any]) -> List[Dict[str, str]]:
        """Enhanced row processing with LLM-powered analysis"""

        # Extract and validate basic data
        base_data = self._extract_base_data(row, column_mapping)

        # Extract review content
        review_content = self._extract_review_content(row, column_mapping)

        # Check if review has meaningful content (minimal check, let LLM decide)
        if not self._has_minimal_content(review_content):
            return []

        # LLM-powered content processing (minimal preprocessing)
        cleaned_content = self._prepare_content_for_llm(review_content, stats)
        self.logger.info(f"[LLM CALL] liked='{cleaned_content['liked']}' | disliked='{cleaned_content['disliked']}'")

        if not cleaned_content['liked'] and not cleaned_content['disliked']:
            return []

        # LLM-powered topic extraction
        topics = self._extract_topics_with_llm(cleaned_content, stats)

        if not topics:
            return []

        # بعد ما تحصل على topics من الLLM
        normalized_topics = []
        for t in topics or []:
            lang_t = self.detect_language(t.get('text', ''))
            # اترك العبري كما هو، وحوّل أي لغة غير إنجليزي إلى إنجليزي
            if lang_t not in ('english', 'hebrew') and t.get('text'):
                t['text'] = self.translate_text(t['text'], 'en')
            normalized_topics.append(t)
        topics = normalized_topics

        # Create output rows
        return self._create_output_rows(base_data, topics)

    def _extract_base_data(self, row: pd.Series, column_mapping: Dict[str, Optional[str]]) -> Dict[str, str]:
        source_val = row.get('source', '')
        rating_col = column_mapping.get('rating', '')
        rating_raw = row.get(rating_col, '')

        name_col = column_mapping.get('name')
        raw_name = row.get(name_col, '') if name_col else ''
        name = self.normalize_data(raw_name, 'text').strip()
        if not name:
            name = self._gen_fallback_name(source_val)

        return {
            'name': name,
            'date': self.normalize_data(row.get(column_mapping.get('date', ''), ''), 'date'),
            'country': self.normalize_data(row.get(column_mapping.get('country', ''), ''), 'country'),
            'rating': self._normalize_rating_tripaware(rating_raw, source_val, rating_col),  # ← هنا التحويل
            'trip_type': self.normalize_data(row.get(column_mapping.get('trip_type', ''), ''), 'text'),
            'source': source_val,  # مفيد لو حابة تدفعيه لـ Airtable
        }

    def _extract_review_content(self, row: pd.Series, column_mapping: Dict[str, Optional[str]]) -> Dict[str, str]:
        """Extract review content from row"""
        return {
            'title': row.get(column_mapping.get('review_title', ''), ''),
            'liked': row.get(column_mapping.get('liked_text', ''), ''),
            'disliked': row.get(column_mapping.get('disliked_text', ''), ''),
            'general': row.get(column_mapping.get('general_feedback', ''), '')
        }

    def _has_minimal_content(self, content: Dict[str, str]) -> bool:
        """Minimal content check - let LLM handle quality assessment"""
        all_texts = list(content.values())

        for text in all_texts:
            if text and str(text).strip() not in ['nan', 'None', '']:
                clean_text = str(text).strip()
                if len(clean_text) >= self.MIN_TEXT_LENGTH:
                    return True
        return False

    def _prepare_content_for_llm(self, content: Dict[str, str], stats: Dict[str, Any]) -> Dict[str, str]:
        """Minimal preprocessing - let LLM handle the heavy lifting"""
        # Simple title merging if meaningful
        liked_text = self._merge_title_if_meaningful(content['title'], content['liked'])
        disliked_text = self._merge_title_if_meaningful(content['title'], content['disliked'])

        # Handle general feedback with minimal processing
        if content['general'] and not liked_text and not disliked_text:
            liked_text = content['general']  # Let LLM split positive/negative
            disliked_text = content['general']

        # Basic cleanup only - remove obvious non-content
        cleaned_liked = self._basic_cleanup(liked_text)
        cleaned_disliked = self._basic_cleanup(disliked_text)

        # Track language detection
        if cleaned_liked:
            lang = self.detect_language(cleaned_liked)
            stats['languages_detected'].add(lang)
        if cleaned_disliked:
            lang = self.detect_language(cleaned_disliked)
            stats['languages_detected'].add(lang)

        return {'liked': cleaned_liked, 'disliked': cleaned_disliked}

    def _basic_cleanup(self, text: str) -> str:
        """Basic cleanup only - let LLM handle intelligent cleaning"""
        if not text:
            return ""

        text = str(text).strip()

        # Only remove obvious non-content
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'^[,.\s!?]+|[,.\s!?]+$', '', text)

        return text.strip()

    def _extract_topics_with_llm(self, content: Dict[str, str], stats: Dict[str, Any]) -> List[Dict[str, str]]:
        """LLM-powered topic extraction with single API call per row"""

        # Combine all content from the row into a single text for one API call
        combined_content = []

        if content['liked']:
            combined_content.append(f"POSITIVE: {content['liked']}")

        if content['disliked']:
            combined_content.append(f"NEGATIVE: {content['disliked']}")

        # If no content, return empty
        if not combined_content:
            return []

        # Join content with separator for the LLM to process
        full_text = " | ".join(combined_content)

        self.logger.info(f"[LLM CALL] combined='{full_text}'")

        # Check cache for the combined content
        content_hash = self.get_text_hash(full_text)
        if content_hash in self.topic_cache:
            topics = self.topic_cache[content_hash]
            stats['cache_hits'] += 1
            self.logger.info(f"[CACHE HIT] Found {len(topics) if topics else 0} topics in cache")
        else:
            # Single API call for the entire row
            topics = self._parse_with_claude_api_enhanced(full_text, 'mixed')
            stats['api_calls'] += 1
            if topics:
                self.topic_cache[content_hash] = topics
                self.logger.info(f"[API CALL] Extracted {len(topics)} topics from combined content")

        if not topics:
            return []

        # Minimal deduplication - trust LLM quality
        return self._minimal_deduplication(topics)

    def _minimal_deduplication(self, topics: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Minimal deduplication - trust LLM to provide quality topics"""
        if not topics:
            return []

        # Only remove exact duplicates
        seen = set()
        unique_topics = []

        for topic in topics:
            signature = f"{topic['topic']}:{topic['text'].lower()}:{topic['sentiment']}"
            if signature not in seen:
                seen.add(signature)
                unique_topics.append(topic)

        return unique_topics

    def _create_output_rows(self, base_data: Dict[str, str], topics: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create output rows from processed topics"""
        output_rows = []

        for topic in topics:
            if topic['sentiment'] == 'positive':
                positive_text = topic['text']
                negative_text = ''
            elif topic['sentiment'] == 'negative':
                positive_text = ''
                negative_text = topic['text']
            else:
                continue  # Skip neutral topics

            output_row = {
                'Name': base_data['name'],
                'Date': base_data['date'],
                'Rating': base_data['rating'],
                'Positive_text': positive_text,
                'Negative_text': negative_text,
                'Country': base_data['country'],
                'Type_travel': base_data['trip_type'],
                'Source': base_data.get('source', '')  # ← جديد
            }

            output_rows.append(output_row)

        return output_rows

    def _show_progress_stats(self, current: int, total: int, stats: Dict[str, Any], skipped: int) -> None:
        """Show enhanced progress statistics"""
        progress = (current / total) * 100
        self.logger.info(f"\n📈 Progress: {current}/{total} ({progress:.1f}%)")
        self.logger.info(f"✅ Topics extracted: {stats['total_topics']}")
        self.logger.info(f"➕ Positive: {stats['positive_topics']} | ➖ Negative: {stats['negative_topics']}")
        self.logger.info(f"🌍 Languages: {', '.join(stats['languages_detected'])}")
        self.logger.info(f"🤖 Claude API calls: {stats['api_calls']} | 💾 Cache hits: {stats['cache_hits']}")
        self.logger.info(f"⏭️ Skipped rows: {skipped}")

    def _finalize_enhanced_analysis(
            self,
            processed_data: List[Dict[str, str]],
            file_path: str,
            stats: Dict[str, Any],
            skipped_rows: int,
            total_rows: int,
            airtable_api_key: str | None = None,
            airtable_base_id: str | None = None,
            airtable_table: str | None = None,
    ) -> Optional[str]:

        if not processed_data:
            self.logger.error("\n❌ No actionable topics found for processing!")
            return None

        # خذي من باراميترات الدالة → وإلا من overrides → وإلا من .env
        api_key = airtable_api_key or self.airtable_overrides.get("api_key") or AIRTABLE_API_KEY
        base_id = airtable_base_id or self.airtable_overrides.get("base_id") or AIRTABLE_BASE_ID
        table = airtable_table or self.airtable_overrides.get("table") or AIRTABLE_RESULTS_TABLE

        ok = self.push_results_to_airtable(
            processed_data,
            table=table,
            api_key=api_key,
            base_id=base_id,
        )
        if not ok:
            self.logger.error("❌ Failed to push results to Airtable.")
            return None

        self._show_final_report("(Airtable)", processed_data, stats, skipped_rows, total_rows)
        return "airtable://pushed"

    def push_results_to_airtable(
            self,
            processed_data: List[Dict[str, str]],
            table: str | None = None,
            api_key: str | None = None,
            base_id: str | None = None,
    ) -> bool:
        """Send final rows to Airtable."""
        if not processed_data:
            self.logger.error("No processed data to push.")
            return False

        # استخدمي قيم الواجهة إن وصلت، وإلا القيَم من .env
        api_key = api_key or AIRTABLE_API_KEY
        base_id = base_id or AIRTABLE_BASE_ID
        table = table or AIRTABLE_RESULTS_TABLE

        if not api_key or not base_id:
            self.logger.error("Missing Airtable API key/base id.")
            return False

        mapped_rows = []
        for row in processed_data:
            mapped_rows.append({
                "Name": row.get("Name", ""),
                "Date": row.get("Date", ""),
                "Rating": row.get("Rating", ""),
                "Positive_text": row.get("Positive_text", ""),
                "Negative_text": row.get("Negative_text", ""),
                "Country": row.get("Country", ""),
                "Type_travel": row.get("Type_travel", ""),
                "Source": row.get("Source", "")
            })

        client = AirtableClient(api_key, base_id)
        client.create_batch(table, mapped_rows)
        self.logger.info("✅ Pushed %d rows to Airtable table '%s'", len(mapped_rows), table)
        return True

    def _show_final_report(self, excel_path: str, processed_data: List[Dict[str, str]],
                           stats: Dict[str, Any], skipped_rows: int, total_rows: int) -> None:
        """Show comprehensive final report"""

        self.logger.info(f"\n" + "=" * 80)
        self.logger.info("🎉 LLM-POWERED INTELLIGENT OPERATIONAL ANALYSIS COMPLETED!")
        self.logger.info("=" * 80)
        self.logger.info(f"📁 Enhanced output file: {excel_path}")
        self.logger.info(f"📊 Total actionable topics extracted: {len(processed_data)}")
        self.logger.info(f"👥 Reviews processed: {total_rows - skipped_rows}/{total_rows}")
        self.logger.info(f"⏭️ Rows skipped: {skipped_rows}")

        self.logger.info(f"\n📈 SENTIMENT DISTRIBUTION:")
        if len(processed_data) > 0:
            self.logger.info(
                f"➕ Positive topics: {stats['positive_topics']} ({stats['positive_topics'] / len(processed_data) * 100:.1f}%)")
            self.logger.info(
                f"➖ Negative topics: {stats['negative_topics']} ({stats['negative_topics'] / len(processed_data) * 100:.1f}%)")

        self.logger.info(f"\n🌍 LANGUAGE ANALYSIS:")
        self.logger.info(f"Languages detected: {', '.join(sorted(stats['languages_detected']))}")

        self.logger.info(f"\n🤖 CLAUDE AI EFFICIENCY:")
        total_calls = stats['api_calls'] + stats['cache_hits']
        if total_calls > 0:
            cache_efficiency = (stats['cache_hits'] / total_calls) * 100
            self.logger.info(f"Claude API calls made: {stats['api_calls']}")
            self.logger.info(f"Cache hits: {stats['cache_hits']}")
            self.logger.info(f"Cache efficiency: {cache_efficiency:.1f}%")
            self.logger.info(f"Model used: {self.CLAUDE_MODEL}")

        self.logger.info(f"\n✨ LLM-POWERED ENHANCEMENT FEATURES:")
        self.logger.info("• 🧠 Intelligent context-aware cleaning by LLM")
        self.logger.info("• 🎯 Focus on actionable operational feedback")
        self.logger.info("• 📚 Few-shot learning with quality examples")
        self.logger.info("• 🌐 Multi-language support with preservation")
        self.logger.info("• 💾 Intelligent caching for performance")
        self.logger.info("• 📊 Comprehensive statistics and reporting")
        self.logger.info("• 🔍 Advanced column detection")
        self.logger.info("• ⚡ Retry mechanisms for reliability")
        self.logger.info("• 🤖 Claude 3.5 Sonnet powered intelligence")

        # Show sample if available
        if processed_data:
            self.logger.info(f"\n📋 SAMPLE RESULT:")
            sample = processed_data[0]
            for key, value in sample.items():
                display_value = str(value)[:60] + "..." if len(str(value)) > 60 else value
                self.logger.info(f"   {key}: {display_value}")

    # Legacy method support
    def parse_with_llm(self, text: str, sentiment_hint: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """Legacy support - calls enhanced Claude version"""
        return self._parse_with_claude_api_enhanced(text, sentiment_hint)

    def read_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Legacy support - calls enhanced version"""
        return self.read_file_enhanced(file_path)

    def smart_column_mapping(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Legacy support - calls enhanced version"""
        return self.smart_column_mapping_enhanced(df)

    def run_analysis(self) -> Optional[str]:
        """Legacy support - calls enhanced version"""
        return self.run_enhanced_analysis()

    # Keep normalization and utility methods
    def normalize_data(self, value: Any, data_type: str) -> str:
        if pd.isna(value) or str(value).strip() in ['nan', 'None', '']:
            return ""

        if data_type == 'date':
            return self._normalize_date(value)
        elif data_type == 'rating':
            return self._normalize_rating(value)
        elif data_type == 'country':
            return self._normalize_country(value)
        else:
            return str(value).strip()

    def _normalize_date(self, date_value: Any) -> str:
        if pd.isna(date_value) or not str(date_value).strip():
            return ""

        s = str(date_value).strip()

        # ISO مع وقت
        if 'T' in s and ('Z' in s or '+' in s or s.count(':') >= 2):
            try:
                date_part = s.split('T')[0]
                dt = datetime.strptime(date_part, '%Y-%m-%d')
                return dt.strftime('%m/%d/%Y')
            except ValueError:
                pass

        # أنماط (سنة-شهر) بدون يوم → استخدم اليوم الأول
        month_only_patterns = [
            ('%Y-%m', True),  # 2025-07
            ('%m/%Y', True),  # 07/2025
            ('%b %Y', True),  # Jul 2025
            ('%B %Y', True),  # July 2025
            ('%Y.%m', True),  # 2025.07
        ]
        for fmt, month_only in month_only_patterns:
            try:
                dt = datetime.strptime(s, fmt)
                # اليوم الأول
                return dt.replace(day=1).strftime('%m/%d/%Y')
            except ValueError:
                pass

        # الأنماط الكاملة المعتادة
        full_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S', '%d-%m-%Y', '%Y.%m.%d'
        ]
        for fmt in full_formats:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.strftime('%m/%d/%Y')
            except ValueError:
                continue

        return ""

        # إذا ما قدر يفهم الصيغة، رجّع فاضي
        return ""

    def _normalize_rating(self, rating: Any) -> Optional[int]:
        if pd.isna(rating):
            return None
        try:
            r = float(str(rating).replace(",", "."))

            if 0 <= r <= 10:
                val = r
            else:
                return None
            return int(round(val))
        except (ValueError, TypeError):
            return None

            return str(int(round(val)))
        except (ValueError, TypeError):
            return ""

    # للاستبدال في الكود الأصلي، ضع هذه الدالة مكان الدالة الموجودة:

    def _normalize_rating_tripaware(self, value: Any, source: str, rating_col_name: Optional[str] = None) -> Optional[
        int]:
        """
        تطبيع التقييم حسب المصدر:
        - Booking.com: يبقى كما هو (1-10)
        - TripAdvisor: يضرب في 2 (1-5 → 2-10)
        """
        if pd.isna(value) or str(value).strip() == "":
            return None

        try:
            raw = float(str(value).replace(",", "."))
        except (ValueError, TypeError):
            return None

        src = (source or "").strip().lower()
        col = (rating_col_name or "").strip().lower()

        # تحديد إذا كان المصدر TripAdvisor
        is_tripadvisor = (
                src == 'tripadvisor' or
                'bubble' in col or
                'bubblerating' in col or
                'tripadvisor' in src
        )

        # المنطق الصحيح:
        if is_tripadvisor:
            # TripAdvisor: التقييم من 1-5، نضربه في 2 ليصبح 2-10
            if 1 <= raw <= 5:
                normalized = raw * 2.0
            else:
                return None  # تقييم خارج النطاق المتوقع
        else:
            # Booking.com وغيره: التقييم من 1-10، يبقى كما هو
            if 1 <= raw <= 10:
                normalized = raw
            else:
                return None  # تقييم خارج النطاق المتوقع

        return int(round(normalized))
    def _clean_country_str(self, s: str) -> str:
        s = str(s).strip()
        # مسافات وحدّة
        s = re.sub(r'\s+', ' ', s)
        # اسم بلد أو مدينة/بلد: اسمح بحروف + فراغ + فاصلة + شرطة + أبستروف
        s = re.sub(r"[^A-Za-z\u0590-\u05FF\u0600-\u06FF ,'\-]", "", s).strip()
        # قصّي القيم الطويلة جدًا
        if len(s) > 60:
            s = s[:60].strip()
        return s

    def _normalize_country(self, val: Any) -> str:
        # NaN / None
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""

        # dict: حاول خُذ اسم بلد
        if isinstance(val, dict):
            for k in ['country', 'countryName', 'name', 'nation']:
                v = val.get(k)
                if isinstance(v, str) and v.strip():
                    return self._clean_country_str(v)
            return ""

        # list: جرّب أول عنصر
        if isinstance(val, list):
            return self._normalize_country(val[0]) if val else ""

        s = str(val).strip()
        if not s:
            return ""

        # قيم لازم نرميها: URLs, UUID/IDs, أرقام صِرفة، placeholders
        if re.search(r'https?://|www\.', s, re.I):
            return ""
        if re.fullmatch(r'[a-f0-9\-]{8,}', s, re.I):  # uuid-ish
            return ""
        if re.fullmatch(r'\d{3,}', s):  # أرقام طويلة = ID
            return ""
        if s.lower() in {"unknown", "null", "none", "nan"}:
            return ""

        # لازم يحتوي على أحرف (لاتيني/عبري/عربي)
        if not re.search(r'[A-Za-z\u0590-\u05FF\u0600-\u06FF]', s):
            return ""

        return self._clean_country_str(s)

    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """Translate text to target language - simplified since LLM handles most translation needs"""
        if not text or len(str(text).strip()) < 3:
            return str(text)

        text = str(text).strip()

        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': 'auto',
                'tl': target_lang,
                'dt': 't',
                'q': text
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                result = response.json()
                translated_text = ""

                if result and len(result) > 0 and result[0]:
                    for item in result[0]:
                        if item and len(item) > 0:
                            translated_text += item[0]

                if translated_text.strip() and translated_text.strip() != text.strip():
                    return translated_text.strip()

            return text

        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return text

    def _merge_title_if_meaningful(self, title: str, text: str) -> str:
        """Merge title with text if complementary - minimal logic since LLM handles context"""
        if not title or not text:
            return text or ""

        title = str(title).strip()
        text = str(text).strip()

        # Simple merge if title is substantial and not already contained
        if len(title) > 5 and title.lower() not in text.lower():
            return f"{title}. {text}"

        return text

    def _get_language_instruction(self, detected_lang: str) -> str:
        """Get language-specific instruction for the prompt"""
        if detected_lang == 'hebrew':
            return '• **PRESERVE HEBREW**: Keep all extracted text in HEBREW - do not translate'
        else:
            return '• **TRANSLATE TO ENGLISH**: Translate the extracted text to ENGLISH for consistency'

    def _log_api_error(self, status_code: int) -> None:
        """Log API error based on status code"""
        error_messages = {
            401: "🔑 Authentication problem - check Claude API key",
            429: "⏰ Rate limit exceeded - try again later",
            500: "🔧 Claude server error",
            503: "🔧 Claude service unavailable"
        }

        message = error_messages.get(status_code, f"Unknown error: {status_code}")
        self.logger.error(f"❌ Claude API error {status_code}: {message}")

    def __del__(self):
        """Enhanced cleanup method"""
        if hasattr(self, 'session') and self.session:
            self.session.close()


def main():
    """Enhanced main function with Apify integration option"""
    try:
        print("🕷️ APIFY + CLAUDE REVIEW ANALYZER")
        print("=" * 50)
        print("🔧 Checking environment...")

        # Check required libraries
        required_libs = ['pandas', 'openpyxl', 'requests']
        optional_libs = ['langdetect', 'spacy']

        missing_required = []
        missing_optional = []

        for lib in required_libs:
            try:
                __import__(lib)
            except ImportError:
                missing_required.append(lib)

        for lib in optional_libs:
            try:
                __import__(lib)
            except ImportError:
                missing_optional.append(lib)

        if missing_required:
            print(f"❌ Missing required libraries: {', '.join(missing_required)}")
            print(f"💡 Install with: pip install {' '.join(missing_required)}")
            input("Press Enter to exit...")
            return

        if missing_optional:
            print(f"⚠️ Missing optional libraries: {', '.join(missing_optional)}")
            print(f"💡 For enhanced features, install: pip install {' '.join(missing_optional)}")

        print("✅ Environment check completed")

        # Initialize analyzer
        analyzer = EnhancedFeedbackAnalyzer(api_key=ANTHROPIC_API_KEY, advanced_analysis=True)


        # Choose analysis mode
        print("\n🚀 CHOOSE ANALYSIS MODE:")
        print("1. 📁 Analyze local file (Excel/CSV/JSON)")
        print("2. 🕷️ Analyze Apify dataset (requires Dataset ID + Token)")

        try:
            choice = input("\nEnter choice (1-2): ").strip()
        except:
            choice = "1"

        if choice == "2":
            # Apify dataset analysis
            print("\n🕷️ APIFY DATASET ANALYSIS")
            print("=" * 30)

            dataset_id = input("📊 Enter Apify Dataset ID: ").strip()
            if not dataset_id:
                print("❌ Dataset ID is required")
                input("Press Enter to exit...")
                return

            apify_token = APIFY_TOKEN or os.getenv("APIFY_TOKEN")
            if not apify_token:
                print("❌ APIFY_TOKEN غير موجود. ضيفيه في ملف .env مثل:")
                print("APIFY_TOKEN=apify_api_XXXXXXXXXXXXXXXX")
                input("Press Enter to exit...")
                return

            print(f"\n🎯 Analyzing dataset: {dataset_id}")
            result = analyzer.analyze_apify_dataset(dataset_id, apify_token)

        else:
            # Local file analysis (your existing logic)
            result = analyzer.run_enhanced_analysis()

        # Show results
        if result:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📁 Excel output file: {result}")
            print("\n🚀 FEATURES USED:")
            print("• 🧠 Claude AI-powered topic extraction")
            print("• 🎯 Operational feedback focus")
            print("• 🌐 Multi-language support")
            print("• 📊 Excel output with statistics")
            if choice == "2":
                print("• 🕷️ Apify dataset integration")
        else:
            print("\n❌ Analysis failed")

    except KeyboardInterrupt:
        print("\n⚠️ Process cancelled by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")

    finally:
        input("\nPress Enter to exit...")


def quick_apify_analysis():
    """Quick function for Apify dataset analysis"""

    print("🕷️ QUICK APIFY ANALYSIS")
    print("=" * 25)

    dataset_id = input("Dataset ID: ").strip()
    apify_token = os.getenv("APIFY_TOKEN")

    if not dataset_id or not apify_token:
        print("❌ Both Dataset ID and Token are required")
        return

    analyzer = EnhancedFeedbackAnalyzer(api_key=ANTHROPIC_API_KEY)
    result = analyzer.analyze_apify_dataset(dataset_id, apify_token)

    if result:
        print(f"✅ Excel file created: {result}")
    else:
        print("❌ Analysis failed")

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()              # شغّال فقط لو شغّلتي الملف مباشرة من الطرفية
    # quick_apify_analysis()  # شغليه يدويًا إذا بدك، بس لا تتركيه يشتغل تلقائيًا
