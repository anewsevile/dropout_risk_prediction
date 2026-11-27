import os
import textwrap
import io
import re
import csv
import json
import pickle
import chardet
import numpy as np
import pandas as pd
import streamlit as st
def show_html(s):
    s = "\n".join(line.strip() for line in str(s).splitlines() if line.strip())
    st.markdown(s, unsafe_allow_html=True)

def _html_patch(body, **kwargs):
    body = "\n".join(line.strip() for line in str(body).splitlines() if line.strip())
    st.markdown(body, unsafe_allow_html=True)
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from fpdf import FPDF

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="EARLY DROPOUT RISK PREDICTION",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# RETRO OS & POSTER NEUBRUTALISM CSS
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Space+Grotesk:wght@500;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap');

    /* Reset Streamlit Defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible !important; background-color: transparent !important;}
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 95% !important;
    }

    /* Outermost Page Background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewRootContainer"] {
        background-color: #B9D6D3 !important;
        background-image: 
            linear-gradient(rgba(26, 26, 26, 0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(26, 26, 26, 0.06) 1px, transparent 1px) !important;
        background-size: 24px 24px !important;
        color: #1A1A1A !important;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFF0 !important;
        border-right: 2.5px solid #1A1A1A !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background-color: #FFFFFF;
        border: 2px solid #1A1A1A;
        border-radius: 4px !important;
        padding: 8px 14px !important;
        margin-bottom: 6px !important;
        box-shadow: 2px 2px 0px #1A1A1A;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        color: #1A1A1A !important;
        transition: all 0.1s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #FFF3C4;
        transform: translate(-1px, -1px);
        box-shadow: 3px 3px 0px #1A1A1A;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background-color: #1A1A1A !important;
        color: #FFFFF0 !important;
        box-shadow: 3px 3px 0px #DE6B35 !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] p,
    [data-testid="stSidebar"] .stRadio label:has(input:checked) p {
        color: #FFFFF0 !important;
        font-weight: 700 !important;
    }
    .sidebar-sys-box {
        background-color: #1A1A1A;
        color: #FFFFF0;
        border: 2px solid #1A1A1A;
        border-radius: 4px;
        padding: 12px;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        box-shadow: 3px 3px 0px #DE6B35;
        margin-top: 15px;
        line-height: 1.6;
    }

    /* Brutalist Headings */
    h1, h2, h3, h4, .brutalist-title {
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        color: #1A1A1A !important;
    }

    /* Retro OS Window Titlebar */
    .window-titlebar {
        background-color: #1A1A1A;
        color: #FFFFF0;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 2.5px solid #1A1A1A;
        margin: -1rem -1rem 1.25rem -1rem;
        box-sizing: border-box;
    }
    .window-dots {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        border: 1.5px solid #1A1A1A;
    }
    .dot-red { background-color: #FF5F56; }
    .dot-yellow { background-color: #FFBD2E; }
    .dot-green { background-color: #27C93F; }

    .window-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #FFFFF0;
        letter-spacing: 0.5px;
        margin-left: 10px;
        flex-grow: 1;
        text-transform: lowercase;
    }
    .window-status {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: #27C93F;
        background-color: #262626;
        border: 1px solid #404040;
        padding: 2px 7px;
        border-radius: 2px;
        text-transform: uppercase;
    }

    /* Streamlit Container (Card) Styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
        background-image: none !important;
        box-shadow: 4px 4px 0px #1A1A1A !important;
        overflow: hidden !important;
        margin-bottom: 1.25rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        background-image: none !important;
    }

    /* Neubrutalist Stat Cards */
    .stat-card {
        border: 2.5px solid #1A1A1A;
        border-radius: 4px !important;
        padding: 18px 20px;
        text-align: left;
        box-shadow: 4px 4px 0px #1A1A1A;
        margin-bottom: 14px;
        transition: transform 0.1s ease;
        background-image: none !important;
    }
    .stat-card:hover {
        transform: translate(-1px, -1px);
        box-shadow: 5px 5px 0px #1A1A1A;
    }
    .stat-card-white {
        background-color: #FFFFFF !important;
        background-image: none !important;
        color: #1A1A1A;
    }
    .stat-card-white .stat-label { color: #555555; }
    .stat-card-white .stat-value { color: #1A1A1A; }
    
    .stat-card-orange {
        background-color: #DE6B35 !important;
        background-image: none !important;
        color: #FFFFFF;
    }
    .stat-card-orange .stat-label { color: #FFF0E6; }
    .stat-card-orange .stat-value { color: #FFFFFF; }
    .stat-card-orange p { color: #FFF0E6 !important; }
    
    .stat-card-yellow {
        background-color: #E8C547 !important;
        background-image: none !important;
        color: #1A1A1A;
    }
    .stat-card-yellow .stat-label { color: #222222; }
    .stat-card-yellow .stat-value { color: #1A1A1A; }
    
    .stat-card-dark {
        background-color: #1A1A1A !important;
        background-image: none !important;
        color: #FFFFF0;
    }
    .stat-card-dark .stat-label { color: #CCCCCC; }
    .stat-card-dark .stat-value { color: #FFFFF0; }
    .stat-card-dark p { color: #E0E0E0 !important; }

    .stat-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 0.95;
        letter-spacing: -1.5px;
    }
    .academic-metric-card {
        height: 120px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
    }
    .academic-metric-card .stat-label {
        flex-shrink: 0;
        margin-bottom: 4px;
    }
    .academic-metric-card .stat-value {
        overflow-wrap: anywhere;
    }
    .academic-category-long-value .stat-value {
        display: -webkit-box;
        overflow: hidden;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        font-size: clamp(1rem, 1.4vw, 1.5rem);
        line-height: 1.05;
        letter-spacing: -0.5px;
    }

    /* Badges */
    .badge-high {
        background-color: #1A1A1A;
        color: #FFFFFF;
        padding: 5px 14px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 999px !important;
        border: 2px solid #1A1A1A;
        box-shadow: 2px 2px 0px #1A1A1A;
    }
    .badge-medium {
        background-color: #E8C547;
        color: #1A1A1A;
        padding: 5px 14px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 999px !important;
        border: 2px solid #1A1A1A;
        box-shadow: 2px 2px 0px #1A1A1A;
    }
    .badge-low {
        background-color: #FFFFFF;
        color: #1A1A1A;
        padding: 5px 14px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 999px !important;
        border: 2px solid #1A1A1A;
        box-shadow: 2px 2px 0px #1A1A1A;
    }
    .badge-invalid {
        background-color: #FF5F56;
        color: #FFFFFF;
        padding: 3px 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.72rem;
        display: inline-block;
        border-radius: 3px;
        border: 1.5px solid #1A1A1A;
    }

    /* Torn Paper */
    .torn-paper-edge {
        height: 8px;
        background: linear-gradient(135deg, #1A1A1A 25%, transparent 25%) -8px 0,
                    linear-gradient(225deg, #1A1A1A 25%, transparent 25%) -8px 0,
                    linear-gradient(315deg, #1A1A1A 25%, transparent 25%),
                    linear-gradient(45deg, #1A1A1A 25%, transparent 25%);
        background-size: 16px 8px;
        background-repeat: repeat-x;
        margin-top: 18px;
        opacity: 0.75;
    }

    .brutalist-divider {
        border-top: 2.5px solid #1A1A1A;
        margin: 20px 0;
    }

    /* Fixed Neubrutalist Top-Right Upload Button (re-styled to match reference INSERT DATA button) */
    div[data-testid="stPopover"] > button {
        background-color: #D1443D !important;
        border: 3.5px solid #000000 !important;
        border-radius: 6px !important;
        color: #000000 !important;
        font-family: 'Oswald', 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 4px 4px 0px #000000 !important;
        padding: 6px 14px 6px 14px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 3px !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }
    div[data-testid="stPopover"] > button::before {
        content: "" !important;
        display: block !important;
        width: 22px !important;
        height: 25px !important;
        margin: 0 auto !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: contain !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 32' fill='none'%3E%3Cpath d='M4 2 H19 L25 8 V29 H4 Z' fill='%23D1443D' stroke='%23000000' stroke-width='2.6' stroke-linejoin='round'/%3E%3Cpath d='M19 2 V8 H25' fill='none' stroke='%23000000' stroke-width='2.6' stroke-linejoin='round'/%3E%3Cline x1='14.5' y1='9' x2='14.5' y2='17' stroke='%23000000' stroke-width='2.5' stroke-linecap='round'/%3E%3Cpolyline points='10.5,14 14.5,18 18.5,14' fill='none' stroke='%23000000' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Crect x='6.5' y='20.5' width='16' height='7' rx='1' fill='none' stroke='%23000000' stroke-width='1.8'/%3E%3Ctext x='14.5' y='26' fill='%23000000' font-family='sans-serif' font-weight='900' font-size='5.5' text-anchor='middle' letter-spacing='0.5'%3ECSV%3C/text%3E%3C/svg%3E") !important;
    }
    div[data-testid="stPopover"] > button p {
        color: #000000 !important;
        font-family: 'Oswald', 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        margin: 0 !important;
        line-height: 1.1 !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background-color: #DE534C !important;
        transform: translate(-1px, -1px) !important;
        box-shadow: 5px 5px 0px #000000 !important;
    }
    div[data-testid="stPopover"] > button:active {
        transform: translate(1px, 1px) !important;
        box-shadow: 2px 2px 0px #000000 !important;
    }
    div[data-testid="stPopoverBody"] {
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        box-shadow: 6px 6px 0px #1A1A1A !important;
        background-color: #FFFFF0 !important;
        padding: 16px !important;
    }
    
    [data-testid="stFileUploader"] {
        border: 2px dashed #1A1A1A !important;
        border-radius: 4px !important;
        padding: 10px !important;
        background-color: #FFFFFF !important;
    }

    /* Disclaimer Tag */
    .disclaimer-badge {
        background-color: #FFFFF0;
        border: 2px solid #1A1A1A;
        border-radius: 4px;
        padding: 6px 14px;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #1A1A1A;
        box-shadow: 2px 2px 0px #1A1A1A;
        display: inline-block;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# MODEL ARTIFACT LOADING
# =====================================================================
def find_file(possible_names):
    for name in possible_names:
        if os.path.exists(name):
            return name
    return None

@st.cache_resource
def load_artifacts():
    filename = find_file([
        "dropout_prediction_artifacts.pkl",
        "dropout_prediction_artifacts (1).pkl",
        "artifacts.pkl"
    ])
    if not filename:
        raise FileNotFoundError("Could not find dropout_prediction_artifacts.pkl in working directory.")
    
    with open(filename, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

try:
    artifacts = load_artifacts()
    model_obj = artifacts['model']
    scaler = artifacts['scaler']
    ordinal_encoder = artifacts['ordinal_encoder']
    dept_encoder = artifacts['dept_encoder']
    category_encoder = artifacts['category_encoder']
    target_encoder = artifacts['target_encoder']
    eval_metrics = artifacts['eval_metrics']
except Exception as e:
    st.error(f"CRITICAL SYSTEM ERROR: Failed to load model artifacts: {str(e)}")
    st.stop()


# =====================================================================
# SESSION STATE INITIALIZATION (SESSION MEMORY ONLY)
# =====================================================================
if 'raw_uploaded_df' not in st.session_state:
    st.session_state['raw_uploaded_df'] = None
if 'uploaded_filename' not in st.session_state:
    st.session_state['uploaded_filename'] = None
if 'column_mapping' not in st.session_state:
    st.session_state['column_mapping'] = {}
if 'column_status' not in st.session_state:
    st.session_state['column_status'] = {}
if 'upload_error' not in st.session_state:
    st.session_state['upload_error'] = None
if 'consent_excluded_count' not in st.session_state:
    st.session_state['consent_excluded_count'] = 0
if 'global_class_filter' not in st.session_state:
    st.session_state['global_class_filter'] = "All classes"
if 'scoring_stats' not in st.session_state:
    st.session_state['scoring_stats'] = {}


# =====================================================================
# HEADER ROW DETECTION & UNIVERSAL FILE READER
# =====================================================================
EXPECTED_COLUMNS = [
    'Name', 'Roll Number', 'Department', 'Year', 'Semester',
    'Attendance %', 'CGPA', 'Backlogs', 'Student Category', 'Consent'
]

SYNONYMS = {
    'Name': ['name', 'student name', 'candidate name', 'full name', 'candidate'],
    'Roll Number': ['roll number', 'roll no', 'rollno', 'roll', 'student id', 'id', 'reg no', 'registration no', 'reg number', 'registration number', 'prn', 'enrollment no', 'enrollment number', 'enrollment'],
    'Department': ['degree / programme', 'degree/programme', 'programme', 'degree', 'department', 'dept', 'branch', 'stream', 'program', 'course', 'major', 'discipline'],
    'Year': ['current year of study', 'year of study', 'academic year', 'study year', 'current year', 'year', 'yr'],
    'Semester': ['current semester', 'semester', 'sem', 'term'],
    'Attendance %': ['current attendance percentage', 'attendance percentage', 'attendance percent', 'attendance %', 'attendance', 'attd', 'present', 'lecture attendance', 'overall attendance', 'attendance rate'],
    'CGPA': ['current cgpa / latest semester percentage', 'latest semester percentage', 'current cgpa', 'cgpa / latest semester percentage', 'cgpa', 'gpa', 'sgpa', 'cumulative gpa', 'grade point average', 'pointer', 'marks avg', 'grade'],
    'Backlogs': ['backlogs', 'backlog', 'kt', 'atkt', 'active backlogs', 'number of backlogs', 'failed subjects', 'arrears', 'standing backlogs', 'kts', 'number of active backlogs'],
    'Student Category': ['student category', 'category', 'quota', 'admission category', 'student type', 'caste category', 'reservation category', 'category type'],
    'Consent': ['consent', 'student consent', 'data consent', 'agreed', 'consent given', 'approval', 'permission'],
    'Class': ['class', 'division', 'div', 'current class', 'section']
}

def normalize_header(raw_col):
    s = str(raw_col).strip()
    s = re.sub(r'^\s*[\(\[]?\d+[\.\)\-\]]?\s*', '', s)
    s = re.sub(r'\s*[\(\[].*?[\)\]]', '', s)
    s = re.sub(r'\s*[-–—].*$', '', s)
    s = re.sub(r'[%_#\-]', ' ', s)
    return ' '.join(s.lower().split())

def detect_header_row(df_raw):
    """
    Scans the first 10 rows and picks the row with the most text cells that look like headers
    (best match to expected-column synonyms). Ignores everything above it.
    Never shows 'Unnamed: n' anywhere. If a header is truly blank, names it 'Column N'.
    """
    if len(df_raw) == 0:
        return df_raw, 0
    max_scan = min(10, len(df_raw))
    best_row = 0
    best_score = -1

    all_syn_flat = set()
    for syn_list in SYNONYMS.values():
        for syn in syn_list:
            all_syn_flat.add(normalize_header(syn))

    for r in range(max_scan):
        row_vals = df_raw.iloc[r]
        score = 0
        text_count = 0
        for val in row_vals:
            if pd.isna(val):
                continue
            s_val = str(val).strip()
            if not s_val or s_val.lower().startswith('unnamed:'):
                continue
            text_count += 1
            norm = normalize_header(s_val)
            matched = False
            for syn in all_syn_flat:
                if syn == norm or syn in norm or norm in syn:
                    matched = True
                    break
            if 'backlog' in norm or norm.startswith('number of') or 'consent' in norm or 'category' in norm or 'attendance' in norm or 'cgpa' in norm:
                matched = True
            if matched:
                score += 15
            else:
                score += 1
        total_row_score = score + text_count
        if total_row_score > best_score:
            best_score = total_row_score
            best_row = r

    raw_headers = df_raw.iloc[best_row].tolist()
    cleaned_headers = []
    for idx, h in enumerate(raw_headers):
        if pd.isna(h) or not str(h).strip() or str(h).strip().lower().startswith('unnamed:'):
            cleaned_headers.append(f"Column {idx + 1}")
        else:
            cleaned_headers.append(str(h).strip())

    cleaned_df = df_raw.iloc[best_row + 1:].copy().reset_index(drop=True)
    cleaned_df.columns = cleaned_headers
    return cleaned_df, best_row


def read_tabular_file(uploaded_file):
    """
    Accepts .csv, .tsv, .txt, .xlsx, .xls, .ods, .json.
    Auto-detects format, encoding (UTF-8, UTF-8 with BOM, Latin-1) and delimiter.
    Scans first 10 rows for real headers and removes all 'Unnamed: n' labels.
    """
    raw_bytes = uploaded_file.getvalue()
    fname = uploaded_file.name.lower()

    # 1. Excel XLSX
    if fname.endswith(('.xlsx', '.xlsm')) or (raw_bytes[:4] == b'PK\x03\x04' and not fname.endswith(('.ods', '.zip'))):
        try:
            raw_df = pd.read_excel(io.BytesIO(raw_bytes), engine='openpyxl', header=None)
            df, _ = detect_header_row(raw_df)
            return df, None
        except Exception as e:
            return None, f"Could not parse Excel (.xlsx) file: {str(e)}"

    # 2. Excel XLS (legacy)
    if fname.endswith('.xls'):
        try:
            raw_df = pd.read_excel(io.BytesIO(raw_bytes), engine='xlrd', header=None)
            df, _ = detect_header_row(raw_df)
            return df, None
        except Exception as e:
            return None, f"Could not parse legacy Excel (.xls) file: {str(e)}"

    # 3. OpenDocument Spreadsheet ODS
    if fname.endswith('.ods'):
        try:
            raw_df = pd.read_excel(io.BytesIO(raw_bytes), engine='odf', header=None)
            df, _ = detect_header_row(raw_df)
            return df, None
        except Exception as e:
            return None, f"Could not parse OpenDocument Spreadsheet (.ods) file: {str(e)}"

    # 4. JSON
    if fname.endswith('.json') or raw_bytes.strip().startswith(b'{') or raw_bytes.strip().startswith(b'['):
        try:
            text = raw_bytes.decode('utf-8', errors='replace')
            data = json.loads(text)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                found_list = False
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        df = pd.DataFrame(v)
                        found_list = True
                        break
                if not found_list:
                    df = pd.DataFrame(data)
            else:
                return None, "JSON structure must be an object or list of records."
            
            cleaned_cols = []
            for idx, c in enumerate(df.columns):
                if pd.isna(c) or not str(c).strip() or str(c).strip().lower().startswith('unnamed:'):
                    cleaned_cols.append(f"Column {idx + 1}")
                else:
                    cleaned_cols.append(str(c).strip())
            df.columns = cleaned_cols
            return df, None
        except Exception as e:
            if fname.endswith('.json'):
                return None, f"Could not parse JSON content: {str(e)}"

    # 5. Text / CSV / TSV with encoding & delimiter auto-detection
    detected_encoding = None
    if raw_bytes.startswith(b'\xef\xbb\xbf'):
        detected_encoding = 'utf-8-sig'
    else:
        try:
            det = chardet.detect(raw_bytes[:25000])
            if det and det.get('confidence', 0) >= 0.7:
                detected_encoding = det.get('encoding')
        except Exception:
            pass

    encodings_to_try = [detected_encoding, 'utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    encodings_to_try = [e for e in encodings_to_try if e]

    text_content = None
    for enc in encodings_to_try:
        try:
            text_content = raw_bytes.decode(enc)
            break
        except Exception:
            continue

    if text_content is None:
        return None, "Could not decode text file: unsupported binary character encoding."

    sample = text_content[:10000]
    sep = ','
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',\t;|')
        sep = dialect.delimiter
    except Exception:
        lines = [l for l in sample.splitlines() if l.strip()][:5]
        counts = {',': 0, '\t': 0, ';': 0, '|': 0}
        for l in lines:
            for d in counts:
                counts[d] += l.count(d)
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            sep = best

    try:
        raw_df = pd.read_csv(io.StringIO(text_content), sep=sep, engine='python', header=None)
        df, _ = detect_header_row(raw_df)
        return df, None
    except Exception as e:
        return None, f"Could not parse tabular structure: {str(e)}"


# =====================================================================
# LOOSE HEADER MATCHING & COLUMN CHECK
# =====================================================================
def auto_match_columns(df_columns, df_sample=None):
    mapping = {}
    statuses = {}
    used_cols = set()
    available_cols = [c for c in df_columns if normalize_header(c) not in ['timestamp', 'time stamp']]

    for exp_col in EXPECTED_COLUMNS + ['Class']:
        syn_list = SYNONYMS.get(exp_col, [])
        matched = None
        for col in available_cols:
            if col in used_cols:
                continue
            norm = normalize_header(col)
            # Special regex/substring matches requested by user
            if exp_col == 'Backlogs' and ('backlog' in norm or norm.startswith('number of')):
                matched = col
                break
            elif exp_col == 'Consent' and 'consent' in norm:
                matched = col
                break
            elif exp_col == 'Student Category' and 'category' in norm:
                matched = col
                break
            elif norm == exp_col.lower() or norm in syn_list:
                matched = col
                break

        if not matched:
            for col in available_cols:
                if col in used_cols:
                    continue
                norm = normalize_header(col)
                for syn in syn_list:
                    if syn in norm or norm in syn:
                        matched = col
                        break
                if matched:
                    break

        if matched:
            mapping[exp_col] = matched
            statuses[exp_col] = "FOUND"
            used_cols.add(matched)
        else:
            mapping[exp_col] = None
            statuses[exp_col] = "MISSING"

    # Name content-based guess rule:
    # "and also 'Column 1' if its values look like names... A guess based on content may be shown only as yellow 'GUESSED, please confirm', never green FOUND."
    if mapping.get('Name') is None and df_sample is not None and "Column 1" in df_sample.columns:
        col1_vals = df_sample["Column 1"].dropna().astype(str).str.strip()
        if len(col1_vals) > 0:
            looks_like_names = col1_vals.apply(lambda s: bool(re.match(r'^[A-Za-z\s\.\'-]{3,50}$', s)) and not any(ch.isdigit() for ch in s))
            if looks_like_names.mean() >= 0.6:
                mapping['Name'] = "Column 1"
                statuses['Name'] = "GUESSED"

    return mapping, statuses


# =====================================================================
# DATA PROCESSING, CELL VALIDATION & MODEL INFERENCE
# =====================================================================
def clean_numeric_str(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s or s.lower() in ['-', 'not', 'n/a', 'na', 'null', 'none', 'nil']:
        return np.nan
    s = s.replace('%', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

def normalize_student_category(category):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(category).lower()).strip()
    if "intern" in normalized or "working" in normalized:
        return "Intern/Working"
    if "sport" in normalized and ("representative" in normalized or " rep" in f" {normalized}"):
        return "Sports Representative"
    return "Regular"

def is_attendance_exempt_category(category):
    return normalize_student_category(category) in {"Intern/Working", "Sports Representative"}

def calculate_student_risk(attendance_percent, cgpa, backlogs, student_category):
    """Return a transparent 0-100 risk index, exempting approved duty categories from attendance penalties."""
    attendance_exempt = is_attendance_exempt_category(student_category)
    factors = {
        "backlogs": min(max(float(backlogs), 0.0), 3.0) * 20.0,
        "cgpa": min(max(7.5 - float(cgpa), 0.0) * 6.0, 30.0),
        "attendance_percent": 0.0 if attendance_exempt else min(
            max(75.0 - float(attendance_percent), 0.0) * 0.75, 30.0
        ),
    }
    score = round(min(sum(factors.values()), 100.0), 1)
    risk_level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    return score, risk_level, factors, attendance_exempt

def process_uploaded_cohort(raw_df, col_mapping):
    """
    Applies column mapping, marks invalid cells, applies consent filter,
    extracts Class, cleans numeric cells (% signs stripped, blanks/text are INVALID, never 0),
    handles Backlogs (none/no/nil=0, plain number=number, 'More than ...'=unparsed),
    and excludes students with invalid Attendance, CGPA or Backlogs from risk scoring ONLY.
    """
    df = raw_df.copy()
    
    # 1. Apply Consent Filter if Consent column exists
    consent_col = col_mapping.get('Consent')
    if consent_col and consent_col in df.columns:
        consent_series = df[consent_col].astype(str).str.strip().str.lower()
        valid_consent = consent_series.isin(['yes', 'y', 'true', '1', 'agreed', 'given'])
        excluded_count = int((~valid_consent).sum())
        st.session_state['consent_excluded_count'] = excluded_count
        df = df[valid_consent].copy().reset_index(drop=True)
    else:
        st.session_state['consent_excluded_count'] = 0

    # 2. Extract and Validate Key Fields
    # Roll Number / ID
    roll_col = col_mapping.get('Roll Number')
    df['student_id'] = df[roll_col].astype(str).str.strip() if roll_col and roll_col in df.columns else [f"STU_{i+1:03d}" for i in range(len(df))]
    
    # Name
    name_col = col_mapping.get('Name')
    df['name'] = df[name_col].astype(str).str.strip() if name_col and name_col in df.columns else df['student_id']

    # Class (Extra field: e.g. TYDS, SYCS, TYIT)
    class_col = col_mapping.get('Class')
    if class_col and class_col in df.columns:
        df['class_name'] = df[class_col].astype(str).str.strip()
    else:
        df['class_name'] = "General"

    # Department
    dept_col = col_mapping.get('Department')
    df['department'] = df[dept_col].astype(str).str.strip() if dept_col and dept_col in df.columns else "General"

    # Year
    year_col = col_mapping.get('Year')
    if year_col and year_col in df.columns:
        y_num = pd.to_numeric(df[year_col], errors='coerce')
        df['year'] = np.where((y_num >= 1) & (y_num <= 6), y_num, np.nan)
    else:
        df['year'] = np.nan

    # Semester
    sem_col = col_mapping.get('Semester')
    df['semester'] = df[sem_col].astype(str).str.strip() if sem_col and sem_col in df.columns else "N/A"

    # Attendance % (bounds 0 to 100)
    att_col = col_mapping.get('Attendance %')
    if att_col and att_col in df.columns:
        att_num = df[att_col].apply(clean_numeric_str)
        df['attendance_percent'] = np.where((att_num >= 0.0) & (att_num <= 100.0), att_num, np.nan)
        df['attendance_invalid'] = df['attendance_percent'].isna()
    else:
        df['attendance_percent'] = np.nan
        df['attendance_invalid'] = True

    # CGPA (bounds 0 to 10; above 10 is INVALID, never converted!)
    cgpa_col = col_mapping.get('CGPA')
    if cgpa_col and cgpa_col in df.columns:
        cgpa_num = df[cgpa_col].apply(clean_numeric_str)
        df['cgpa'] = np.where((cgpa_num >= 0.0) & (cgpa_num <= 10.0), cgpa_num, np.nan)
        df['cgpa_invalid'] = df['cgpa'].isna()
    else:
        df['cgpa'] = np.nan
        df['cgpa_invalid'] = True

    # Backlogs (none/no/nil -> 0, plain number -> number, 'More than ...' -> unparsed text)
    bk_col = col_mapping.get('Backlogs')
    bk_numeric_list = []
    bk_disp_list = []
    bk_status_list = []
    if bk_col and bk_col in df.columns:
        for val in df[bk_col]:
            if pd.isna(val):
                bk_numeric_list.append(np.nan)
                bk_disp_list.append("-")
                bk_status_list.append("INVALID")
            else:
                s = str(val).strip()
                if not s or s.lower() in ['-', 'not', 'n/a', 'na', 'null']:
                    bk_numeric_list.append(np.nan)
                    bk_disp_list.append(s if s else "-")
                    bk_status_list.append("INVALID")
                elif s.lower() in ['none', 'no', 'nil', 'zero']:
                    bk_numeric_list.append(0.0)
                    bk_disp_list.append("0")
                    bk_status_list.append("VALID")
                else:
                    s_clean = s.replace('%', '').strip()
                    try:
                        f = float(s_clean)
                        if f >= 0:
                            bk_numeric_list.append(f)
                            bk_disp_list.append(str(int(f) if f.is_integer() else f))
                            bk_status_list.append("VALID")
                        else:
                            bk_numeric_list.append(np.nan)
                            bk_disp_list.append(s)
                            bk_status_list.append("INVALID")
                    except ValueError:
                        # e.g. "More than 3"
                        bk_numeric_list.append(np.nan)
                        bk_disp_list.append(s)
                        bk_status_list.append("UNPARSED")
    else:
        bk_numeric_list = [np.nan] * len(df)
        bk_disp_list = ["-"] * len(df)
        bk_status_list = ["INVALID"] * len(df)

    df['backlogs'] = bk_numeric_list
    df['backlogs_display'] = bk_disp_list
    df['backlogs_status'] = bk_status_list
    df['backlogs_invalid'] = df['backlogs'].isna()

    # Student Category
    cat_col = col_mapping.get('Student Category')
    if cat_col and cat_col in df.columns:
        df['student_category'] = df[cat_col].apply(normalize_student_category)
    else:
        df['student_category'] = "Regular"

    # 3. Check Risk Scoring Pre-conditions
    missing_risk_cols = []
    if not (att_col and att_col in df.columns):
        missing_risk_cols.append("Attendance %")
    if not (cgpa_col and cgpa_col in df.columns):
        missing_risk_cols.append("CGPA")
    if not (bk_col and bk_col in df.columns):
        missing_risk_cols.append("Backlogs")

    # Score only rows with valid academic predictors.
    valid_row_mask = df['attendance_percent'].notna() & df['cgpa'].notna() & df['backlogs'].notna()
    df['risk_score'] = np.nan
    df['predicted_risk'] = "UNSCORED"

    # Record cohort scoring stats in session state
    st.session_state['scoring_stats'] = {
        'total_students': len(df),
        'scored_count': int(valid_row_mask.sum()),
        'excluded_count': int((~valid_row_mask).sum()),
        'invalid_att_count': int(df['attendance_invalid'].sum()),
        'invalid_cgpa_count': int(df['cgpa_invalid'].sum()),
        'unparsed_bk_count': int((df['backlogs_status'] == 'UNPARSED').sum()),
        'invalid_bk_count': int((df['backlogs_status'] == 'INVALID').sum())
    }

    if missing_risk_cols:
        risk_scoring_available = False
        risk_status_msg = f"Risk scoring unavailable, missing: {', '.join(missing_risk_cols)}"
    else:
        risk_scoring_available = True
        risk_status_msg = "RULE-BASED METRICS SCORE"

        # Score every row with valid academic metrics using the same transparent rules.
        if valid_row_mask.sum() > 0:
            valid_sub = df[valid_row_mask].copy()
            results = [
                calculate_student_risk(
                    row['attendance_percent'],
                    row['cgpa'],
                    row['backlogs'],
                    row['student_category'],
                )
                for _, row in valid_sub.iterrows()
            ]
            df.loc[valid_row_mask, 'risk_score'] = [result[0] for result in results]
            df.loc[valid_row_mask, 'predicted_risk'] = [result[1] for result in results]

    return df, risk_scoring_available, risk_status_msg


# =====================================================================
# UI HELPERS & EMPTY STATE (MATCHING USER SCREENSHOT)
# =====================================================================
def render_window_header(filename: str, status: str = "ONLINE"):
    show_html(textwrap.dedent(f"""\
<div class="window-titlebar">
<div class="window-dots">
<span class="dot dot-red"></span>
<span class="dot dot-yellow"></span>
<span class="dot dot-green"></span>
</div>
<div class="window-title">{filename}</div>
<div class="window-status">{status}</div>
</div>
""").strip())

def render_top_upload_button(key_suffix: str = ""):
    """
    Renders fixed neubrutalist upload button matching the attached screenshot:
    [ ⇪ CSV UPLOAD ] in the top-right of the page.
    """
    with st.popover("CSV UPLOAD", use_container_width=False):
        show_html(textwrap.dedent("""\
<div style="font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #DE6B35; margin-bottom: 2px;">SYS.IO // INGEST</div>
<h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; margin: 0 0 10px 0;">UPLOAD DATASET</h4>
<p style="font-size: 0.75rem; font-family: Inter, sans-serif; color: #333333; margin-bottom: 12px;">Supports .csv, .tsv, .txt, .xlsx, .xls, .ods, .json.</p>
""").strip())

        up_file = st.file_uploader(
            "Upload file",
            type=["csv", "tsv", "txt", "xlsx", "xls", "ods", "json"],
            key=f"top_uploader_{key_suffix}"
        )
        if up_file is not None:
            df_loaded, err = read_tabular_file(up_file)
            if err:
                st.session_state['upload_error'] = err
            else:
                st.session_state['upload_error'] = None
                st.session_state['raw_uploaded_df'] = df_loaded
                st.session_state['uploaded_filename'] = up_file.name
                m, s = auto_match_columns(df_loaded.columns, df_loaded)
                st.session_state['column_mapping'] = m
                st.session_state['column_status'] = s
                st.session_state['global_class_filter'] = "All classes"
                st.rerun()

        if st.session_state.get('raw_uploaded_df') is not None:
            show_html("<div class='brutalist-divider' style='margin: 12px 0;'></div>")
            if st.button("RESET / CLEAR DATASET", use_container_width=True, key=f"btn_reset_{key_suffix}"):
                st.session_state['raw_uploaded_df'] = None
                st.session_state['uploaded_filename'] = None
                st.session_state['column_mapping'] = {}
                st.session_state['column_status'] = {}
                st.session_state['upload_error'] = None
                st.session_state['consent_excluded_count'] = 0
                st.session_state['global_class_filter'] = "All classes"
                st.session_state['scoring_stats'] = {}
                st.rerun()

def render_empty_state_card(window_name: str = "profile_query.exe"):
    """
    Renders the exact empty-state card matching the user reference image:
    Light steel-blue fill (#A2BDC8), 4px solid black border, 6px hard offset drop shadow,
    heavy bold condensed uppercase heading, compact layout, folder icon removed.
    """
    col_l, col_r = st.columns([4, 1.2])
    with col_l:
        show_html(textwrap.dedent(f"""\
<div style="display: inline-flex; align-items: center; background-color: #1A1A1A; border: 2px solid #1A1A1A; border-radius: 4px; padding: 6px 14px; margin-bottom: 8px;">
<span class="dot dot-red" style="margin-right: 6px;"></span>
<span class="dot dot-yellow" style="margin-right: 6px;"></span>
<span class="dot dot-green" style="margin-right: 12px;"></span>
<span style="font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #FFFFF0;">{window_name}</span>
</div>
""").strip())
    with col_r:
        render_top_upload_button(key_suffix=f"empty_{window_name}")

    show_html(textwrap.dedent("""\
<div style="background-color: #A2BDC8; border: 4px solid #000000; border-radius: 4px; box-shadow: 6px 6px 0px #000000; padding: 34px 24px 34px 24px; text-align: center; margin-top: 6px; margin-bottom: 24px;">
<h1 style="font-family: 'Oswald', 'Space Grotesk', sans-serif; font-size: 2.35rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: #000000; margin: 0 0 14px 0; line-height: 1.1;">NO DATASET UPLOADED YET</h1>
<p style="font-family: 'Oswald', 'Space Grotesk', sans-serif; font-size: 0.96rem; font-weight: 600; color: #000000; max-width: 640px; margin: 0 auto; line-height: 1.38; letter-spacing: 0.2px; text-align: center;">
Use the CSV Upload button above to load a student dataset. (Expected columns: Name, Roll Number, Department, Year, Semester, Attendance %, CGPA, Backlogs, Student Category, Consent).
</p>
</div>
""").strip())

def render_stat_card(label: str, value: str, variant: str = "white", subtext: str = None, card_class: str = ""):
    subtext_html = f"<p style='font-size:0.75rem; margin-top:8px; font-weight:500; font-family: Inter, sans-serif;'>{subtext}</p>" if subtext else ""
    return textwrap.dedent(f"""\
<div class="stat-card stat-card-{variant} {card_class}">
<div class="stat-label">{label}</div>
<div class="stat-value">{value}</div>
{subtext_html}
</div>
""").strip()

def get_risk_badge_html(risk_level):
    if risk_level == "High":
        return '<span class="badge-high">● HIGH RISK</span>'
    elif risk_level == "Medium":
        return '<span class="badge-medium">▲ MEDIUM RISK</span>'
    elif risk_level == "Low":
        return '<span class="badge-low">■ LOW RISK</span>'
    else:
        return '<span class="badge-invalid">⚠ UNSCORED</span>'

def calculate_top_factors_and_intervention(student_row):
    att = student_row.get('attendance_percent', np.nan)
    cgpa = student_row.get('cgpa', np.nan)
    bk = student_row.get('backlogs', np.nan)
    category = student_row.get('student_category', 'Regular')
    _, risk_level, factors, attendance_exempt = calculate_student_risk(att, cgpa, bk, category)
    top_factors = sorted(
        (factor for factor, points in factors.items() if points > 0),
        key=factors.get,
        reverse=True,
    )

    reasons = []
    for f in top_factors:
        if f == 'attendance_percent' and pd.notna(att):
            reasons.append(f"low attendance ({int(att)}%)")
        elif f == 'cgpa' and pd.notna(cgpa):
            reasons.append(f"low CGPA ({cgpa})")
        elif f == 'backlogs' and pd.notna(bk):
            reasons.append(f"{int(bk)} active backlog(s)")

    if not reasons:
        explanation = "Academic indicators are within the low-risk range."
    else:
        risk_description = {"High": "High risk", "Medium": "Moderate risk", "Low": "Low risk"}[risk_level]
        explanation = f"{risk_description} based on: " + ", ".join(reasons) + "."

    if attendance_exempt and pd.notna(att) and att < 75:
        explanation += (
            " Attendance is not counted against this student because their category "
            "is eligible for college-approved attendance accommodation."
        )

    top_factor = top_factors[0] if top_factors else "None"
    if attendance_exempt and pd.notna(att) and att < 75:
        intervention = "Coordinate college attendance accommodation for approved duties"
    elif top_factor == 'attendance_percent':
        intervention = "Recommend attendance counseling"
    elif top_factor == 'backlogs':
        intervention = "Recommend academic support"
    elif top_factor == 'cgpa':
        intervention = "Recommend academic support program"
    else:
        intervention = "Routine academic progress tracking"

    return explanation, intervention


# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
st.sidebar.html(textwrap.dedent("""\
<div style='margin-bottom:12px;'>
<div style='font-family: "Space Mono", monospace; font-size: 0.72rem; font-weight:700; color: #DE6B35; letter-spacing: 1px;'>SYS.CORE // SAARTHI</div>
<h2 class='brutalist-title' style='margin-bottom:0px; font-size:1.6rem; letter-spacing: -0.5px;'>DROPOUT RISK</h2>
<p style='font-family: "Space Mono", monospace; font-size:0.75rem; font-weight:600; color:#1A1A1A; margin-top:2px; margin-bottom:18px;'>EARLY WARNING ENGINE</p>
</div>
""").strip())

NAV_OPTIONS = [
    "▲ Overview",
    "Student Risk Directory",
    "Individual Student Profile",
    "Model Intelligence"
]

page = st.sidebar.radio(
    "SYSTEM SECTIONS",
    NAV_OPTIONS,
    index=0
)

st.sidebar.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

status_color = "#27C93F" if st.session_state.get('raw_uploaded_df') is not None else "#FFBD2E"
status_text = "● SYSTEM ONLINE" if st.session_state.get('raw_uploaded_df') is not None else "○ AWAITING DATASET"
cohort_info = f"{len(st.session_state['raw_uploaded_df'])} ROWS" if st.session_state.get('raw_uploaded_df') is not None else "NO FILE LOADED"

st.sidebar.html(textwrap.dedent(f"""\
<div class="sidebar-sys-box">
<div style="color: {status_color}; font-weight:700; margin-bottom:4px;">{status_text}</div>
<div>KERNEL: <b>METRICS_RULES</b></div>
<div>COHORT: <b>{cohort_info}</b></div>
<div>CHROME: <b>RETRO_OS_2.0</b></div>
</div>
""").strip())

# Requirement 18: Prototype Disclaimer Label in Sidebar
st.sidebar.html(textwrap.dedent("""\
<div class="disclaimer-badge">
⚠️ Rule-based live score; not a calibrated probability. Offline model benchmark: UCI dataset.
</div>
""").strip())


# =====================================================================
# GLOBAL TOP ERROR / WARNING DISPLAY
# =====================================================================
if st.session_state.get('upload_error'):
    st.error(f"❌ {st.session_state['upload_error']}")


# =====================================================================
# ACTIVE COHORT DATA PREPARATION (IF FILE UPLOADED)
# =====================================================================
df = None
risk_available = False
risk_msg = ""


def render_page_top_bar(window_name: str, key_suffix: str, cohort_df: pd.DataFrame):
    """
    Renders top row across all five active sections:
    .exe window indicator, global Class Selector, and CSV Upload button.
    """
    top_l, top_m, top_r = st.columns([2.5, 2.0, 1.2], vertical_alignment="center")
    with top_l:
        show_html(textwrap.dedent(f"""\
<div style="display: inline-flex; align-items: center; background-color: #1A1A1A; border: 2px solid #1A1A1A; border-radius: 4px; padding: 6px 14px; margin-bottom: 0px;">
<span class="dot dot-red" style="margin-right: 6px;"></span>
<span class="dot dot-yellow" style="margin-right: 6px;"></span>
<span class="dot dot-green" style="margin-right: 12px;"></span>
<span style="font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #FFFFF0;">{window_name}</span>
</div>
""").strip())
    with top_m:
        raw_classes = cohort_df['class_name'].dropna().astype(str).str.strip().unique() if (cohort_df is not None and 'class_name' in cohort_df.columns) else []
        classes = sorted([c for c in raw_classes if c and c.lower() != 'nan'])
        class_options = ["All classes"] + classes

        curr_class = st.session_state.get('global_class_filter', 'All classes')
        if curr_class not in class_options:
            curr_class = 'All classes'
            st.session_state['global_class_filter'] = 'All classes'

        st.selectbox(
            "CLASS FILTER",
            class_options,
            index=class_options.index(curr_class),
            key=f"class_sel_{key_suffix}",
            label_visibility="collapsed"
        )
        if st.session_state.get(f"class_sel_{key_suffix}") != st.session_state.get('global_class_filter'):
            st.session_state['global_class_filter'] = st.session_state[f"class_sel_{key_suffix}"]
            st.rerun()

    with top_r:
        render_top_upload_button(key_suffix=key_suffix)

full_df = None
df = None
risk_available = False
risk_msg = ""

if st.session_state.get('raw_uploaded_df') is not None:
    raw_df = st.session_state['raw_uploaded_df']
    col_map = st.session_state['column_mapping']
    full_df, risk_available, risk_msg = process_uploaded_cohort(raw_df, col_map)
    
    # Apply active global class filter
    raw_classes = full_df['class_name'].dropna().astype(str).str.strip().unique() if 'class_name' in full_df.columns else []
    classes = sorted([c for c in raw_classes if c and c.lower() != 'nan'])
    class_options = ["All classes"] + classes
    
    selected_class = st.session_state.get('global_class_filter', 'All classes')
    if selected_class not in class_options:
        selected_class = "All classes"
        st.session_state['global_class_filter'] = "All classes"
        
    if selected_class != "All classes" and 'class_name' in full_df.columns:
        df = full_df[full_df['class_name'] == selected_class].copy().reset_index(drop=True)
    else:
        df = full_df.copy()


# =====================================================================
# PAGE 1: OVERVIEW PAGE
# =====================================================================
if "Overview" in page:
    if df is None:
        render_empty_state_card("overview.exe")
    else:
        render_page_top_bar("overview.exe", "overview_active", full_df)

        # Main Hero Window Container
        with st.container(border=True):
            render_window_header("saarthi_core.exe", "SYSTEM_READY")
            fname = st.session_state.get('uploaded_filename', 'DATASET')
            show_html(textwrap.dedent(f"""\
<div style="padding: 6px 4px 10px 4px;">
<div style="font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; color: #DE6B35; letter-spacing: 1px; margin-bottom: 4px;">// COHORT ANALYTICS // {fname.upper()} // {len(df)} STUDENTS LOADED</div>
<h1 class='brutalist-title' style='font-size: 2.8rem; margin: 0; line-height: 1.05;'>EARLY DROPOUT RISK PREDICTION</h1>
<p style='font-size: 1rem; color: #1A1A1A; font-weight: 500; margin-top: 8px; margin-bottom: 0px;'>Active cohort: <b>{fname}</b> ({len(df)} records). Analytics computed dynamically from uploaded file.</p>
<div class="torn-paper-edge"></div>
</div>
""").strip())

        # Requirement 11: Column-Check Panel (Hide full grid when all matched, one line shown; else collapsed expander for unmatched/guessed)
        raw_columns = list(raw_df.columns)
        curr_map = dict(st.session_state.get('column_mapping', {}))
        curr_status = dict(st.session_state.get('column_status', {}))
        
        # Check which expected columns are unmatched or guessed
        unmatched_keys = [exp for exp in EXPECTED_COLUMNS if curr_status.get(exp) != "FOUND" or curr_map.get(exp) not in raw_columns]
        
        if not unmatched_keys:
            show_html(textwrap.dedent("""\
<div style="background-color: #E8F5E9; border: 2.5px solid #1A1A1A; border-radius: 4px; padding: 10px 16px; margin: 8px 0 14px 0; font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #1B5E20; box-shadow: 3px 3px 0px #1A1A1A;">
✓ All columns matched
</div>
""").strip())
        else:
            with st.expander(f"⚙ UNMATCHED / GUESSED COLUMNS ({len(unmatched_keys)}) // MANUAL MAPPING REQUIRED", expanded=False):
                show_html(textwrap.dedent("""\
<div style="font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; margin-bottom: 12px; color: #1A1A1A;">
THE FOLLOWING COLUMNS REQUIRE MANUAL CONFIRMATION OR MAPPING:
</div>
""").strip())
                updated_map = dict(curr_map)
                updated_status = dict(curr_status)
                
                c_cols = st.columns(2)
                for idx, exp in enumerate(unmatched_keys):
                    col_box = c_cols[idx % 2]
                    curr_matched = updated_map.get(exp)
                    c_stat = updated_status.get(exp, "MISSING")
                    if c_stat == "GUESSED":
                        badge = "<span style='background-color:#FFD54F; color:#1A1A1A; padding:2px 8px; border-radius:3px; font-weight:700; font-family: monospace;'>GUESSED, please confirm</span>"
                    else:
                        badge = "<span style='background-color:#FF8A80; color:#1A1A1A; padding:2px 8px; border-radius:3px; font-weight:700; font-family: monospace;'>MISSING</span>"
                    
                    with col_box:
                        st.markdown(f"**{exp}** {badge}", unsafe_allow_html=True)
                        opts = ["-- Not Mapped / Missing --"] + raw_columns
                        default_idx = (opts.index(curr_matched)) if curr_matched in raw_columns else 0
                        sel = st.selectbox(
                            f"Map column for {exp}",
                            opts,
                            index=default_idx,
                            key=f"sel_map_{exp}",
                            label_visibility="collapsed"
                        )
                        new_val = None if sel == "-- Not Mapped / Missing --" else sel
                        if new_val != updated_map.get(exp):
                            updated_map[exp] = new_val
                            updated_status[exp] = "FOUND" if new_val is not None else "MISSING"

                if updated_map != st.session_state['column_mapping']:
                    st.session_state['column_mapping'] = updated_map
                    st.session_state['column_status'] = updated_status
                    st.rerun()

        # Requirement 12: Loading Summary & Preview (Rows loaded, invalid cells per column, 5-row preview)
        stats = st.session_state.get('scoring_stats', {})
        total_rows = stats.get('total_students', len(full_df))
        inv_att = stats.get('invalid_att_count', 0)
        inv_cgpa = stats.get('invalid_cgpa_count', 0)
        unparsed_bk = stats.get('unparsed_bk_count', 0)
        inv_bk = stats.get('invalid_bk_count', 0)
        excluded_scoring = stats.get('excluded_count', 0)

        with st.container(border=True):
            render_window_header("dataset_ingest_telemetry.exe", "CLEANED_DATA_SUMMARY")
            show_html(textwrap.dedent(f"""\
<div style="font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #1A1A1A; margin-bottom: 10px;">
DATASET INGESTION SUMMARY: {total_rows} ROWS LOADED
</div>
""").strip())
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                show_html(f'''<div style="border:2px solid #1A1A1A; background:#FFFFFF; padding:10px; border-radius:4px; box-shadow:3px 3px 0 #1A1A1A;"><div style="font-size:0.72rem; font-weight:700; font-family:'Space Mono',monospace; color:#666;">TOTAL ROWS LOADED</div><div style="font-size:1.5rem; font-weight:800; font-family:'Space Mono',monospace;">{total_rows}</div></div>''')
            with col_s2:
                att_bg = "#FFCDD2" if inv_att > 0 else "#FFFFFF"
                show_html(f'''<div style="border:2px solid #1A1A1A; background:{att_bg}; padding:10px; border-radius:4px; box-shadow:3px 3px 0 #1A1A1A;"><div style="font-size:0.72rem; font-weight:700; font-family:'Space Mono',monospace; color:#1A1A1A;">INVALID ATTENDANCE</div><div style="font-size:1.5rem; font-weight:800; font-family:'Space Mono',monospace;">{inv_att}</div></div>''')
            with col_s3:
                cgpa_bg = "#FFCDD2" if inv_cgpa > 0 else "#FFFFFF"
                show_html(f'''<div style="border:2px solid #1A1A1A; background:{cgpa_bg}; padding:10px; border-radius:4px; box-shadow:3px 3px 0 #1A1A1A;"><div style="font-size:0.72rem; font-weight:700; font-family:'Space Mono',monospace; color:#1A1A1A;">INVALID/BOUND CGPA</div><div style="font-size:1.5rem; font-weight:800; font-family:'Space Mono',monospace;">{inv_cgpa}</div></div>''')
            with col_s4:
                bk_bg = "#FFE082" if (unparsed_bk + inv_bk) > 0 else "#FFFFFF"
                show_html(f'''<div style="border:2px solid #1A1A1A; background:{bk_bg}; padding:10px; border-radius:4px; box-shadow:3px 3px 0 #1A1A1A;"><div style="font-size:0.72rem; font-weight:700; font-family:'Space Mono',monospace; color:#1A1A1A;">UNPARSED/INVALID BACKLOGS</div><div style="display:flex; align-items:baseline; gap:8px; white-space:nowrap;"><span style="font-size:1.5rem; font-weight:800; font-family:'Space Mono',monospace;">{unparsed_bk + inv_bk}</span><span style="font-size:0.72rem; font-weight:700; font-family:'Space Mono',monospace;">{unparsed_bk} unparsed</span></div></div>''')
            
            if excluded_scoring > 0:
                show_html(textwrap.dedent(f"""\
<div style="margin-top:12px; background:#FFF3C4; border:2px solid #1A1A1A; padding:8px 12px; border-radius:4px; font-family:'Space Mono',monospace; font-size:0.78rem; font-weight:700; color:#1A1A1A;">
⚠️ {excluded_scoring} student(s) could not be scored because one or more predictors are missing or invalid.
</div>
""").strip())

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.markdown("**CLEANED DATA PREVIEW (FIRST 5 ROWS):**")
            preview_cols = [c for c in ['student_id', 'name', 'class_name', 'department', 'year', 'attendance_percent', 'cgpa', 'backlogs_display', 'predicted_risk', 'student_category'] if c in df.columns]
            st.dataframe(df[preview_cols].head(5), use_container_width=True, hide_index=True)

        # Consent Notice if filtered
        if st.session_state.get('consent_excluded_count', 0) > 0:
            st.info(f"🔒 Consent Filter: Excluded {st.session_state['consent_excluded_count']} student(s) without affirmative consent.")

        # 3 Stat Cards (Requirement 12)
        total_students = len(df)
        if risk_available:
            high_risk_count = int((df['predicted_risk'] == 'High').sum())
            high_risk_pct = (high_risk_count / total_students * 100) if total_students > 0 else 0.0
            high_risk_val_str = f"{high_risk_pct:.1f}%"
            at_risk_count_str = str(high_risk_count)
            risk_subtext = "Immediate counseling intervention"
        else:
            high_risk_val_str = "UNAVAILABLE"
            at_risk_count_str = "UNAVAILABLE"
            risk_subtext = risk_msg

        c1, c2, c3 = st.columns(3)
        with c1:
            show_html(render_stat_card("TOTAL STUDENTS", str(total_students), variant="white", subtext="Valid records in uploaded class dataset"))
        with c2:
            show_html(render_stat_card("HIGH RISK %", high_risk_val_str, variant="orange", subtext=risk_subtext))
        with c3:
            show_html(render_stat_card("AT RISK COUNT", at_risk_count_str, variant="yellow", subtext=risk_subtext))

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Card: Risk Distribution Bar Chart (Requirement 12 & 16)
        with st.container(border=True):
            render_window_header("risk_distribution.exe", "COHORT_TELEMETRY")
            st.markdown("<h3 class='brutalist-title'>RISK DISTRIBUTION</h3>", unsafe_allow_html=True)

            if not risk_available:
                show_html(textwrap.dedent(f"""\
<div style="background-color: #FFF3C4; border: 2.5px solid #1A1A1A; border-radius: 4px; padding: 16px; font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #1A1A1A;">
⚠️ {risk_msg}
</div>
""").strip())
            else:
                counts = df['predicted_risk'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
                fig, ax = plt.subplots(figsize=(10, 3.2))
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')

                bars = ax.barh(counts.index, counts.values, height=0.55)
                bars[0].set_color('#FFFFFF'); bars[0].set_edgecolor('#1A1A1A'); bars[0].set_linewidth(2.5)
                bars[1].set_color('#E8C547'); bars[1].set_edgecolor('#1A1A1A'); bars[1].set_linewidth(2.5)
                bars[2].set_color('#DE6B35'); bars[2].set_edgecolor('#1A1A1A'); bars[2].set_linewidth(2.5)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#1A1A1A'); ax.spines['left'].set_color('#1A1A1A')
                ax.spines['bottom'].set_linewidth(2.5); ax.spines['left'].set_linewidth(2.5)
                ax.tick_params(colors='#1A1A1A', labelsize=10, width=2)
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')

                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + max(0.5, total_students * 0.01), bar.get_y() + bar.get_height()/2, f'{int(w)}', 
                            va='center', ha='left', color='#1A1A1A', fontweight='bold', fontsize=11)

                plt.tight_layout()
                st.pyplot(fig)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Requirement 13: Data Explorer Auto-Generated from Whatever Columns Exist
        with st.container(border=True):
            render_window_header("data_explorer.exe", "AUTO_TELEMETRY")
            st.markdown("<h3 class='brutalist-title'>DATA EXPLORER</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.85rem; color:#555555; font-family:\"Space Mono\", monospace; font-weight:600; margin-bottom:16px;'>DYNAMIC SUMMARY STATS & CHARTS AUTO-GENERATED FROM UPLOADED COLUMNS</p>", unsafe_allow_html=True)

            # Discover numeric vs categorical columns in raw uploaded dataset
            candidate_numeric = []
            candidate_categorical = []

            for col in raw_df.columns:
                if normalize_header(col) in ['timestamp', 'time stamp']:
                    continue
                # Try numeric conversion
                num_s = pd.to_numeric(raw_df[col], errors='coerce')
                valid_num_pct = num_s.notna().mean()
                if valid_num_pct >= 0.5 and num_s.nunique() > 2 and 'roll' not in col.lower() and 'id' not in col.lower():
                    candidate_numeric.append(col)
                elif raw_df[col].nunique() <= 20:
                    candidate_categorical.append(col)

            tab_num, tab_cat, tab_corr = st.tabs(["📊 NUMERIC DISTRIBUTIONS", "☷ CATEGORICAL BREAKDOWNS", "◈ CORRELATION VIEW"])

            with tab_num:
                if not candidate_numeric:
                    st.info("No numeric performance columns detected in this file.")
                else:
                    sel_num_col = st.selectbox("Select Numeric Column to Analyze", candidate_numeric, key="sel_num_exp")
                    s = pd.to_numeric(raw_df[sel_num_col], errors='coerce').dropna()
                    
                    if len(s) > 0:
                        col_stat, col_chart = st.columns([1, 2])
                        with col_stat:
                            stat_df = pd.DataFrame({
                                "METRIC": ["Mean", "Median", "Min", "Max", "Std Dev", "Valid Records", "Missing/Invalid"],
                                "VALUE": [
                                    f"{s.mean():.2f}",
                                    f"{s.median():.2f}",
                                    f"{s.min():.2f}",
                                    f"{s.max():.2f}",
                                    f"{s.std():.2f}",
                                    str(len(s)),
                                    str(len(raw_df) - len(s))
                                ]
                            })
                            st.dataframe(stat_df, use_container_width=True, hide_index=True)
                        
                        with col_chart:
                            fig_hist, ax_h = plt.subplots(figsize=(7, 3.2))
                            fig_hist.patch.set_facecolor('#FFFFFF')
                            ax_h.set_facecolor('#FFFFFF')
                            ax_h.hist(s, bins=min(15, max(5, s.nunique())), color='#E8C547', edgecolor='#1A1A1A', linewidth=2)
                            ax_h.spines['top'].set_visible(False); ax_h.spines['right'].set_visible(False)
                            ax_h.spines['bottom'].set_color('#1A1A1A'); ax_h.spines['left'].set_color('#1A1A1A')
                            ax_h.spines['bottom'].set_linewidth(2.5); ax_h.spines['left'].set_linewidth(2.5)
                            ax_h.set_title(f"Histogram: {sel_num_col}", fontfamily='sans-serif', fontweight='bold', fontsize=11)
                            plt.tight_layout()
                            st.pyplot(fig_hist)

            with tab_cat:
                if not candidate_categorical:
                    st.info("No categorical columns detected in this file.")
                else:
                    sel_cat_col = st.selectbox("Select Categorical Column", candidate_categorical, key="sel_cat_exp")
                    cat_counts = raw_df[sel_cat_col].astype(str).str.strip().value_counts().head(10)
                    
                    fig_cat, ax_c = plt.subplots(figsize=(8, max(2.5, 0.4 * len(cat_counts))))
                    fig_cat.patch.set_facecolor('#FFFFFF')
                    ax_c.set_facecolor('#FFFFFF')
                    bars_c = ax_c.barh(cat_counts.index, cat_counts.values, color='#DE6B35', edgecolor='#1A1A1A', linewidth=2, height=0.55)
                    ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
                    ax_c.spines['bottom'].set_color('#1A1A1A'); ax_c.spines['left'].set_color('#1A1A1A')
                    ax_c.spines['bottom'].set_linewidth(2.5); ax_c.spines['left'].set_linewidth(2.5)
                    for bar in bars_c:
                        w = bar.get_width()
                        ax_c.text(w + 0.2, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', ha='left', fontweight='bold', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig_cat)

            with tab_corr:
                if len(candidate_numeric) < 2:
                    st.info("At least two numeric columns required for correlation analysis.")
                else:
                    # Numeric correlation matrix
                    num_matrix = raw_df[candidate_numeric].apply(pd.to_numeric, errors='coerce').dropna()
                    if len(num_matrix) > 2:
                        corr = num_matrix.corr().round(2)
                        st.markdown("**CORRELATION MATRIX:**")
                        st.dataframe(corr, use_container_width=True)

                        c_x, c_y = st.columns(2)
                        with c_x:
                            x_col = st.selectbox("X-Axis Variable", candidate_numeric, index=0)
                        with c_y:
                            y_col = st.selectbox("Y-Axis Variable", candidate_numeric, index=min(1, len(candidate_numeric)-1))

                        fig_sc, ax_sc = plt.subplots(figsize=(7, 3.2))
                        fig_sc.patch.set_facecolor('#FFFFFF')
                        ax_sc.set_facecolor('#FFFFFF')
                        ax_sc.scatter(num_matrix[x_col], num_matrix[y_col], color='#E8C547', edgecolor='#1A1A1A', linewidth=1.5, s=60)
                        ax_sc.spines['top'].set_visible(False); ax_sc.spines['right'].set_visible(False)
                        ax_sc.spines['bottom'].set_color('#1A1A1A'); ax_sc.spines['left'].set_color('#1A1A1A')
                        ax_sc.spines['bottom'].set_linewidth(2.5); ax_sc.spines['left'].set_linewidth(2.5)
                        ax_sc.set_xlabel(x_col, fontweight='bold'); ax_sc.set_ylabel(y_col, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig_sc)


# =====================================================================
# PAGE 2: STUDENT RISK DIRECTORY
# =====================================================================
elif "Student Risk Directory" in page:
    if df is None:
        render_empty_state_card("student_directory.exe")
    else:
        render_page_top_bar("student_directory.exe", "dir", full_df)

        with st.container(border=True):
            render_window_header("student_directory.exe", "DATABASE_QUERY")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>STUDENT RISK DIRECTORY</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 20px;'>FILTERABLE ACADEMIC & BEHAVIORAL REGISTER</p>", unsafe_allow_html=True)

            col_dep, col_risk = st.columns(2)
            actual_depts = ["ALL"] + sorted([str(d) for d in df['department'].unique() if pd.notna(d)])
            actual_risks = ["ALL"] + sorted([str(r) for r in df['predicted_risk'].unique()])

            with col_dep:
                sel_dept = st.selectbox("FILTER BY DEPARTMENT", actual_depts)
            with col_risk:
                sel_risk = st.selectbox("FILTER BY RISK LEVEL", actual_risks)

        filtered = df.copy()
        if sel_dept != "ALL":
            filtered = filtered[filtered['department'] == sel_dept]
        if sel_risk != "ALL":
            filtered = filtered[filtered['predicted_risk'] == sel_risk]

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            render_window_header("risk_registry.exe", "RECORDS_LOADED")
            
            # Prepare display table marking invalid cells
            display_rows = []
            for _, r in filtered.iterrows():
                att_disp = f"{r['attendance_percent']}%" if pd.notna(r['attendance_percent']) else "INVALID"
                cgpa_disp = f"{r['cgpa']:.2f}" if pd.notna(r['cgpa']) else "INVALID"
                if r.get('backlogs_status') == 'UNPARSED':
                    bk_disp = str(r.get('backlogs_display', 'UNPARSED'))
                elif pd.notna(r['backlogs']):
                    bk_disp = f"{int(r['backlogs'])}"
                else:
                    bk_disp = "INVALID"
                score_disp = f"{r['risk_score']:.1f}" if pd.notna(r['risk_score']) else "N/A"
                
                display_rows.append({
                    "ROLL NUMBER": str(r['student_id']),
                    "NAME": str(r['name']),
                    "CLASS": str(r.get('class_name', 'General')),
                    "DEPARTMENT": str(r['department']),
                    "YEAR": f"Year {int(r['year'])}" if pd.notna(r['year']) else "INVALID",
                    "SEMESTER": str(r.get('semester', 'N/A')),
                    "CGPA": cgpa_disp,
                    "ATTENDANCE": att_disp,
                    "BACKLOGS": bk_disp,
                    "RISK LEVEL": str(r['predicted_risk']),
                    "SCORE": score_disp,
                    "STUDENT CATEGORY": str(r.get('student_category', 'Regular'))
                })

            disp_df = pd.DataFrame(display_rows)
            st.dataframe(disp_df, use_container_width=True, hide_index=True)
            show_html(textwrap.dedent(f"""\
<div style="font-family: 'Space Mono', monospace; font-size:0.8rem; font-weight:700; margin-top:10px; color:#1A1A1A;">
STATUS: SHOWING {len(disp_df)} OF {len(df)} RECORDS // INVALID CELLS MARKED EXPLICITLY
</div>
""").strip())


# =====================================================================
# PAGE 3: INDIVIDUAL STUDENT PROFILE
# =====================================================================
elif "Individual Student Profile" in page:
    if df is None:
        render_empty_state_card("profile_query.exe")
    else:
        render_page_top_bar("profile_query.exe", "profile", full_df)

        with st.container(border=True):
            render_window_header("profile_query.exe", "STUDENT_INDEX")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>INDIVIDUAL STUDENT PROFILE</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 16px;'>DEEP-DIVE DIAGNOSIS & INTERVENTION PROTOCOL</p>", unsafe_allow_html=True)

            student_options = sorted([str(n) for n in df['name'].unique()])
            sel_student_name = st.selectbox("SELECT STUDENT NAME", student_options)

        student = df[df['name'] == sel_student_name].iloc[0]
        has_risk = pd.notna(student.get('risk_score')) and student.get('predicted_risk') != "UNSCORED"

        if has_risk:
            explanation, intervention = calculate_top_factors_and_intervention(student)
            score_str = f"{student['risk_score']:.1f}"
        else:
            explanation = "Risk scoring unavailable due to missing or invalid predictors."
            intervention = "Ensure Attendance %, CGPA and Backlogs are provided"
            score_str = "N/A"

        c1, c2 = st.columns([1, 2])
        with c1:
            with st.container(border=True):
                render_window_header("risk_eval.exe", "INFERENCE")
                st.markdown("<div class='stat-label'>SCORE (0-100)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='stat-value' style='font-size: 4.5rem; margin: 8px 0; color:#1A1A1A;'>{score_str}</div>", unsafe_allow_html=True)
                st.markdown(get_risk_badge_html(student['predicted_risk']), unsafe_allow_html=True)
                st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
                y_disp = f"Year {int(student['year'])}" if pd.notna(student['year']) else "INVALID"
                show_html(textwrap.dedent(f"""\
<div style="font-family: 'Space Mono', monospace; font-size:0.85rem; line-height: 1.8;">
<div><b>ROLL NO:</b> {student['student_id']}</div>
<div><b>DEPT:</b> {student['department']}</div>
<div><b>YEAR:</b> {y_disp}</div>
<div><b>SEMESTER:</b> {student.get('semester', 'N/A')}</div>
</div>
""").strip())

        with c2:
            with st.container(border=True):
                render_window_header("risk_diagnosis.exe", "TELEMETRY")
                st.markdown("<h3 class='brutalist-title'>PRIMARY RISK CAUSES</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 1.05rem; font-weight:500; color:#1A1A1A;'>{explanation}</p>", unsafe_allow_html=True)
                st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
                st.markdown("<h3 class='brutalist-title'>RECOMMENDED INTERVENTION</h3>", unsafe_allow_html=True)
                show_html(textwrap.dedent(f"""\
<div style="background-color:#E8C547; border:2.5px solid #1A1A1A; border-radius:4px; padding:14px 18px; font-weight:700; color:#1A1A1A; font-family: 'Space Mono', monospace; box-shadow: 3px 3px 0px #1A1A1A; font-size: 0.95rem;">
{intervention.upper()}
</div>
""").strip())

        # Real Academic Metrics Grid
        with st.container(border=True):
            render_window_header("academic_metrics.exe", "TELEMETRY")
            st.markdown("<h3 class='brutalist-title'>ACADEMIC METRICS GRID</h3>", unsafe_allow_html=True)

            cgpa_val = f"{student['cgpa']:.2f}" if pd.notna(student['cgpa']) else "INVALID"
            att_val = f"{student['attendance_percent']}%" if pd.notna(student['attendance_percent']) else "INVALID"
            if student.get('backlogs_status') == 'UNPARSED':
                bk_val = str(student.get('backlogs_display', 'UNPARSED'))
            elif pd.notna(student['backlogs']):
                bk_val = f"{int(student['backlogs'])}"
            else:
                bk_val = "INVALID"
            cat_val = str(student.get('student_category', 'Regular'))
            category_card_class = "academic-metric-card academic-category-card"
            if cat_val != "Regular":
                category_card_class += " academic-category-long-value"

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                show_html(render_stat_card("CGPA", cgpa_val, variant="white", card_class="academic-metric-card"))
            with m2:
                show_html(render_stat_card("ATTENDANCE", att_val, variant="yellow", card_class="academic-metric-card"))
            with m3:
                show_html(render_stat_card("BACKLOGS", bk_val, variant="orange", card_class="academic-metric-card"))
            with m4:
                show_html(render_stat_card("CATEGORY", cat_val, variant="dark", card_class=category_card_class))


# =====================================================================
# PAGE 4: MODEL INTELLIGENCE (BENCHMARK ONLY FOR METRICS)
# =====================================================================
elif "Model Intelligence" in page:
    if df is None:
        render_empty_state_card("model_telemetry.exe")
    else:
        render_page_top_bar("model_telemetry.exe", "model", full_df)

        with st.container(border=True):
            render_window_header("model_telemetry.exe", "GRADIENT_BOOST")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>MODEL INTELLIGENCE</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 20px;'>COHORT RULE-BASED RISK, OFFLINE FEATURE WEIGHTS & MODEL BENCHMARKS</p>", unsafe_allow_html=True)

        total_scored = len(df[df['predicted_risk'] != "UNSCORED"])
        high_count = int((df['predicted_risk'] == 'High').sum())
        high_pct = (high_count / total_scored * 100) if total_scored > 0 else 0.0
        avg_score = df['risk_score'].dropna().mean() if total_scored > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        with c1:
            show_html(render_stat_card("COHORT SCORED", str(total_scored), variant="white", subtext=f"Total: {len(df)} records in file"))
        with c2:
            show_html(render_stat_card("PREDICTED HIGH RISK", f"{high_count} ({high_pct:.1f}%)" if total_scored > 0 else "N/A", variant="orange", subtext="Priority intervention required"))
        with c3:
            show_html(render_stat_card("AVERAGE RISK SCORE", f"{avg_score:.1f}/100" if total_scored > 0 else "N/A", variant="yellow", subtext="Average rules-based score (0-100)"))

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        col_feat, col_dist = st.columns([1.2, 0.8])
        with col_feat:
            with st.container(border=True):
                render_window_header("feature_weights.exe", "IMPORTANCES")
                st.markdown("<h3 class='brutalist-title'>OFFLINE MODEL FEATURE IMPORTANCE</h3>", unsafe_allow_html=True)

                importances = model_obj.feature_importances_
                feat_series = pd.Series(importances, index=model_obj.feature_names_in_).sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')

                bars = ax.barh(feat_series.index, feat_series.values, color='#1A1A1A', height=0.6, edgecolor='#1A1A1A', linewidth=2)
                bars[-1].set_color('#DE6B35'); bars[-1].set_edgecolor('#1A1A1A')
                if len(bars) > 1:
                    bars[-2].set_color('#E8C547'); bars[-2].set_edgecolor('#1A1A1A')

                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#1A1A1A'); ax.spines['left'].set_color('#1A1A1A')
                ax.spines['bottom'].set_linewidth(2.5); ax.spines['left'].set_linewidth(2.5)
                ax.tick_params(colors='#1A1A1A', labelsize=9, width=2)
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')
                plt.tight_layout()
                st.pyplot(fig)

        with col_dist:
            with st.container(border=True):
                render_window_header("cohort_distribution.exe", "LIVE_COHORT")
                st.markdown("<h3 class='brutalist-title'>COHORT RISK BREAKDOWN</h3>", unsafe_allow_html=True)

                if total_scored == 0:
                    st.info("No students scored yet. Ensure Attendance %, CGPA and Backlogs are mapped.")
                else:
                    c_dist = df['predicted_risk'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
                    dist_df = pd.DataFrame({
                        "RISK LEVEL": ["Low Risk", "Medium Risk", "High Risk"],
                        "COUNT": [int(c_dist['Low']), int(c_dist['Medium']), int(c_dist['High'])],
                        "PERCENTAGE": [
                            f"{(c_dist['Low'] / total_scored * 100):.1f}%",
                            f"{(c_dist['Medium'] / total_scored * 100):.1f}%",
                            f"{(c_dist['High'] / total_scored * 100):.1f}%"
                        ]
                    })
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Offline Benchmark Container (Requirement 17: clearly labelled benchmark, not this cohort)
        with st.container(border=True):
            render_window_header("offline_benchmarks.exe", "OFFLINE_TEST_SPLIT")
            st.markdown("<h3 class='brutalist-title'>OFFLINE MODEL BENCHMARK (CALIBRATION BASELINE — NOT THIS COHORT)</h3>", unsafe_allow_html=True)
            show_html(textwrap.dedent("""\
<div style="background-color: #FFF3C4; border: 2px solid #1A1A1A; border-radius: 4px; padding: 10px 14px; font-family: 'Space Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #1A1A1A; margin-bottom: 16px; box-shadow: 2px 2px 0px #1A1A1A;">
[BENCHMARK EVALUATION — NOT THIS COHORT]: The metrics below reflect the pre-trained model's performance on a held-out test split (20% unseen data) during training calibration. Because the uploaded file contains live operational data without ground-truth dropout outcome labels, accuracy and recall are not computed on the uploaded cohort.
</div>
""").strip())

            bm_acc = eval_metrics['accuracy']
            bm_prec = eval_metrics['precision']
            bm_rec = eval_metrics['recall']

            bm1, bm2, bm3 = st.columns(3)
            with bm1:
                show_html(render_stat_card("BENCHMARK ACCURACY", f"{bm_acc*100:.1f}%", variant="white", subtext="Independent held-out calibration split"))
            with bm2:
                show_html(render_stat_card("BENCHMARK PRECISION", f"{bm_prec*100:.1f}%", variant="orange", subtext="Macro precision on calibration split"))
            with bm3:
                show_html(render_stat_card("BENCHMARK RECALL", f"{bm_rec*100:.1f}%", variant="yellow", subtext="Macro recall on calibration split"))

            st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
            st.markdown("<h4 class='brutalist-title' style='font-size: 1.1rem;'>BENCHMARK CONFUSION MATRIX (OFFLINE TEST EVALUATION)</h4>", unsafe_allow_html=True)
            labels = eval_metrics['cm_labels']
            cm = eval_metrics['confusion_matrix']
            cm_df = pd.DataFrame(cm, index=[f"ACTUAL {l} (BENCHMARK)" for l in labels], columns=[f"PRED {l}" for l in labels])
            st.table(cm_df)


# =====================================================================
# PAGE 5: HOD REPORT
# =====================================================================
elif "Department Report" in page:
    if df is None:
        render_empty_state_card("hod_report.exe")
    else:
        render_page_top_bar("hod_report.exe", "hod", full_df)

        with st.container(border=True):
            render_window_header("department_audit.exe", "DEPARTMENT_REPORT")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>DEPARTMENT SUMMARY REPORT</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 18px;'>DEPARTMENT-LEVEL RISK SUMMARY</p>", unsafe_allow_html=True)

            dept_options = ["All Departments"] + sorted([str(d) for d in df['department'].unique() if pd.notna(d)])
            selected_dept = st.selectbox("SELECT DEPARTMENT", dept_options)

        if selected_dept == "All Departments":
            report_df = df.copy()
        else:
            report_df = df[df['department'] == selected_dept].copy()

        total = len(report_df)
        high_count = int((report_df['predicted_risk'] == 'High').sum())
        med_count = int((report_df['predicted_risk'] == 'Medium').sum())
        low_count = int((report_df['predicted_risk'] == 'Low').sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_html(render_stat_card("TOTAL STUDENTS", str(total), variant="white"))
        with c2:
            show_html(render_stat_card("HIGH RISK", str(high_count) if risk_available else "N/A", variant="orange"))
        with c3:
            show_html(render_stat_card("MEDIUM RISK", str(med_count) if risk_available else "N/A", variant="yellow"))
        with c4:
            show_html(render_stat_card("LOW RISK", str(low_count) if risk_available else "N/A", variant="dark"))

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Consultation Priority List
        with st.container(border=True):
            render_window_header("consultation_roster.exe", "PRIORITY_LIST")
            st.markdown("<h3 class='brutalist-title'>STUDENTS REQUIRING CONSULTATION</h3>", unsafe_allow_html=True)

            high_risk_students = report_df[report_df['predicted_risk'] == 'High'].sort_values('risk_score', ascending=False) if risk_available else pd.DataFrame()

            if len(high_risk_students) == 0:
                st.markdown("<p style='font-family:\"Space Mono\", monospace; font-weight:700; color:#1A1A1A;'>No High-risk students in this selection. [STATUS: NOMINAL]</p>", unsafe_allow_html=True)
                consult_df = pd.DataFrame(columns=["Student","ID","Year","Risk Score","Primary Concern"])
            else:
                consult_rows = []
                for _, srow in high_risk_students.iterrows():
                    expl, _ = calculate_top_factors_and_intervention(srow)
                    y_str = f"Year {int(srow['year'])}" if pd.notna(srow['year']) else "N/A"
                    consult_rows.append({
                        "Student": srow['name'],
                        "ID": srow['student_id'],
                        "Year": y_str,
                        "Risk Score": f"{srow['risk_score']:.1f}",
                        "Primary Concern": expl
                    })
                consult_df = pd.DataFrame(consult_rows)
                st.dataframe(consult_df, use_container_width=True, hide_index=True)

        # Download PDF Report
        with st.container(border=True):
            render_window_header("export_engine.exe", "PDF_RENDERER")
            st.markdown("<h3 class='brutalist-title'>DOWNLOAD REPORT</h3>", unsafe_allow_html=True)

            def generate_pdf_report(dept_name, total_cnt, high_c, med_c, low_c, students_df):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 18)
                pdf.cell(0, 12, "EARLY DROPOUT RISK - DEPARTMENT SUMMARY REPORT", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 8, f"Department: {dept_name}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "OVERVIEW", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 7, f"Total Students: {total_cnt}", new_x="LMARGIN", new_y="NEXT")
                if risk_available:
                    pdf.cell(0, 7, f"High Risk: {high_c}   Medium Risk: {med_c}   Low Risk: {low_c}", new_x="LMARGIN", new_y="NEXT")
                else:
                    pdf.cell(0, 7, "Risk Scoring: Unavailable (Predictor columns missing)", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(6)

                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "STUDENTS REQUIRING CONSULTATION", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)

                if len(students_df) == 0:
                    pdf.cell(0, 7, "No High-risk students in this selection.", new_x="LMARGIN", new_y="NEXT")
                else:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(45, 7, "Student", border=1)
                    pdf.cell(25, 7, "ID", border=1)
                    pdf.cell(15, 7, "Year", border=1)
                    pdf.cell(25, 7, "Score", border=1)
                    pdf.cell(80, 7, "Primary Concern", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "", 9)
                    for _, r in students_df.iterrows():
                        pdf.cell(45, 7, str(r["Student"])[:22], border=1)
                        pdf.cell(25, 7, str(r["ID"])[:12], border=1)
                        pdf.cell(15, 7, str(r["Year"]), border=1)
                        pdf.cell(25, 7, str(r["Risk Score"]), border=1)
                        pdf.cell(80, 7, str(r["Primary Concern"])[:45], border=1, new_x="LMARGIN", new_y="NEXT")

                return bytes(pdf.output())

            pdf_bytes = generate_pdf_report(
                selected_dept, total, high_count, med_count, low_count,
                consult_df if len(high_risk_students) > 0 else pd.DataFrame(columns=["Student","ID","Year","Risk Score","Primary Concern"])
            )

            st.download_button(
                label="DOWNLOAD REPORT",
                data=pdf_bytes,
                file_name=f"Department_Report_{selected_dept.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
