import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from fpdf import FPDF

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="SAARTHI",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# RETRO OS & POSTER NEUBRUTALISM CSS
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap');

    /* Reset Streamlit Defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible !important; background-color: transparent !important;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 95% !important;
    }

    /* Global Typography & Outermost Page Background */
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
        margin: 0 0 1.25rem 0;
        border-radius: 2px 2px 0 0;
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

    [data-testid="stVerticalBlockBorderWrapper"] 
    {
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
        background-image: 
            linear-gradient(rgba(26, 26, 26, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(26, 26, 26, 0.05) 1px, transparent 1px) !important;
        background-size: 20px 20px !important;
        box-shadow: 4px 4px 0px #1A1A1A !important;
        overflow: hidden !important;
        margin-bottom: 1.25rem !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > 
    div {
        background-color: transparent !important;
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
    .stat-card-white .stat-label {
        color: #555555;
    }
    .stat-card-white .stat-value {
        color: #1A1A1A;
    }
    .stat-card-orange {
        background-color: #DE6B35 !important;
        background-image: none !important;
        color: #FFFFFF;
    }
    .stat-card-orange .stat-label {
        color: #FFF0E6;
    }
    .stat-card-orange .stat-value {
        color: #FFFFFF;
    }
    .stat-card-orange p {
        color: #FFF0E6 !important;
    }
    .stat-card-yellow {
        background-color: #E8C547 !important;
        background-image: none !important;
        color: #1A1A1A;
    }
    .stat-card-yellow .stat-label {
        color: #222222;
    }
    .stat-card-yellow .stat-value {
        color: #1A1A1A;
    }
    .stat-card-dark {
        background-color: #1A1A1A !important;
        background-image: none !important;
        color: #FFFFF0;
    }
    .stat-card-dark .stat-label {
        color: #CCCCCC;
    }
    .stat-card-dark .stat-value {
        color: #FFFFF0;
    }
    .stat-card-dark p {
        color: #E0E0E0 !important;
    }

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
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 0.95;
        letter-spacing: -1.5px;
    }

    /* Rounded-Rectangle Neubrutalist Pill Badges */
    .badge-high {
        background-color: #1A1A1A;
        color: #FFFFFF;
        padding: 6px 16px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 999px !important;
        border: 2.5px solid #1A1A1A;
        box-shadow: 2px 2px 0px #1A1A1A;
    }
    .badge-medium {
        background-color: #E8C547;
        color: #1A1A1A;
        padding: 6px 16px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 999px !important;
        border: 2.5px solid #1A1A1A;
        box-shadow: 2px 2px 0px #1A1A1A;
    }
    .badge-low {
        background-color: #FFFFFF;
        color: #1A1A1A;
        padding: 6px 16px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 999px !important;
        border: 2.5px solid #1A1A1A;
        box-shadow: 2px 2px 0px #1A1A1A;
    }

    /* Torn Paper & Texture Edge */
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

    /* Section Divider */
    .brutalist-divider {
        border-top: 2.5px solid #1A1A1A;
        margin: 24px 0;
    }

    /* Streamlit Input Overrides */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        color: #1A1A1A !important;
        box-shadow: 3px 3px 0px #1A1A1A !important;
    }
    div[data-baseweb="select"] span {
        color: #1A1A1A !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 600 !important;
    }

    /* Table Styling */
    .stDataFrame {
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        box-shadow: 4px 4px 0px #1A1A1A !important;
    }
    .stTable {
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        box-shadow: 3px 3px 0px #1A1A1A !important;
    }
    .stTable table {
        font-family: 'Space Mono', monospace !important;
    }

    /* Buttons */
    div[data-testid="stButton"] button, div[data-testid="stButton"] button p {
        background-color: #E8C547 !important;
        color: #1A1A1A !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        padding: 8px 18px !important;
        box-shadow: 3px 3px 0px #1A1A1A !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translate(-1px, -1px);
        box-shadow: 4px 4px 0px #1A1A1A !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translate(1px, 1px);
        box-shadow: 1px 1px 0px #1A1A1A !important;
    }

    .stDownloadButton, .stDownloadButton > button, .stDownloadButton button {
        background-color: #DE6B35 !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        border: 2.5px solid #1A1A1A !important;
        border-radius: 4px !important;
        padding: 8px 18px !important;
        box-shadow: 3px 3px 0px #1A1A1A !important;
    }
    .stDownloadButton * {
        color: #FFFFFF !important;
    }

    .stTextArea textarea {
        border: 2px solid #1A1A1A !important;
        border-radius: 4px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
        background-color: #FAFAFA !important;
        color: #1A1A1A !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# FILE LOADING WITH VARIANT & MISSING FILE HANDLING
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

@st.cache_data
def load_dataset():
    filename = find_file([
        "student_dropout_risk_dataset.csv",
        "student_dropout_risk_dataset (1).csv",
        "dataset.csv"
    ])
    if not filename:
        raise FileNotFoundError("Could not find student_dropout_risk_dataset.csv in working directory.")
    
    return pd.read_csv(filename)


# Initialize Data & Models safely
try:
    artifacts = load_artifacts()
    df_raw = load_dataset()
except Exception as e:
    st.error(f"CRITICAL SYSTEM ERROR: {str(e)}")
    st.info("Please ensure student_dropout_risk_dataset.csv and dropout_prediction_artifacts.pkl are located in the root directory.")
    st.stop()


# =====================================================================
# PREPROCESSING & MODEL SAFEGUARD
# =====================================================================
try:
    model_obj = artifacts['model']
    scaler = artifacts['scaler']
    ordinal_encoder = artifacts['ordinal_encoder']
    dept_encoder = artifacts['dept_encoder']
    category_encoder = artifacts['category_encoder']
    target_encoder = artifacts['target_encoder']

    # Explicit Model Validation & Casting Check
    if not isinstance(model_obj, GradientBoostingClassifier):
        if not hasattr(model_obj, 'feature_importances_'):
            raise TypeError("Loaded model is not a GradientBoostingClassifier and lacks feature_importances_.")

    # Process dataset
    df = df_raw.copy()
    
    num_cols = [
        'year', 'attendance_percent', 'cgpa', 'internal_marks_avg',
        'assignment_submission_rate', 'backlogs', 'lms_engagement_score'
    ]
    
    # Check for required columns
    required_cols = num_cols + ['department', 'socio_economic_proxy', 'student_category']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Scale numeric features
    scaled_num = scaler.transform(df[num_cols])
    X_prep = pd.DataFrame(scaled_num, columns=num_cols)

    # Encode categorical features
    X_prep['department'] = dept_encoder.transform(df['department'])
    X_prep['socio_economic_proxy'] = ordinal_encoder.transform(df[['socio_economic_proxy']]).ravel()
    X_prep['student_category'] = category_encoder.transform(df['student_category'])

    # Align columns with model expectation
    feature_names = list(model_obj.feature_names_in_)
    X_prep = X_prep[feature_names]

    # Model Predictions
    preds = model_obj.predict(X_prep)
    probs = model_obj.predict_proba(X_prep)

    df['predicted_risk'] = target_encoder.inverse_transform(preds)

    # Calculate bounded Risk Score (0-100)
    high_class_idx = list(target_encoder.classes_).index('High')
    raw_scores = probs[:, high_class_idx] * 100.0
    df['risk_score'] = np.clip(np.nan_to_num(raw_scores, nan=0.0), 0.0, 100.0).round(1)

except Exception as e:
    st.error(f"DATA PREPROCESSING / MODEL ERROR: {str(e)}")
    st.stop()


# =====================================================================
# UI HELPERS (RETRO OS CHROME & NEUBRUTALISM)
# =====================================================================
def render_window_header(filename: str, status: str = "ONLINE"):
    st.markdown(f"""
    <div class="window-titlebar">
        <div class="window-dots">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
        </div>
        <div class="window-title">{filename}</div>
        <div class="window-status">{status}</div>
    </div>
    """, unsafe_allow_html=True)

def render_stat_card(label: str, value: str, variant: str = "white", subtext: str = None, extra_style: str = ""):
    subtext_html = f"<p style='font-size:0.75rem; margin-top:8px; font-weight:500; font-family: Inter, sans-serif;'>{subtext}</p>" if subtext else ""
    return f"""
    <div class="stat-card stat-card-{variant}" style="{extra_style}">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {subtext_html}
    </div>
    """

def get_risk_badge_html(risk_level):
    if risk_level == "High":
        return '<span class="badge-high">● HIGH RISK</span>'
    elif risk_level == "Medium":
        return '<span class="badge-medium">▲ MEDIUM RISK</span>'
    else:
        return '<span class="badge-low">■ LOW RISK</span>'

def calculate_top_factors_and_intervention(student_row, gb_model):
    # Extract feature importances strictly from GradientBoostingClassifier
    importances = gb_model.feature_importances_
    feat_names = list(gb_model.feature_names_in_)
    imp_dict = dict(zip(feat_names, importances))

    # Determine deviations / severity
    severity = {
        'attendance_percent': max(0, (75.0 - student_row.get('attendance_percent', 100)) / 75.0) if student_row.get('attendance_percent', 100) < 75 else 0,
        'cgpa': max(0, (6.0 - student_row.get('cgpa', 10)) / 6.0) if student_row.get('cgpa', 10) < 6.0 else 0,
        'backlogs': min(1.0, student_row.get('backlogs', 0) / 3.0) if student_row.get('backlogs', 0) > 0 else 0,
        'assignment_submission_rate': max(0, (70.0 - student_row.get('assignment_submission_rate', 100)) / 70.0) if student_row.get('assignment_submission_rate', 100) < 70 else 0,
        'internal_marks_avg': max(0, (50.0 - student_row.get('internal_marks_avg', 100)) / 50.0) if student_row.get('internal_marks_avg', 100) < 50 else 0,
        'lms_engagement_score': max(0, (50.0 - student_row.get('lms_engagement_score', 100)) / 50.0) if student_row.get('lms_engagement_score', 100) < 50 else 0
    }

    # Calculate weighted impact per feature
    impacts = {f: imp_dict.get(f, 0.01) * severity.get(f, 0) for f in severity}
    sorted_factors = sorted(impacts.items(), key=lambda x: x[1], reverse=True)

    top_3 = [f for f, imp in sorted_factors if imp > 0][:3]

    reasons = []
    for f in top_3:
        if f == 'attendance_percent':
            reasons.append(f"low attendance ({int(student_row['attendance_percent'])}%)")
        elif f == 'cgpa':
            reasons.append(f"low CGPA ({student_row['cgpa']})")
        elif f == 'backlogs':
            reasons.append(f"{int(student_row['backlogs'])} active backlog(s)")
        elif f == 'assignment_submission_rate':
            reasons.append(f"low assignment submission rate ({int(student_row['assignment_submission_rate'])}%)")
        elif f == 'internal_marks_avg':
            reasons.append(f"low internal exam average ({int(student_row['internal_marks_avg'])}/100)")
        elif f == 'lms_engagement_score':
            reasons.append(f"low online platform activity ({int(student_row['lms_engagement_score'])} pts)")

    if not reasons:
        explanation = "Academic performance metrics are within normal bounds."
        top_factor = "None"
    else:
        explanation = "High risk due to: " + ", ".join(reasons) + "."
        top_factor = top_3[0]

    # Required Specific Intervention Routing
    if top_factor == 'attendance_percent':
        intervention = "Recommend attendance counseling"
    elif top_factor == 'backlogs':
        intervention = "Recommend academic mentorship"
    elif top_factor == 'cgpa':
        intervention = "Recommend academic support program"
    elif top_factor == 'assignment_submission_rate':
        intervention = "Recommend faculty check-in"
    else:
        intervention = "Routine academic progress tracking"

    return explanation, intervention


# =====================================================================
# SIDEBAR NAVIGATION (MONOCHROME ICON-BASED)
# =====================================================================
st.sidebar.markdown("""
<div style='margin-bottom:12px;'>
    <div style='font-family: "Space Mono", monospace; font-size: 0.72rem; font-weight:700; color: #DE6B35; letter-spacing: 1px;'>SYS.CORE // SAARTHI</div>
    <h2 class='brutalist-title' style='margin-bottom:0px; font-size:1.6rem; letter-spacing: -0.5px;'>DROPOUT RISK</h2>
    <p style='font-family: "Space Mono", monospace; font-size:0.75rem; font-weight:600; color:#1A1A1A; margin-top:2px; margin-bottom:18px;'>EARLY WARNING ENGINE</p>
</div>
""", unsafe_allow_html=True)

NAV_OPTIONS = [
    "⌂ Overview",
    "☷ Student Risk Directory",
    "◈ Individual Student Profile",
    "⚙ Model Intelligence",
    "▤ HOD Report"
]

page = st.sidebar.radio(
    "SYSTEM SECTIONS",
    NAV_OPTIONS,
    index=0
)

st.sidebar.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-sys-box">
    <div style="color: #27C93F; font-weight:700; margin-bottom:4px;">● SYSTEM ONLINE</div>
    <div>KERNEL: <b>GRADIENT_BOOST</b></div>
    <div>METRICS: <b>CALIBRATED</b></div>
    <div>CHROME: <b>RETRO_OS_2.0</b></div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGE 1: OVERVIEW
# =====================================================================
if "Overview" in page:
    try:
        # Hero Window Container with Retro OS Header & Torn Paper Edge
        with st.container(border=True):
            render_window_header("saarthi_core.exe", "SYSTEM_READY")
            st.markdown("""
            <div style="padding: 6px 4px 10px 4px;">
                <div style="font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; color: #DE6B35; letter-spacing: 1px; margin-bottom: 4px;">// COHORT ANALYTICS & INTERVENTION PROTOCOL</div>
                <h1 class='brutalist-title' style='font-size: 2.8rem; margin: 0; line-height: 1.05;'>EARLY DROPOUT RISK PREDICTION</h1>
                <p style='font-size: 1rem; color: #1A1A1A; font-weight: 500; margin-top: 8px; margin-bottom: 0px;'>Machine learning cohort monitoring, risk vector decomposition, and automated student intervention system.</p>
                <div class="torn-paper-edge"></div>
            </div>
            """, unsafe_allow_html=True)

        total_students = len(df)
        high_risk_count = int((df['predicted_risk'] == 'High').sum())
        high_risk_pct = (high_risk_count / total_students * 100) if total_students > 0 else 0.0

        # Rhythmic Alternating Stat Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(render_stat_card(
                "TOTAL STUDENTS", 
                str(total_students), 
                variant="white",
                subtext="Registered students tracked in current active cohort"
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(render_stat_card(
                "HIGH RISK %", 
                f"{high_risk_pct:.1f}%", 
                variant="orange",
                subtext="Percentage flagged for immediate counseling intervention"
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(render_stat_card(
                "AT RISK COUNT", 
                str(high_risk_count), 
                variant="yellow",
                subtext="Absolute student count requiring early academic support"
            ), unsafe_allow_html=True)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Card 1: Risk Distribution
        with st.container(border=True):
            render_window_header("risk_distribution.exe", "LIVE")
            st.markdown("<h3 class='brutalist-title'>RISK DISTRIBUTION</h3>", unsafe_allow_html=True)

            counts = df['predicted_risk'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)

            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')

            # Chart styling with White, Mustard Yellow, and Dark Charcoal
            bars = ax.barh(counts.index, counts.values, height=0.55)
            bars[0].set_color('#FFFFFF')
            bars[0].set_edgecolor('#1A1A1A')
            bars[0].set_linewidth(2.5)
        
            bars[1].set_color('#E8C547')
            bars[1].set_edgecolor('#1A1A1A')
            bars[1].set_linewidth(2.5)
        
            bars[2].set_color('#DE6B35')
            bars[2].set_edgecolor('#1A1A1A')
            bars[2].set_linewidth(2.5)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#1A1A1A')
            ax.spines['left'].set_color('#1A1A1A')
            ax.spines['bottom'].set_linewidth(2.5)
            ax.spines['left'].set_linewidth(2.5)
        
            ax.tick_params(colors='#1A1A1A', labelsize=10, width=2)
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_family('sans-serif')

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 5, bar.get_y() + bar.get_height()/2, f'{int(w)}', 
                        va='center', ha='left', color='#1A1A1A', fontweight='bold', fontsize=11)

            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Card 2: Department-wise Risk Heatmap
        with st.container(border=True):
            render_window_header("department_heatmap.exe", "AGGREGATED")
            st.markdown("<h3 class='brutalist-title'>DEPARTMENT-WISE RISK HEATMAP</h3>", unsafe_allow_html=True)

            from matplotlib.colors import LinearSegmentedColormap

            heat_data = pd.crosstab(df['department'], df['predicted_risk'])
            for level in ['Low', 'Medium', 'High']:
                if level not in heat_data.columns:
                    heat_data[level] = 0
            heat_data = heat_data[['Low', 'Medium', 'High']]

            brutal_cmap = LinearSegmentedColormap.from_list(
                'brutal', ['#FFFFFF', '#E8C547', '#DE6B35', '#1A1A1A']
            )

            fig2, ax2 = plt.subplots(figsize=(8, max(3, 0.5 * len(heat_data))))
            fig2.patch.set_facecolor('#FFFFFF')
            ax2.set_facecolor('#FFFFFF')

            im = ax2.imshow(heat_data.values, cmap=brutal_cmap, aspect='auto')

            ax2.set_xticks(range(len(heat_data.columns)))
            ax2.set_xticklabels(heat_data.columns, fontweight='bold', fontsize=10)
            ax2.set_yticks(range(len(heat_data.index)))
            ax2.set_yticklabels(heat_data.index, fontweight='bold', fontsize=10)

            for i in range(len(heat_data.index)):
                for j in range(len(heat_data.columns)):
                    val = heat_data.values[i, j]
                    text_color = '#FFFFFF' if val > heat_data.values.max() * 0.55 else '#1A1A1A'
                    ax2.text(j, i, str(val), ha='center', va='center',
                              color=text_color, fontweight='bold', fontsize=11)

            for spine in ax2.spines.values():
                spine.set_color('#1A1A1A')
                spine.set_linewidth(2.5)

            ax2.set_xticks([x - 0.5 for x in range(1, len(heat_data.columns))], minor=True)
            ax2.set_yticks([y - 0.5 for y in range(1, len(heat_data.index))], minor=True)
            ax2.grid(which='minor', color='#1A1A1A', linewidth=2)
            ax2.tick_params(which='minor', bottom=False, left=False)

            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Card 3: Semester Risk Trend
        with st.container(border=True):
            render_window_header("semester_trend.exe", "HISTORICAL")
            st.markdown("<h3 class='brutalist-title'>SEMESTER RISK TREND</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.85rem; color:#1A1A1A; font-family:\"Space Mono\", monospace; font-weight:600; margin-bottom:12px;'>AVG RISK SCORE ACROSS ACADEMIC YEARS // COHORT ANALYSIS</p>", unsafe_allow_html=True)

            risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
            df['risk_numeric'] = df['predicted_risk'].map(risk_map)
            trend_data = df.groupby('year')['risk_numeric'].mean().sort_index()

            fig3, ax3 = plt.subplots(figsize=(10, 3.5))
            fig3.patch.set_facecolor('#FFFFFF')
            ax3.set_facecolor('#FFFFFF')

            ax3.plot(trend_data.index, trend_data.values, color='#1A1A1A', linewidth=3.5,
                     marker='o', markersize=10, markerfacecolor='#E8C547', markeredgecolor='#1A1A1A', markeredgewidth=2.5)

            ax3.set_xticks(trend_data.index)
            ax3.set_xticklabels([f'Year {int(y)}' for y in trend_data.index], fontweight='bold', fontsize=10)
            ax3.set_ylim(-0.2, 2.2)
            ax3.set_yticks([0, 1, 2])
            ax3.set_yticklabels(['Low', 'Medium', 'High'], fontweight='bold', fontsize=10)

            for x, y in zip(trend_data.index, trend_data.values):
                ax3.text(x, y + 0.15, f'{y:.2f}', ha='center', fontweight='bold', fontsize=10, color='#1A1A1A')

            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_color('#1A1A1A')
            ax3.spines['left'].set_color('#1A1A1A')
            ax3.spines['bottom'].set_linewidth(2.5)
            ax3.spines['left'].set_linewidth(2.5)
            ax3.tick_params(colors='#1A1A1A', width=2)
            ax3.grid(axis='y', color='#E5E5E5', linewidth=1, linestyle='--')

            plt.tight_layout()
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error rendering Overview page: {str(e)}")


# =====================================================================
# PAGE 2: STUDENT RISK DIRECTORY
# =====================================================================
elif "Risk Directory" in page:
    try:
        # Directory Header & Filter Container
        with st.container(border=True):
            render_window_header("student_directory.exe", "DATABASE_QUERY")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>STUDENT RISK DIRECTORY</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 20px;'>FILTERABLE ACADEMIC & BEHAVIORAL REGISTER</p>", unsafe_allow_html=True)

            col_dep, col_risk = st.columns(2)
            
            actual_depts = ["ALL"] + sorted(list(df['department'].unique()))
            actual_risks = ["ALL"] + sorted(list(df['predicted_risk'].unique()))

            with col_dep:
                sel_dept = st.selectbox("FILTER BY DEPARTMENT", actual_depts)
            with col_risk:
                sel_risk = st.selectbox("FILTER BY RISK LEVEL", actual_risks)

        filtered = df.copy()
        if sel_dept != "ALL":
            filtered = filtered[filtered['department'] == sel_dept]
        if sel_risk != "ALL":
            filtered = filtered[filtered['predicted_risk'] == sel_risk]

        filtered = filtered.sort_values(by='risk_score', ascending=False)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        # Table Container
        with st.container(border=True):
            render_window_header("risk_registry.exe", "RECORDS_LOADED")
            display_df = filtered[[
                'student_id', 'name', 'department', 'year', 'cgpa',
                'attendance_percent', 'backlogs', 'predicted_risk', 'risk_score'
            ]].copy()

            display_df.columns = [
                'ID', 'NAME', 'DEPARTMENT', 'YEAR', 'CGPA',
                'ATTENDANCE %', 'BACKLOGS', 'RISK LEVEL', 'SCORE'
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "SCORE": st.column_config.NumberColumn("SCORE", format="%.1f"),
                    "RISK LEVEL": st.column_config.TextColumn("RISK LEVEL")
                }
            )
            st.markdown(f"""
            <div style='font-family: "Space Mono", monospace; font-size:0.8rem; font-weight:700; margin-top:10px; color:#1A1A1A;'>
                STATUS: SHOWING {len(display_df)} OF {len(df)} RECORDS // SORTED BY RISK_SCORE DESC
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering Student Risk Directory: {str(e)}")


# =====================================================================
# PAGE 3: INDIVIDUAL STUDENT PROFILE
# =====================================================================
elif "Student Profile" in page:
    try:
        with st.container(border=True):
            render_window_header("profile_query.exe", "STUDENT_INDEX")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>INDIVIDUAL STUDENT PROFILE</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 16px;'>DEEP-DIVE DIAGNOSIS & INTERVENTION PROTOCOL</p>", unsafe_allow_html=True)

            actual_student_names = sorted(list(df['name'].unique()))
            selected_student_name = st.selectbox("SELECT STUDENT NAME", actual_student_names)

        student = df[df['name'] == selected_student_name].iloc[0]

        # Get risk explanation & intervention using strictly GradientBoostingClassifier
        explanation, intervention = calculate_top_factors_and_intervention(student, model_obj)

        c1, c2 = st.columns([1, 2])

        with c1:
            with st.container(border=True):
                render_window_header("risk_eval.exe", "INFERENCE")
                st.markdown("<div class='stat-label'>SCORE (0-100)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='stat-value' style='font-size: 4.8rem; margin: 8px 0; color:#1A1A1A;'>{student['risk_score']}</div>", unsafe_allow_html=True)
                st.markdown(get_risk_badge_html(student['predicted_risk']), unsafe_allow_html=True)
                st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='font-family: "Space Mono", monospace; font-size:0.85rem; line-height: 1.8;'>
                    <div><b>PID:</b> {student['student_id']}</div>
                    <div><b>DEPT:</b> {student['department']}</div>
                    <div><b>YEAR:</b> Year {int(student['year'])}</div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            with st.container(border=True):
                render_window_header("risk_diagnosis.exe", "SHAP_ACTIVE")
                st.markdown("<h3 class='brutalist-title'>PRIMARY RISK CAUSES</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 1.05rem; font-weight:500; color:#1A1A1A;'>{explanation}</p>", unsafe_allow_html=True)
            
                st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
            
                st.markdown("<h3 class='brutalist-title'>RECOMMENDED INTERVENTION</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background-color:#E8C547; border:2.5px solid #1A1A1A; border-radius:4px; padding:14px 18px; font-weight:700; color:#1A1A1A; font-family: "Space Mono", monospace; box-shadow: 3px 3px 0px #1A1A1A; font-size: 0.95rem;'>
                    {intervention.upper()}
                </div>
                """, unsafe_allow_html=True)

        # Academic Metrics Grid with Alternating Colors
        with st.container(border=True):
            render_window_header("academic_metrics.exe", "TELEMETRY")
            st.markdown("<h3 class='brutalist-title'>ACADEMIC METRICS GRID</h3>", unsafe_allow_html=True)
        
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(render_stat_card("CGPA", f"{student['cgpa']}", variant="white"), unsafe_allow_html=True)
            with m2:
                st.markdown(render_stat_card("ATTENDANCE", f"{student['attendance_percent']}%", variant="yellow"), unsafe_allow_html=True)
            with m3:
                st.markdown(render_stat_card("BACKLOGS", f"{student['backlogs']}", variant="orange"), unsafe_allow_html=True)
            with m4:
                st.markdown(render_stat_card("ASSIGNMENTS", f"{student['assignment_submission_rate']}%", variant="dark"), unsafe_allow_html=True)

        # Cohort Benchmark
        with st.container(border=True):
            render_window_header("cohort_benchmark.exe", "COMPARATIVE")
            st.markdown("<h3 class='brutalist-title'>RISK VS. YEAR COHORT AVERAGE</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:0.85rem; color:#1A1A1A; font-family:\"Space Mono\", monospace; font-weight:600; margin-bottom:12px;'>METRIC COMPARISON: {student['name'].upper()} VS YEAR {int(student['year'])} AVERAGE</p>", unsafe_allow_html=True)

            risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
            df['risk_numeric'] = df['predicted_risk'].map(risk_map)
            trend_data = df.groupby('year')['risk_numeric'].mean().sort_index()
            student_risk_numeric = risk_map[student['predicted_risk']]

            fig4, ax4 = plt.subplots(figsize=(10, 3.5))
            fig4.patch.set_facecolor('#FFFFFF')
            ax4.set_facecolor('#FFFFFF')

            ax4.plot(trend_data.index, trend_data.values, color='#1A1A1A', linewidth=3.5,
                     marker='o', markersize=8, markerfacecolor='#FFFFFF', markeredgecolor='#1A1A1A', markeredgewidth=2.5,
                     label='Year Average', zorder=2)

            ax4.scatter([student['year']], [student_risk_numeric], color='#DE6B35', edgecolor='#1A1A1A',
                        linewidth=2.5, s=260, zorder=3, label=student['name'])

            ax4.set_xticks(trend_data.index)
            ax4.set_xticklabels([f'Year {int(y)}' for y in trend_data.index], fontweight='bold', fontsize=10)
            ax4.set_ylim(-0.2, 2.2)
            ax4.set_yticks([0, 1, 2])
            ax4.set_yticklabels(['Low', 'Medium', 'High'], fontweight='bold', fontsize=10)

            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['bottom'].set_color('#1A1A1A')
            ax4.spines['left'].set_color('#1A1A1A')
            ax4.spines['bottom'].set_linewidth(2.5)
            ax4.spines['left'].set_linewidth(2.5)
            ax4.tick_params(colors='#1A1A1A', width=2)
            ax4.grid(axis='y', color='#E5E5E5', linewidth=1, linestyle='--')
            legend = ax4.legend(loc='upper left', frameon=True, edgecolor='#1A1A1A', fontsize=9,
                                 fancybox=False, borderpad=0.8, handletextpad=0.6)
            legend.get_frame().set_linewidth(2.5)
            legend.get_frame().set_facecolor('#FFFFFF')

            plt.tight_layout()
            st.pyplot(fig4)

        # Email Outreach Simulation
        with st.container(border=True):
            render_window_header("email_dispatcher.exe", "SIMULATION")
            st.markdown("<h3 class='brutalist-title'>EARLY WARNING EMAIL SIMULATION</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.85rem; color:#555555; font-family:\"Space Mono\", monospace; font-weight:600; margin-bottom:12px;'>SIMULATED INTERVENTION DISPATCH // WORKFLOW DEMO</p>", unsafe_allow_html=True)

            if st.button("PREVIEW EARLY WARNING EMAIL"):
                email_subject = f"Checking In — Support Available for {student['name']}"
                email_body = f"""Dear {student['name']},

We hope this message finds you well. As part of our academic support program, we periodically review student progress to ensure everyone has the resources they need to succeed.

Our records show some areas where additional support could help you this semester, particularly around: {explanation}

This is not a disciplinary notice — it's an invitation. Our academic mentors are available to discuss study strategies, attendance flexibility, or any personal circumstances affecting your coursework.

Please reach out to the {student['department']} Academic Support Office at your convenience, or reply to this email to schedule a short meeting.

We're here to help you succeed.

Warm regards,
Academic Support Office
{student['department']} Department
"""
                st.markdown(f"""
                <div style='font-family: "Space Mono", monospace; font-size: 0.85rem; line-height: 1.8; margin-bottom: 12px;'>
                    <div><b>TO:</b> {student['name'].lower().replace(' ', '.')}@college.edu</div>
                    <div><b>FROM:</b> academic.support@college.edu</div>
                    <div><b>SUBJECT:</b> {email_subject}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
                st.text_area("Email Body", value=email_body, height=280, label_visibility="collapsed")

    except Exception as e:
        st.error(f"Error rendering Individual Student Profile: {str(e)}")


# =====================================================================
# PAGE 4: MODEL INTELLIGENCE
# =====================================================================
elif "Model Intelligence" in page:
    try:
        with st.container(border=True):
            render_window_header("model_telemetry.exe", "GRADIENT_BOOST")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>MODEL INTELLIGENCE</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 20px;'>GRADIENT BOOSTING METRICS, FEATURE WEIGHTS & CONFUSION MATRIX</p>", unsafe_allow_html=True)

        eval_metrics = artifacts['eval_metrics']
        acc = eval_metrics['accuracy']
        prec = eval_metrics['precision']
        rec = eval_metrics['recall']

        # Alternating Stat Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(render_stat_card(
                "ACCURACY", 
                f"{acc*100:.1f}%", 
                variant="white",
                subtext="Overall percentage of correct risk predictions across all students."
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(render_stat_card(
                "PRECISION", 
                f"{prec*100:.1f}%", 
                variant="orange",
                subtext="Ability of the model to avoid false-positive risk flags."
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(render_stat_card(
                "RECALL", 
                f"{rec*100:.1f}%", 
                variant="yellow",
                subtext="Ability of the model to catch all actual at-risk students."
            ), unsafe_allow_html=True)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        col_feat, col_cm = st.columns([1.2, 0.8])

        with col_feat:
            with st.container(border=True):
                render_window_header("feature_weights.exe", "IMPORTANCES")
                st.markdown("<h3 class='brutalist-title'>FEATURE IMPORTANCE</h3>", unsafe_allow_html=True)

                # Access directly from GradientBoostingClassifier
                importances = model_obj.feature_importances_
                feat_series = pd.Series(importances, index=model_obj.feature_names_in_).sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')

                bars = ax.barh(feat_series.index, feat_series.values, color='#1A1A1A', height=0.6, edgecolor='#1A1A1A', linewidth=2)
            
                # Highlight top feature in Warm Burnt Orange
                bars[-1].set_color('#DE6B35')
                bars[-1].set_edgecolor('#1A1A1A')
                if len(bars) > 1:
                    bars[-2].set_color('#E8C547')
                    bars[-2].set_edgecolor('#1A1A1A')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#1A1A1A')
                ax.spines['left'].set_color('#1A1A1A')
                ax.spines['bottom'].set_linewidth(2.5)
                ax.spines['left'].set_linewidth(2.5)
                ax.tick_params(colors='#1A1A1A', labelsize=9, width=2)

                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')

                plt.tight_layout()
                st.pyplot(fig)

        with col_cm:
            with st.container(border=True):
                render_window_header("confusion_matrix.exe", "TEST_EVAL")
                st.markdown("<h3 class='brutalist-title'>CONFUSION MATRIX</h3>", unsafe_allow_html=True)

                labels = eval_metrics['cm_labels']
                cm = eval_metrics['confusion_matrix']
                cm_df = pd.DataFrame(cm, index=[f"ACTUAL {l}" for l in labels], columns=[f"PRED {l}" for l in labels])

                st.table(cm_df)

    except Exception as e:
        st.error(f"Error rendering Model Intelligence: {str(e)}")


# =====================================================================
# PAGE 5: HOD REPORT
# =====================================================================
elif "HOD Report" in page:
    try:
        with st.container(border=True):
            render_window_header("hod_audit.exe", "FACULTY_PORTAL")
            st.markdown("<h1 class='brutalist-title' style='margin-bottom:4px;'>HOD SUMMARY REPORT</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-family:\"Space Mono\", monospace; color: #555555; font-weight: 600; margin-bottom: 18px;'>DEPARTMENT-LEVEL AUDIT & FACULTY ESCALATION REGISTER</p>", unsafe_allow_html=True)

            dept_options = ["All Departments"] + sorted(df['department'].unique().tolist())
            selected_dept = st.selectbox("SELECT DEPARTMENT", dept_options)

        if selected_dept == "All Departments":
            report_df = df.copy()
        else:
            report_df = df[df['department'] == selected_dept].copy()

        total = len(report_df)
        high_count = int((report_df['predicted_risk'] == 'High').sum())
        med_count = int((report_df['predicted_risk'] == 'Medium').sum())
        low_count = int((report_df['predicted_risk'] == 'Low').sum())

        # 4 Poster Stat Cards with Color Rhythm
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(render_stat_card("TOTAL STUDENTS", str(total), variant="white"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_stat_card("HIGH RISK", str(high_count), variant="orange"), unsafe_allow_html=True)
        with c3:
            st.markdown(render_stat_card("MEDIUM RISK", str(med_count), variant="yellow"), unsafe_allow_html=True)
        with c4:
            st.markdown(render_stat_card("LOW RISK", str(low_count), variant="dark"), unsafe_allow_html=True)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            render_window_header("risk_breakdown.exe", "DISTRIBUTION")
            st.markdown(f"<h3 class='brutalist-title'>RISK BREAKDOWN — {selected_dept.upper()}</h3>", unsafe_allow_html=True)

            counts = report_df['predicted_risk'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
            fig5, ax5 = plt.subplots(figsize=(10, 3))
            fig5.patch.set_facecolor('#FFFFFF')
            ax5.set_facecolor('#FFFFFF')
            bars5 = ax5.barh(counts.index, counts.values, height=0.55)
            bars5[0].set_color('#FFFFFF'); bars5[0].set_edgecolor('#1A1A1A'); bars5[0].set_linewidth(2.5)
            bars5[1].set_color('#E8C547'); bars5[1].set_edgecolor('#1A1A1A'); bars5[1].set_linewidth(2.5)
            bars5[2].set_color('#DE6B35'); bars5[2].set_edgecolor('#1A1A1A'); bars5[2].set_linewidth(2.5)
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['bottom'].set_color('#1A1A1A'); ax5.spines['left'].set_color('#1A1A1A')
            ax5.spines['bottom'].set_linewidth(2.5); ax5.spines['left'].set_linewidth(2.5)
            ax5.tick_params(colors='#1A1A1A', labelsize=10, width=2)
            for label in ax5.get_yticklabels():
                label.set_fontweight('bold')
            for bar in bars5:
                w = bar.get_width()
                ax5.text(w + 2, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', ha='left', color='#1A1A1A', fontweight='bold', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig5)

        with st.container(border=True):
            render_window_header("consultation_roster.exe", "PRIORITY_LIST")
            st.markdown("<h3 class='brutalist-title'>STUDENTS REQUIRING CONSULTATION</h3>", unsafe_allow_html=True)

            high_risk_students = report_df[report_df['predicted_risk'] == 'High'].sort_values('risk_score', ascending=False)

            if len(high_risk_students) == 0:
                st.markdown("<p style='font-family:\"Space Mono\", monospace; font-weight:700; color:#1A1A1A;'>No High-risk students in this selection. [STATUS: NOMINAL]</p>", unsafe_allow_html=True)
            else:
                consult_rows = []
                for _, srow in high_risk_students.iterrows():
                    expl, _ = calculate_top_factors_and_intervention(srow, model_obj)
                    consult_rows.append({
                        "Student": srow['name'],
                        "ID": srow['student_id'],
                        "Year": int(srow['year']),
                        "Risk Score": srow['risk_score'],
                        "Primary Concern": expl
                    })
                consult_df = pd.DataFrame(consult_rows)
                st.dataframe(consult_df, use_container_width=True, hide_index=True)

        with st.container(border=True):
            render_window_header("export_engine.exe", "PDF_RENDERER")
            st.markdown("<h3 class='brutalist-title'>DOWNLOAD REPORT</h3>", unsafe_allow_html=True)

            def generate_pdf_report(dept_name, total, high_c, med_c, low_c, students_df):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 18)
                pdf.cell(0, 12, "EARLY DROPOUT RISK - HOD SUMMARY REPORT", ln=True)
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 8, f"Department: {dept_name}", ln=True)
                pdf.ln(4)

                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "OVERVIEW", ln=True)
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 7, f"Total Students: {total}", ln=True)
                pdf.cell(0, 7, f"High Risk: {high_c}   Medium Risk: {med_c}   Low Risk: {low_c}", ln=True)
                pdf.ln(6)

                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "STUDENTS REQUIRING CONSULTATION", ln=True)
                pdf.set_font("Helvetica", "", 10)

                if len(students_df) == 0:
                    pdf.cell(0, 7, "No High-risk students in this selection.", ln=True)
                else:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(45, 7, "Student", border=1)
                    pdf.cell(25, 7, "ID", border=1)
                    pdf.cell(15, 7, "Year", border=1)
                    pdf.cell(25, 7, "Score", border=1)
                    pdf.cell(80, 7, "Primary Concern", border=1, ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    for _, r in students_df.iterrows():
                        pdf.cell(45, 7, str(r["Student"])[:22], border=1)
                        pdf.cell(25, 7, str(r["ID"]), border=1)
                        pdf.cell(15, 7, str(r["Year"]), border=1)
                        pdf.cell(25, 7, str(r["Risk Score"]), border=1)
                        pdf.cell(80, 7, str(r["Primary Concern"])[:45], border=1, ln=True)

                return bytes(pdf.output())

            pdf_bytes = generate_pdf_report(
                selected_dept, total, high_count, med_count, low_count,
                consult_df if len(high_risk_students) > 0 else pd.DataFrame(columns=["Student","ID","Year","Risk Score","Primary Concern"])
            )

            st.download_button(
                label="DOWNLOAD REPORT",
                data=pdf_bytes,
                file_name=f"HOD_Report_{selected_dept.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Error rendering HOD Report: {str(e)}")
