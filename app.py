import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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
# SOFT BRUTALISM CSS (STRICT ZERO-ROUNDING & SHARP BORDERS)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&display=swap');

    /* Reset Streamlit Defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* Global Typography & Colors */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFF0 !important;
        color: #1A1A1A !important;
        font-family: 'Inter', sans-serif !alignment: left;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFF0 !important;
        border-right: 2px solid #1A1A1A !important;
    }

    /* Sidebar Radio Button Text Fix */
[data-testid="stSidebar"] .stRadio label {
    color: #1A1A1A !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio label p {
    color: #1A1A1A !important;
}
[data-testid="stSidebar"] .stRadio div {
    color: #1A1A1A !important;
}

    /* Brutalist Headings */
    h1, h2, h3, h4, .brutalist-title {
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        color: #1A1A1A !important;
    }

    /* Brutalist Cards */
    .brutalist-card {
        background-color: #FFFFFF;
        border: 2px solid #1A1A1A;
        border-radius: 0px !important;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: none !important;
    }

    /* Stat Cards */
    .stat-card {
        background-color: #FFFFFF;
        border: 2px solid #1A1A1A;
        border-radius: 0px !important;
        padding: 16px 20px;
        text-align: left;
    }
    .stat-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #1A1A1A;
        margin-bottom: 4px;
    }
    .stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1A1A1A;
        line-height: 1.0;
    }

    /* Soft Brutalist Badges */
    .badge-high {
        background-color: #1A1A1A;
        color: #FFFFFF;
        padding: 4px 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 0px !important;
        border: 2px solid #1A1A1A;
    }
    .badge-medium {
        background-color: #E8C547;
        color: #1A1A1A;
        padding: 4px 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 0px !important;
        border: 2px solid #1A1A1A;
    }
    .badge-low {
        background-color: #FFFFFF;
        color: #1A1A1A;
        padding: 4px 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        display: inline-block;
        border-radius: 0px !important;
        border: 2px solid #1A1A1A;
    }

    /* Section Divider */
    .brutalist-divider {
        border-top: 2px solid #1A1A1A;
        margin: 24px 0;
    }

    /* Streamlit Input Overrides */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #1A1A1A !important;
        border-radius: 0px !important;
        color: #1A1A1A !important;
    }
    
    div[data-baseweb="select"] span {
        color: #1A1A1A !important;
        font-weight: 500 !important;
    }

    /* Table Styling */
    .stDataFrame {
        border: 2px solid #1A1A1A !important;
        border-radius: 0px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #E8C547 !important;
        color: #1A1A1A !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        border: 2px solid #1A1A1A !important;
        border-radius: 0px !important;
        padding: 8px 16px !important;
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
    target_encoder = artifacts['target_encoder']

    # Explicit Model Validation & Casting Check
    if not isinstance(model_obj, GradientBoostingClassifier):
        # Force strict check while allowing subclassing
        if not hasattr(model_obj, 'feature_importances_'):
            raise TypeError("Loaded model is not a GradientBoostingClassifier and lacks feature_importances_.")

    # Process dataset
    df = df_raw.copy()
    
    num_cols = [
        'year', 'attendance_percent', 'cgpa', 'internal_marks_avg',
        'assignment_submission_rate', 'backlogs', 'lms_engagement_score'
    ]
    
    # Check for required columns
    required_cols = num_cols + ['department', 'socio_economic_proxy']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Scale numeric features
    scaled_num = scaler.transform(df[num_cols])
    X_prep = pd.DataFrame(scaled_num, columns=num_cols)

    # Encode categorical features
    X_prep['department'] = dept_encoder.transform(df['department'])
    X_prep['socio_economic_proxy'] = ordinal_encoder.transform(df[['socio_economic_proxy']]).ravel()

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
# HELPER FUNCTIONS
# =====================================================================
def get_risk_badge_html(risk_level):
    if risk_level == "High":
        return '<span class="badge-high">HIGH RISK</span>'
    elif risk_level == "Medium":
        return '<span class="badge-medium">MEDIUM RISK</span>'
    else:
        return '<span class="badge-low">LOW RISK</span>'

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
# SIDEBAR NAVIGATION
# =====================================================================
st.sidebar.markdown("<h2 class='brutalist-title' style='margin-bottom:0px;'>DROPOUT RISK</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:0.8rem; font-weight:600; color:#1A1A1A; margin-bottom:20px;'>EARLY WARNING SYSTEM</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Student Risk Directory", "Individual Student Profile", "Model Intelligence"],
    index=0
)

st.sidebar.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='font-size: 0.75rem; font-weight:600;'>
    SYSTEM: ACTIVE<br>
    MODEL: GRADIENT BOOSTING<br>
    DESIGN: SOFT BRUTALISM
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGE 1: OVERVIEW
# =====================================================================
if page == "Overview":
    try:
        st.markdown("<h1 class='brutalist-title' style='font-size: 2.5rem;'>EARLY DROPOUT RISK PREDICTION</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1rem; color: #1A1A1A; font-weight: 500; margin-bottom: 24px;'>Cohorts analytics and automated risk assessment overview.</p>", unsafe_allow_html=True)

        total_students = len(df)
        high_risk_count = int((df['predicted_risk'] == 'High').sum())
        high_risk_pct = (high_risk_count / total_students * 100) if total_students > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">TOTAL STUDENTS</div>
                <div class="stat-value">{total_students}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">HIGH RISK %</div>
                <div class="stat-value">{high_risk_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">AT RISK COUNT</div>
                <div class="stat-value">{high_risk_count}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        st.markdown("<div class='brutalist-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='brutalist-title'>RISK DISTRIBUTION</h3>", unsafe_allow_html=True)

        counts = df['predicted_risk'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        # Soft Brutalism chart styling (Black, Yellow, White with Outlines)
        bars = ax.barh(counts.index, counts.values, height=0.55)
        bars[0].set_color('#FFFFFF')
        bars[0].set_edgecolor('#1A1A1A')
        bars[0].set_linewidth(2)
        
        bars[1].set_color('#E8C547')
        bars[1].set_edgecolor('#1A1A1A')
        bars[1].set_linewidth(2)
        
        bars[2].set_color('#1A1A1A')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1A1A1A')
        ax.spines['left'].set_color('#1A1A1A')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
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
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering Overview page: {str(e)}")


# =====================================================================
# PAGE 2: STUDENT RISK DIRECTORY
# =====================================================================
elif page == "Student Risk Directory":
    try:
        st.markdown("<h1 class='brutalist-title'>STUDENT RISK DIRECTORY</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1rem; color: #1A1A1A; font-weight: 500; margin-bottom: 20px;'>Filterable student performance register.</p>", unsafe_allow_html=True)

        col_dep, col_risk = st.columns(2)
        
        # Populate dropdowns strictly from actual data values
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

        # Format dataframe for display
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
        st.markdown(f"<p style='font-size:0.85rem; font-weight:600;'>SHOWING {len(display_df)} OF {len(df)} RECORDS</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering Student Risk Directory: {str(e)}")


# =====================================================================
# PAGE 3: INDIVIDUAL STUDENT PROFILE
# =====================================================================
elif page == "Individual Student Profile":
    try:
        st.markdown("<h1 class='brutalist-title'>INDIVIDUAL STUDENT PROFILE</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1rem; color: #1A1A1A; font-weight: 500; margin-bottom: 20px;'>Deep-dive diagnosis and intervention protocol.</p>", unsafe_allow_html=True)

        actual_student_names = sorted(list(df['name'].unique()))
        selected_student_name = st.selectbox("SELECT STUDENT NAME", actual_student_names)

        student = df[df['name'] == selected_student_name].iloc[0]

        # Get risk explanation & intervention using strictly GradientBoostingClassifier
        explanation, intervention = calculate_top_factors_and_intervention(student, model_obj)

        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown("<div class='brutalist-card'>", unsafe_allow_html=True)
            st.markdown("<div class='stat-label'>SCORE (0-100)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-value' style='font-size: 4rem; margin: 8px 0;'>{student['risk_score']}</div>", unsafe_allow_html=True)
            st.markdown(get_risk_badge_html(student['predicted_risk']), unsafe_allow_html=True)
            st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
            st.markdown(f"**ID:** {student['student_id']}")
            st.markdown(f"**DEPARTMENT:** {student['department']}")
            st.markdown(f"**YEAR:** {student['year']}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='brutalist-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='brutalist-title'>PRIMARY RISK CAUSES</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 1.05rem; font-weight:500; color:#1A1A1A;'>{explanation}</p>", unsafe_allow_html=True)
            
            st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)
            
            st.markdown("<h3 class='brutalist-title'>RECOMMENDED INTERVENTION</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background-color:#E8C547; border:2px solid #1A1A1A; padding:12px; font-weight:700; color:#1A1A1A;'>
                {intervention.upper()}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='brutalist-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='brutalist-title'>ACADEMIC METRICS GRID</h3>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='stat-card'><div class='stat-label'>CGPA</div><div class='stat-value'>{student['cgpa']}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='stat-card'><div class='stat-label'>ATTENDANCE</div><div class='stat-value'>{student['attendance_percent']}%</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='stat-card'><div class='stat-label'>BACKLOGS</div><div class='stat-value'>{student['backlogs']}</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='stat-card'><div class='stat-label'>ASSIGNMENTS</div><div class='stat-value'>{student['assignment_submission_rate']}%</div></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering Individual Student Profile: {str(e)}")


# =====================================================================
# PAGE 4: MODEL INTELLIGENCE
# =====================================================================
elif page == "Model Intelligence":
    try:
        st.markdown("<h1 class='brutalist-title'>MODEL INTELLIGENCE</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1rem; color: #1A1A1A; font-weight: 500; margin-bottom: 20px;'>Gradient Boosting model metrics, weights, and evaluation table.</p>", unsafe_allow_html=True)

        y_true = df['dropout_risk_label']
        y_pred = df['predicted_risk']

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">ACCURACY</div>
                <div class="stat-value">{acc*100:.1f}%</div>
                <p style="font-size:0.75rem; margin-top:6px; font-weight:500;">Overall percentage of correct risk predictions across all students.</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">PRECISION</div>
                <div class="stat-value">{prec*100:.1f}%</div>
                <p style="font-size:0.75rem; margin-top:6px; font-weight:500;">Ability of the model to avoid false-positive risk flags.</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">RECALL</div>
                <div class="stat-value">{rec*100:.1f}%</div>
                <p style="font-size:0.75rem; margin-top:6px; font-weight:500;">Ability of the model to catch all actual at-risk students.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='brutalist-divider'></div>", unsafe_allow_html=True)

        col_feat, col_cm = st.columns([1.2, 0.8])

        with col_feat:
            st.markdown("<div class='brutalist-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='brutalist-title'>FEATURE IMPORTANCE</h3>", unsafe_allow_html=True)

            # Access directly from GradientBoostingClassifier
            importances = model_obj.feature_importances_
            feat_series = pd.Series(importances, index=model_obj.feature_names_in_).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')

            bars = ax.barh(feat_series.index, feat_series.values, color='#1A1A1A', height=0.6, edgecolor='#1A1A1A', linewidth=2)
            
            # Highlight top feature in Yellow
            bars[-1].set_color('#E8C547')
            bars[-1].set_edgecolor('#1A1A1A')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#1A1A1A')
            ax.spines['left'].set_color('#1A1A1A')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.tick_params(colors='#1A1A1A', labelsize=9, width=2)

            for label in ax.get_yticklabels():
                label.set_fontweight('bold')

            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_cm:
            st.markdown("<div class='brutalist-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='brutalist-title'>CONFUSION MATRIX</h3>", unsafe_allow_html=True)

            labels = ['High', 'Medium', 'Low']
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"ACTUAL {l}" for l in labels], columns=[f"PRED {l}" for l in labels])

            st.table(cm_df)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering Model Intelligence: {str(e)}")