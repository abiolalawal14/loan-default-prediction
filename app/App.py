import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }

    .stSlider label,
    .stNumberInput label,
    .stSelectbox label,
    .stCheckbox label,
    .stTextInput label {
        color: #1F3864 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stCheckbox span {
        color: #1F3864 !important;
        font-size: 14px !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1F3864 !important;
        border: 1px solid #B5D4F4 !important;
    }
    .stNumberInput div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #B5D4F4 !important;
    }
    .stNumberInput input {
        color: #1F3864 !important;
        background-color: #FFFFFF !important;
    }
    input[type="number"] {
        color: #1F3864 !important;
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] span {
        color: #1F3864 !important;
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] div {
        background-color: #FFFFFF !important;
        color: #1F3864 !important;
    }
    div[data-baseweb="input"] input {
        color: #1F3864 !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #1F3864 !important;
    }
    .stCheckbox > label > div:first-child {
        background-color: #FFFFFF !important;
        border: 2px solid #2E75B6 !important;
        border-radius: 4px !important;
    }
    .stCheckbox > label > div:last-child {
        color: #1F3864 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stNumberInput div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
    }
    .stNumberInput button {
        background-color: #EBF4FB !important;
        color: #1F3864 !important;
        border: 1px solid #B5D4F4 !important;
    }
    div[data-baseweb] {
        background-color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #1F3864 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1F3864 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] a {
        color: #B5D4F4 !important;
    }
    .stButton > button {
        background-color: #1F3864 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #2E75B6 !important;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2E75B6;
    }
    .risk-high {
        background: #FFF0F0;
        border-left: 5px solid #E24B4A;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-medium {
        background: #FFFBF0;
        border-left: 5px solid #F0A500;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: #F0FFF4;
        border-left: 5px solid #1D9E75;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1F3864;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid #2E75B6;
    }
    .insight-box {
        background: #EBF4FB;
        border-radius: 8px;
        padding: 14px;
        font-size: 14px;
        color: #1F3864;
        margin-top: 10px;
        line-height: 1.7;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL — RELATIVE PATHS
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    # Get the directory where app.py lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '..', 'models')

    model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))
    feature_names = joblib.load(
        os.path.join(models_dir, 'feature_names.pkl')
    )
    return model, feature_names

model, feature_names = load_model()


# ─────────────────────────────────────────
# NIGERIAN FEATURE LABEL MAP
# ─────────────────────────────────────────
FEATURE_LABELS = {
    'EXT_SOURCE_MEAN':              'Credit Bureau Score (Average)',
    'EXT_SOURCE_2':                 'Credit Bureau Score 2',
    'EXT_SOURCE_3':                 'Credit Bureau Score 3',
    'EXT_SOURCE_1':                 'Credit Bureau Score 1',
    'EXT_SOURCE_MIN':               'Credit Bureau Score (Weakest)',
    'CREDIT_INCOME_RATIO':          'Loan-to-Income Ratio',
    'ANNUITY_INCOME_RATIO':         'Monthly Repayment Burden',
    'CREDIT_TERM_YEARS':            'Loan Repayment Duration',
    'AMT_CREDIT':                   'Loan Amount (₦)',
    'AMT_INCOME_TOTAL':             'Annual Income (₦)',
    'AMT_ANNUITY':                  'Monthly Annuity (₦)',
    'AMT_GOODS_PRICE':              'Asset / Goods Value (₦)',
    'AGE_YEARS':                    'Borrower Age (Years)',
    'DAYS_BIRTH':                   'Borrower Age',
    'EMPLOYMENT_YEARS':             'Years in Employment',
    'DAYS_EMPLOYED':                'Employment Duration',
    'EMPLOYMENT_AGE_RATIO':         'Employment Stability Index',
    'INCOME_PER_PERSON':            'Income Per Family Member (₦)',
    'CNT_CHILDREN':                 'Number of Dependants',
    'CNT_FAM_MEMBERS':              'Total Family Size',
    'CODE_GENDER_M':                'Gender (Male)',
    'FLAG_OWN_CAR':                 'Owns a Vehicle',
    'FLAG_OWN_REALTY':              'Owns Property / Real Estate',
    'FLAG_DOCUMENT_3':              'ID Document 3 Submitted',
    'NAME_CONTRACT_TYPE':           'Loan Type (Revolving)',
    'NAME_FAMILY_STATUS_Married':   'Marital Status (Married)',
    'NAME_EDUCATION_TYPE_Higher education': 'Education (Tertiary)',
    'DAYS_ID_PUBLISH':              'ID Document Age (Days)',
    'REGION_RATING_CLIENT':         'Region Risk Rating',
    'REGION_RATING_CLIENT_W_CITY':  'City Region Risk Rating',
}


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def get_risk_label(probability):
    if probability >= 0.6:
        return "HIGH RISK", "#E24B4A", "risk-high", "🔴"
    elif probability >= 0.3:
        return "MEDIUM RISK", "#F0A500", "risk-medium", "🟡"
    else:
        return "LOW RISK", "#1D9E75", "risk-low", "🟢"


def build_input_dataframe(inputs, feature_names):
    row = pd.DataFrame([{f: 0 for f in feature_names}])
    for key, value in inputs.items():
        if key in row.columns:
            row[key] = value
    return row


def get_label(feature):
    return FEATURE_LABELS.get(
        feature,
        feature.replace('_', ' ').title()
    )


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Default Risk Predictor")
    st.markdown("---")
    st.markdown("""
    This tool uses a machine learning model trained on
    **307,511 real loan applications** to predict the
    probability that a borrower will default.
    """)
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Enter borrower details on the right
    2. Click **Predict Default Risk**
    3. View the risk score and explanation
    """)
    st.markdown("---")
    st.markdown("### Model Performance")
    st.markdown("""
    - **AUC-ROC:** 0.7679
    - **Recall:** 65.8% of defaults caught
    - **Algorithm:** XGBoost
    - **Training records:** 307,511
    """)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px'>
    Built by <strong>Abiola Lawal</strong><br>
    Data Scientist | Credit Risk & Fintech ML<br>
    <a href='https://linkedin.com/in/abiola-lawal-abdulrafiu'
    target='_blank'>LinkedIn</a> ·
    <a href='https://github.com/abiolalawal14'
    target='_blank'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────
st.markdown("# 🏦 Loan Default Risk Predictor")
st.markdown(
    "Enter borrower information below to get an instant "
    "credit risk assessment powered by machine learning."
)
st.markdown("---")


# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
st.markdown(
    '<div class="section-header">Borrower Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<p style="font-size:15px;font-weight:700;color:#1F3864;'
        'border-bottom:2px solid #2E75B6;padding-bottom:6px">'
        'Personal Details</p>',
        unsafe_allow_html=True
    )
    age = st.slider(
        "Age (years)", 18, 70, 35,
        help="Borrower's current age"
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_status = st.selectbox(
        "Family Status",
        ["Single", "Married", "Divorced", "Widow"]
    )
    children = st.number_input(
        "Number of Children", 0, 10, 0
    )
    family_members = st.number_input(
        "Total Family Members", 1, 15, 2
    )
    education = st.selectbox(
        "Education Level",
        [
            "Secondary",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree"
        ]
    )

with col2:
    st.markdown(
        '<p style="font-size:15px;font-weight:700;color:#1F3864;'
        'border-bottom:2px solid #2E75B6;padding-bottom:6px">'
        'Financial Details</p>',
        unsafe_allow_html=True
    )
    income = st.number_input(
        "Annual Income (₦)",
        min_value=10000,
        max_value=10000000,
        value=150000,
        step=10000,
        format="%d",
        help="Total annual income in Naira"
    )
    loan_amount = st.number_input(
        "Loan Amount Requested (₦)",
        min_value=10000,
        max_value=10000000,
        value=500000,
        step=10000,
        format="%d",
        help="Total loan amount being requested"
    )
    annuity = st.number_input(
        "Monthly Repayment Amount (₦)",
        min_value=1000,
        max_value=500000,
        value=25000,
        step=1000,
        format="%d",
        help="Expected monthly repayment instalment"
    )
    goods_price = st.number_input(
        "Asset / Goods Value (₦)",
        min_value=0,
        max_value=10000000,
        value=450000,
        step=10000,
        format="%d",
        help="Value of goods or property being financed"
    )
    contract_type = st.selectbox(
        "Loan Type",
        ["Cash loans", "Revolving loans"],
        help="Type of credit facility"
    )

with col3:
    st.markdown(
        '<p style="font-size:15px;font-weight:700;color:#1F3864;'
        'border-bottom:2px solid #2E75B6;padding-bottom:6px">'
        'Credit History & Employment</p>',
        unsafe_allow_html=True
    )
    ext_source_1 = st.slider(
        "Credit Bureau Score 1",
        0.0, 1.0, 0.5, 0.01,
        help="External credit score from Bureau 1 "
             "(0 = poor, 1 = excellent). "
             "Leave at 0.5 if unknown."
    )
    ext_source_2 = st.slider(
        "Credit Bureau Score 2",
        0.0, 1.0, 0.5, 0.01,
        help="External credit score from Bureau 2 "
             "(0 = poor, 1 = excellent). "
             "Leave at 0.5 if unknown."
    )
    ext_source_3 = st.slider(
        "Credit Bureau Score 3",
        0.0, 1.0, 0.5, 0.01,
        help="External credit score from Bureau 3 "
             "(0 = poor, 1 = excellent). "
             "Leave at 0.5 if unknown."
    )
    employment_years = st.slider(
        "Years in Current Employment",
        0, 40, 5,
        help="How long the borrower has been in current job"
    )
    st.markdown(
        '<p style="font-size:13px;font-weight:600;'
        'color:#1F3864;margin-top:8px;margin-bottom:4px">'
        'Asset Ownership</p>',
        unsafe_allow_html=True
    )
    own_car = st.checkbox("✅ Owns a Vehicle")
    own_realty = st.checkbox("✅ Owns Property / Real Estate")
    flag_document_3 = st.checkbox(
        "✅ Key ID Document Submitted",
        help="Borrower has provided the required "
             "identification document"
    )

st.markdown("---")


# ─────────────────────────────────────────
# ENGINEERED FEATURES
# ─────────────────────────────────────────
def compute_engineered_features(
        age, income, loan_amount, annuity,
        employment_years, ext_source_1,
        ext_source_2, ext_source_3,
        family_members, goods_price,
        children, gender, own_car,
        own_realty, flag_document_3,
        contract_type, education,
        family_status):

    ext_sources = [
        s for s in [ext_source_1, ext_source_2, ext_source_3]
        if s > 0
    ]
    ext_mean = np.mean(ext_sources) if ext_sources else 0.5
    ext_min = np.min(ext_sources) if ext_sources else 0.5

    return {
        'AGE_YEARS': age,
        'EMPLOYMENT_YEARS': employment_years,
        'CREDIT_INCOME_RATIO': loan_amount / max(income, 1),
        'ANNUITY_INCOME_RATIO': annuity / max(income, 1),
        'CREDIT_TERM_YEARS': annuity / max(loan_amount, 1),
        'INCOME_PER_PERSON': income / max(family_members, 1),
        'EXT_SOURCE_MEAN': ext_mean,
        'EXT_SOURCE_MIN': ext_min,
        'EMPLOYMENT_AGE_RATIO': employment_years / max(age, 1),
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'AMT_CREDIT': loan_amount,
        'AMT_INCOME_TOTAL': income,
        'AMT_ANNUITY': annuity,
        'AMT_GOODS_PRICE': goods_price,
        'DAYS_BIRTH': -age * 365,
        'DAYS_EMPLOYED': -employment_years * 365,
        'CNT_CHILDREN': children,
        'CNT_FAM_MEMBERS': family_members,
        'CODE_GENDER_M': 1 if gender == "Male" else 0,
        'FLAG_OWN_CAR': 1 if own_car else 0,
        'FLAG_OWN_REALTY': 1 if own_realty else 0,
        'FLAG_DOCUMENT_3': 1 if flag_document_3 else 0,
        'NAME_CONTRACT_TYPE':
            1 if contract_type == "Revolving loans" else 0,
        'NAME_EDUCATION_TYPE_Higher education':
            1 if education == "Higher education" else 0,
        'NAME_FAMILY_STATUS_Married':
            1 if family_status == "Married" else 0,
    }


# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_button = st.button("🔍 Predict Default Risk")

if predict_button:

    engineered = compute_engineered_features(
        age, income, loan_amount, annuity,
        employment_years, ext_source_1,
        ext_source_2, ext_source_3,
        family_members, goods_price,
        children, gender, own_car,
        own_realty, flag_document_3,
        contract_type, education,
        family_status
    )
    input_df = build_input_dataframe(engineered, feature_names)

    probability = model.predict_proba(input_df)[0][1]
    risk_label, risk_color, risk_class, risk_icon = \
        get_risk_label(probability)

    credit_income = loan_amount / max(income, 1)
    ext_avg = np.mean([ext_source_1, ext_source_2, ext_source_3])

    st.markdown("---")
    st.markdown("## Assessment Result")

    # ── RESULT CARDS ────────────────────────
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown(f"""
        <div class="{risk_class}">
            <div style="font-size:36px">{risk_icon}</div>
            <div style="font-size:20px;font-weight:700;
                        color:{risk_color};margin-top:6px">
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:12px;color:#888;margin-bottom:4px">
                Default Probability
            </div>
            <div style="font-size:36px;font-weight:700;
                        color:{risk_color}">
                {probability*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        ci_color = (
            "#E24B4A" if credit_income > 5 else
            "#F0A500" if credit_income > 3 else
            "#1D9E75"
        )
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:12px;color:#888;margin-bottom:4px">
                Loan-to-Income Ratio
            </div>
            <div style="font-size:36px;font-weight:700;
                        color:{ci_color}">
                {credit_income:.1f}x
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        bureau_color = (
            "#E24B4A" if ext_avg < 0.35 else
            "#F0A500" if ext_avg < 0.55 else
            "#1D9E75"
        )
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:12px;color:#888;margin-bottom:4px">
                Avg Credit Bureau Score
            </div>
            <div style="font-size:36px;font-weight:700;
                        color:{bureau_color}">
                {ext_avg:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── INSIGHT BOX ─────────────────────────
    if probability >= 0.6:
        insight = (
            "⚠️ This application carries significant default risk. "
            "The combination of credit bureau scores and debt burden "
            "suggests this borrower may struggle with repayment. "
            "Manual review strongly recommended before approval."
        )
    elif probability >= 0.3:
        insight = (
            "⚡ This application shows moderate risk indicators. "
            "Consider requesting additional documentation, a guarantor, "
            "or reducing the loan amount to bring the "
            "loan-to-income ratio within acceptable bounds."
        )
    else:
        insight = (
            "✅ This application presents a low default risk profile. "
            "Strong credit bureau scores and a manageable "
            "loan-to-income ratio suggest this borrower is likely "
            "to meet repayment obligations. "
            "Standard approval process recommended."
        )

    st.markdown(
        f'<div class="insight-box">{insight}</div>',
        unsafe_allow_html=True
    )

    # ── SHAP CHART ──────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">'
        'Why did the model give this score?'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "The chart below shows which factors pushed the risk "
        "score **up** 🔴 or **down** 🔵 for this specific borrower."
    )

    with st.spinner("Generating credit risk explanation..."):
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)

            shap_series = pd.Series(
                shap_vals[0], index=feature_names
            ).abs().sort_values(ascending=False).head(12)

            top_features = shap_series.index.tolist()
            top_shap = pd.Series(
                shap_vals[0], index=feature_names
            )[top_features]

            # Apply Nigerian-friendly labels
            top_labels = [
                get_label(f) for f in top_features
            ]

            colors = [
                '#E24B4A' if v > 0 else '#2E75B6'
                for v in top_shap.values
            ]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                top_labels[::-1],
                top_shap.values[::-1],
                color=colors[::-1],
                edgecolor='none',
                height=0.6
            )
            ax.axvline(x=0, color='#333333', linewidth=0.8)
            ax.set_xlabel(
                'Impact on Default Probability',
                fontsize=11
            )
            ax.set_title(
                'Credit Risk Drivers — '
                'Why This Borrower Was Scored This Way\n'
                '🔴 Red = increases default risk  |  '
                '🔵 Blue = reduces default risk',
                fontsize=11, fontweight='bold', pad=14
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', labelsize=10)
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FAFAFA')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning(
                f"SHAP explanation could not be generated: {e}"
            )

    # ── KEY RISK FACTORS ────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Key Risk Factors</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        bureau_status = (
            "🔴 Weak — High Risk" if ext_avg < 0.35 else
            "🟡 Moderate — Review" if ext_avg < 0.55 else
            "🟢 Strong — Low Risk"
        )
        st.markdown(f"""
        <div style="background:white;border-radius:10px;
                    padding:16px;border-left:4px solid #2E75B6;
                    box-shadow:0 2px 6px rgba(0,0,0,0.06)">
            <div style="font-size:12px;color:#888;margin-bottom:4px">
                Credit Bureau History
            </div>
            <div style="font-size:24px;font-weight:700;color:#1F3864">
                {ext_avg:.2f} / 1.00
            </div>
            <div style="font-size:13px;color:#444;margin-top:6px">
                {bureau_status}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        debt_color = (
            "#E24B4A" if credit_income > 5 else
            "#F0A500" if credit_income > 3 else
            "#1D9E75"
        )
        debt_status = (
            "🔴 Very High — Decline Risk" if credit_income > 5 else
            "🟡 Elevated — Review" if credit_income > 3 else
            "🟢 Manageable — Acceptable"
        )
        st.markdown(f"""
        <div style="background:white;border-radius:10px;
                    padding:16px;border-left:4px solid {debt_color};
                    box-shadow:0 2px 6px rgba(0,0,0,0.06)">
            <div style="font-size:12px;color:#888;margin-bottom:4px">
                Loan-to-Income Ratio
            </div>
            <div style="font-size:24px;font-weight:700;
                        color:{debt_color}">
                {credit_income:.1f}x annual income
            </div>
            <div style="font-size:13px;color:#444;margin-top:6px">
                {debt_status}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        emp_color = (
            "#E24B4A" if employment_years < 1 else
            "#F0A500" if employment_years < 3 else
            "#1D9E75"
        )
        employ_status = (
            "🔴 Unstable — No Employment" if employment_years < 1 else
            "🟡 Early Stage — Less than 3yrs" if employment_years < 3
            else "🟢 Stable — 3+ Years"
        )
        st.markdown(f"""
        <div style="background:white;border-radius:10px;
                    padding:16px;border-left:4px solid {emp_color};
                    box-shadow:0 2px 6px rgba(0,0,0,0.06)">
            <div style="font-size:12px;color:#888;margin-bottom:4px">
                Employment Stability
            </div>
            <div style="font-size:24px;font-weight:700;
                        color:{emp_color}">
                {employment_years} years
            </div>
            <div style="font-size:13px;color:#444;margin-top:6px">
                {employ_status}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RECOMMENDATION ──────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        'Credit Officer Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        if probability >= 0.6:
            st.markdown(f"""
            <div style="background:#FFF0F0;border-radius:10px;
                        padding:18px;border-left:5px solid #E24B4A">
                <div style="font-size:15px;font-weight:700;
                            color:#E24B4A;margin-bottom:8px">
                    ❌ Recommendation: DECLINE or REFER
                </div>
                <div style="font-size:14px;color:#333;line-height:1.7">
                    Default probability exceeds acceptable threshold.
                    If referral is considered, require collateral,
                    guarantor, or significant loan reduction before
                    re-assessment.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif probability >= 0.3:
            st.markdown(f"""
            <div style="background:#FFFBF0;border-radius:10px;
                        padding:18px;border-left:5px solid #F0A500">
                <div style="font-size:15px;font-weight:700;
                            color:#B8860B;margin-bottom:8px">
                    ⚡ Recommendation: CONDITIONAL APPROVAL
                </div>
                <div style="font-size:14px;color:#333;line-height:1.7">
                    Proceed with caution. Consider reducing loan amount,
                    shortening repayment term, or requesting additional
                    supporting documents before final approval.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#F0FFF4;border-radius:10px;
                        padding:18px;border-left:5px solid #1D9E75">
                <div style="font-size:15px;font-weight:700;
                            color:#1D9E75;margin-bottom:8px">
                    ✅ Recommendation: APPROVE
                </div>
                <div style="font-size:14px;color:#333;line-height:1.7">
                    Risk profile is within acceptable bounds.
                    Proceed with standard loan processing and
                    documentation verification.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:18px;
                    border:0.5px solid #D6E4F0;
                    box-shadow:0 2px 6px rgba(0,0,0,0.05)">
            <div style="font-size:15px;font-weight:700;color:#1F3864;
                        margin-bottom:12px;
                        border-bottom:2px solid #2E75B6;
                        padding-bottom:6px">
                Summary for Credit File
            </div>
            <table style="width:100%;font-size:13px;
                          border-collapse:collapse">
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Borrower Age
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:#1F3864;text-align:right">
                        {age} years
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Annual Income
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:#1F3864;text-align:right">
                        ₦{income:,}
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Loan Requested
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:#1F3864;text-align:right">
                        ₦{loan_amount:,}
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Monthly Repayment
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:#1F3864;text-align:right">
                        ₦{annuity:,}
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Loan-to-Income Ratio
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:{ci_color};text-align:right">
                        {credit_income:.1f}x
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Avg Bureau Score
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:{bureau_color};text-align:right">
                        {ext_avg:.2f} / 1.00
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Employment
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:#1F3864;text-align:right">
                        {employment_years} years
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #f0f0f0">
                    <td style="padding:6px 0;color:#888">
                        Model Score
                    </td>
                    <td style="padding:6px 0;font-weight:600;
                               color:{risk_color};text-align:right">
                        {probability*100:.1f}% default probability
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#888">
                        Risk Category
                    </td>
                    <td style="padding:6px 0;font-weight:700;
                               color:{risk_color};text-align:right">
                        {risk_label}
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <strong>Abiola Lawal</strong> —
    Data Scientist | Credit Risk & Fintech ML |
    <a href='https://linkedin.com/in/abiola-lawal-abdulrafiu'
    target='_blank'>LinkedIn</a> ·
    <a href='https://github.com/abiolalawal14'
    target='_blank'>GitHub</a><br>
    Trained on 307,511 loan applications ·
    XGBoost · SHAP Explainability ·
    Built for Nigerian Credit Risk Operations
</div>
""", unsafe_allow_html=True)