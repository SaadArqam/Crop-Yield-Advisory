import streamlit as st
import pandas as pd
import joblib
import os
import json
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from fpdf import FPDF
from io import BytesIO
from datetime import datetime

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="FarmIQ - AI Crop Advisory", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE SESSION STATE ---
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Home"
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'raw_inputs' not in st.session_state:
    st.session_state.raw_inputs = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# --- CUSTOM CSS ---
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Theme-aware background and text */
    [data-testid="stAppViewContainer"] {
        background: var(--background-color);
        background-image: linear-gradient(135deg, rgba(27, 94, 32, 0.05) 0%, rgba(255, 255, 255, 0.05) 100%);
    }
    
    /* Navigation Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Uniform Card Styling */
    .stCard {
        background-color: var(--secondary-background-color);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(27, 94, 32, 0.2);
        margin-bottom: 25px;
        color: var(--text-color);
    }
    
    /* Metric Styling */
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-bottom: 4px solid #1B5E20;
        color: var(--text-color);
    }
    
    .yield-high { color: #2E7D32; font-weight: 700; font-size: 3.5rem; margin: 0; }
    .yield-medium { color: #FBC02D; font-weight: 700; font-size: 3.5rem; margin: 0; }
    .yield-low { color: #C62828; font-weight: 700; font-size: 3.5rem; margin: 0; }
    
    .badge {
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        margin-top: 10px;
    }
    .badge-high { background-color: rgba(46, 125, 50, 0.2); color: #2E7D32; border: 1px solid #2E7D32; }
    .badge-medium { background-color: rgba(251, 192, 45, 0.2); color: #FBC02D; border: 1px solid #FBC02D; }
    .badge-low { background-color: rgba(198, 40, 40, 0.2); color: #C62828; border: 1px solid #C62828; }
    
    .risk-chip {
        display: inline-block;
        padding: 10px 18px;
        margin: 8px 4px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 500;
        border-left: 4px solid #1B5E20;
        background-color: rgba(128, 128, 128, 0.1);
    }
    .risk-high { background-color: rgba(198, 40, 40, 0.1); color: #C62828; border-left-color: #C62828; }
    .risk-medium { background-color: rgba(251, 192, 45, 0.1); color: #FBC02D; border-left-color: #FBC02D; }
    
    /* Buttons */
    .stButton > button {
        background: #1B5E20;
        color: white !important;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #2E7D32;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.4);
        transform: translateY(-2px);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #1B5E20;
        font-weight: 700;
    }
    
    /* Custom Sidebar Title */
    .sidebar-title {
        color: #1B5E20;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .sidebar-sub {
        color: var(--text-color);
        font-size: 0.9rem;
        text-align: center;
        opacity: 0.8;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- PRESERVE CORE LOGIC ---
@st.cache_resource
def load_llm():
    try:
        from transformers import pipeline
        return pipeline("text2text-generation", model="google/flan-t5-base")
    except Exception:
        return None

llm_pipeline = load_llm()
LLM_AVAILABLE = llm_pipeline is not None

def load_artifacts():
    linear_model, tree_model, label_encoders, feature_order, medians = None, None, {}, None, {}
    if os.path.exists('models/crop_yield_model.pkl'): linear_model = joblib.load('models/crop_yield_model.pkl')
    if os.path.exists('models/crop_yield_tree.pkl'): tree_model = joblib.load('models/crop_yield_tree.pkl')
    if os.path.exists('models/label_encoders_pipeline.pkl'): label_encoders = joblib.load('models/label_encoders_pipeline.pkl')
    elif os.path.exists('models/label_encoders.pkl'): label_encoders = joblib.load('models/label_encoders.pkl')
    if os.path.exists('models/feature_order.json'):
        with open('models/feature_order.json') as f: feature_order = json.load(f)
    if os.path.exists('models/medians.json'):
        with open('models/medians.json') as f: medians = json.load(f)
    return linear_model, tree_model, label_encoders, feature_order, medians

linear_model, tree_model, label_encoders, feature_order, medians = load_artifacts()

def apply_simple_preprocessing(df_input, feature_order, numeric_cols, categorical_cols, medians, label_encoders):
    df = df_input.copy()
    for col in (numeric_cols or []):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(medians.get(col, 0))
    for col in (categorical_cols or []):
        df[col] = df[col].fillna('missing').astype(str)
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: int(le.transform([x])[0]) if str(x) in le.classes_ else (int(le.transform(['missing'])[0]) if 'missing' in le.classes_ else 0))
        else:
            df[col] = df[col].astype('category').cat.codes
    if feature_order is not None:
        df = df[[c for c in feature_order if c in df.columns]]
    return df

def align_to_model(df, model, feature_order, medians_dict):
    expected = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else feature_order
    if expected is None: return df
    df2 = df.copy()
    for col in expected:
        if col not in df2.columns: df2[col] = medians_dict.get(col, 0)
    return df2.reindex(columns=expected)

def analyze_risks(inputs, yield_t):
    risks = []
    rain = float(inputs.get('average_rain_fall_mm_per_year', 1000))
    temp = float(inputs.get('avg_temp', 25))
    if rain < 500: risks.append(("Arid Conditions: High drought risk", "high"))
    elif rain < 800: risks.append(("Low Rainfall: Moisture stress", "medium"))
    if temp > 35: risks.append(("Extreme Heat: Thermal stress", "high"))
    if yield_t < 2.0: risks.append(("Performance Risk: Low yield potential", "high"))
    return risks

def generate_advice(risks, inputs, yield_t):
    if LLM_AVAILABLE and risks:
        try:
            prompt = f"Yield: {yield_t:.2f} t/ha. Risks: {', '.join([r[0] for r in risks])}. Give 3 farming tips."
            return [llm_pipeline(prompt, max_length=150)[0]['generated_text']]
        except Exception: pass
    advice = []
    for r, l in risks:
        if "drought" in r.lower(): advice.append("Adopt high-efficiency irrigation.")
        elif "heat" in r.lower(): advice.append("Use organic mulching.")
    return list(set(advice)) if advice else ["Maintain standard maintenance."]

# --- PDF GENERATION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(27, 94, 32)
        self.cell(0, 10, 'FARMIQ ADVISORY REPORT', 0, 1, 'C')
        self.ln(10)
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(232, 245, 233)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(2)
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        safe_body = body.encode('ascii', 'ignore').decode('ascii')
        self.multi_cell(0, 8, safe_body)
        self.ln(5)

def generate_pdf(inputs, pred_text, yield_label, risks, actions):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title("I. Farm Summary")
    pdf.chapter_body("\n".join([f"{k}: {v}" for k, v in inputs.items()]))
    pdf.chapter_title("II. Predicted Yield")
    pdf.chapter_body(f"Forecast: {pred_text} | Level: {yield_label}")
    pdf.chapter_title("III. Risks & Actions")
    pdf.chapter_body("\n".join([f"- {r[0]}" for r in risks]) if risks else "No risks.")
    pdf.chapter_body("\n".join([f"Action: {a}" for a in actions]))
    return bytes(pdf.output())

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🌾 FarmIQ</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Agricultural Intelligence</div>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predict & Advise", "About"],
        icons=["house", "cpu", "info-circle"],
        default_index=["Home", "Predict & Advise", "About"].index(st.session_state.selected_page),
        styles={
            "container": {"background-color": "transparent", "padding": "5px!important"},
            "icon": {"color": "#1B5E20", "font-size": "1.2rem"},
            "nav-link": {"font-size": "1rem", "text-align": "left", "margin": "10px", "color": "var(--text-color)"},
            "nav-link-selected": {"background-color": "#1B5E20", "color": "white !important"},
        }
    )
    st.session_state.selected_page = selected

# --- PAGES ---
if selected == "Home":
    st.markdown("<h1>Cultivating Precision</h1>", unsafe_allow_html=True)
    st.write("AI-powered agricultural yield forecasting and risk analysis.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='stCard'><h3>📈 Analytics</h3><p>Highly accurate yield forecasting models.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stCard'><h3>🤖 Advisory</h3><p>Real-time AI-based farming recommendations.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='stCard'><h3>📋 Export</h3><p>Professional agronomy reports in PDF format.</p></div>", unsafe_allow_html=True)
        
    if st.button("Get Started →"):
        st.session_state.selected_page = "Predict & Advise"
        st.rerun()

elif selected == "Predict & Advise":
    st.markdown("<h2>Agricultural Intelligence Hub</h2>", unsafe_allow_html=True)
    
    with st.expander("📝 DATA ENTRY PORTAL", expanded=not st.session_state.prediction_made):
        if st.button("🧩 Demo Data"):
            st.session_state.sample_data = {"Item": "Maize", "Rain": 1200, "Temp": 28, "Soil": "Loamy", "Area": 2.5}
        sd = st.session_state.get('sample_data', {})
        with st.form("input_form"):
            c1, c2 = st.columns(2)
            with c1:
                crop = st.selectbox("Crop", ["Maize", "Rice, paddy", "Wheat", "Potatoes"], index=0)
                rainfall = st.slider("Rainfall (mm)", 0, 3000, sd.get("Rain", 1000))
                temp = st.slider("Temp (°C)", 5, 50, sd.get("Temp", 25))
            with c2:
                soil = st.selectbox("Soil", ["Loamy", "Clay", "Sandy"], index=0)
                area = st.number_input("Area (ha)", 0.1, 100.0, float(sd.get("Area", 1.0)))
            if st.form_submit_button("⚡ ANALYZE"):
                st.session_state.prediction_made = True
                st.session_state.raw_inputs = {"Item": crop, "average_rain_fall_mm_per_year": rainfall, "avg_temp": temp, "Soil": soil, "Area": area, "pesticides_tonnes": 1.5}
                st.rerun()

    if st.session_state.prediction_made and st.session_state.raw_inputs:
        inputs = st.session_state.raw_inputs
        model = tree_model if tree_model else linear_model
        if model:
            # Preprocessing and prediction logic
            df_in = pd.DataFrame([{ "Area": "India", "Item": inputs['Item'], "Year": 2024, "average_rain_fall_mm_per_year": inputs['average_rain_fall_mm_per_year'], "pesticides_tonnes": inputs['pesticides_tonnes'], "avg_temp": inputs['avg_temp'] }])
            cat_cols = [c for c in df_in.columns if df_in[c].dtype == 'object']
            num_cols = [c for c in df_in.columns if c not in cat_cols]
            Xp_processed = apply_simple_preprocessing(df_in, feature_order, num_cols, cat_cols, medians, label_encoders)
            Xp_model = align_to_model(Xp_processed, model, feature_order, medians)
            val = model.predict(Xp_model)[0] / 10000.0
            
            y_cl, y_co, y_ba = ("High", "yield-high", "badge-high") if val >= 4 else (("Medium", "yield-medium", "badge-medium") if val >= 2 else ("Low", "yield-low", "badge-low"))
            st.markdown(f"<div class='metric-card'><h1 class='{y_co}'>{val:.2f} t/ha</h1><span class='badge {y_ba}'>{y_cl} Yield</span></div>", unsafe_allow_html=True)
            
            risks = analyze_risks(inputs, val)
            advice = generate_advice(risks, inputs, val)
            
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("##### ⚠️ Risks")
                    if risks:
                        for r, l in risks: st.markdown(f"<span class='risk-chip {'risk-high' if l=='high' else 'risk-medium'}'>{r}</span>", unsafe_allow_html=True)
                    else: st.write("No major risks identified.")
            with col2:
                with st.container(border=True):
                    st.markdown("##### ✅ Advice")
                    for a in advice: st.write(f"- {a}")

            pdf_bytes = generate_pdf(inputs, f"{val:.2f} t/ha", y_cl, risks, advice)
            st.download_button("📥 DOWNLOAD REPORT (PDF)", bytes(pdf_bytes), f"FarmIQ_{inputs['Item']}.pdf", "application/pdf")
            
            if st.button("🔄 NEW ANALYSIS"):
                st.session_state.prediction_made = False
                st.session_state.raw_inputs = None
                st.rerun()

elif selected == "About":
    st.markdown("<h2>System Architecture</h2>", unsafe_allow_html=True)
    st.write("FarmIQ uses Random Forest ensembles for yield prediction and Flan-T5 for agentic farm advisory.")
