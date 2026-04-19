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
    
    .stApp {
        background: linear-gradient(135deg, #F9F3E3 0%, #FFFFFF 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(27, 94, 32, 0.1);
        margin-bottom: 25px;
    }
    
    /* Metric styling */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        border-bottom: 4px solid #1B5E20;
    }
    
    .yield-high { color: #1B5E20; font-weight: 700; font-size: 3rem; margin: 0; }
    .yield-medium { color: #FBC02D; font-weight: 700; font-size: 3rem; margin: 0; }
    .yield-low { color: #C62828; font-weight: 700; font-size: 3rem; margin: 0; }
    
    .badge {
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-high { background-color: #E8F5E9; color: #1B5E20; border: 1px solid #1B5E20; }
    .badge-medium { background-color: #FFFDE7; color: #FBC02D; border: 1px solid #FBC02D; }
    .badge-low { background-color: #FFEBEE; color: #C62828; border: 1px solid #C62828; }
    
    .risk-chip {
        display: inline-block;
        padding: 6px 14px;
        margin: 5px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .risk-high { background-color: #FFEBEE; color: #C62828; }
    .risk-medium { background-color: #FFF9C4; color: #7F6500; }
    
    .stButton>button {
        background: linear-gradient(90deg, #1B5E20 0%, #2E7D32 100%);
        color: white;
        border: None;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.3);
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

# --- PDF GENERATION (FIXED BYTE FORMAT) ---
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
    # Important: Convert bytearray to bytes for Streamlit
    return bytes(pdf.output())

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color: #1B5E20;'>🌾 FarmIQ</h1>", unsafe_allow_html=True)
    selected = option_menu(None, ["Home", "Predict & Advise", "About"], icons=["house", "cpu", "info"], 
                          styles={"nav-link-selected": {"background-color": "#1B5E20"}}, 
                          default_index=["Home", "Predict & Advise", "About"].index(st.session_state.selected_page))
    st.session_state.selected_page = selected

# --- PAGES ---
if selected == "Home":
    st.markdown("<h1 style='color: #1B5E20;'>Cultivating Precision</h1>", unsafe_allow_html=True)
    st.write("AI-powered agricultural yield forecasting and risk analysis.")
    if st.button("Get Started →"):
        st.session_state.selected_page = "Predict & Advise"
        st.rerun()

elif selected == "Predict & Advise":
    st.markdown("<h2 style='color: #1B5E20;'>Agricultural Intelligence Hub</h2>", unsafe_allow_html=True)
    
    # Input Portal
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
                # Store inputs in session state to persist
                st.session_state.raw_inputs = {"Item": crop, "average_rain_fall_mm_per_year": rainfall, "avg_temp": temp, "Soil": soil, "Area": area, "pesticides_tonnes": 1.5}
                st.rerun()

    # Results Display (Persistent via Session State)
    if st.session_state.prediction_made and st.session_state.raw_inputs:
        inputs = st.session_state.raw_inputs
        model = tree_model if tree_model else linear_model
        if model:
            # Correctly identify categorical columns for preprocessing
            df_in = pd.DataFrame([{ 
                "Area": "India", 
                "Item": inputs['Item'], 
                "Year": 2024, 
                "average_rain_fall_mm_per_year": inputs['average_rain_fall_mm_per_year'], 
                "pesticides_tonnes": inputs['pesticides_tonnes'], 
                "avg_temp": inputs['avg_temp'] 
            }])
            
            # Use detected categorical/numeric columns to ensure nothing is missed
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
                    for r, l in risks: st.markdown(f"<span class='risk-chip {'risk-high' if l=='high' else 'risk-medium'}'>{r}</span>", unsafe_allow_html=True)
            with col2:
                with st.container(border=True):
                    st.markdown("##### ✅ Advice")
                    for a in advice: st.write(f"- {a}")

            # PDF Download (Fixed)
            pdf_bytes = generate_pdf(inputs, f"{val:.2f} t/ha", y_cl, risks, advice)
            st.download_button("📥 DOWNLOAD REPORT (PDF)", bytes(pdf_bytes), f"FarmIQ_{inputs['Item']}.pdf", "application/pdf")
            
            if st.button("🔄 NEW ANALYSIS"):
                st.session_state.prediction_made = False
                st.session_state.raw_inputs = None
                st.rerun()
else:
    st.write("FarmIQ Architecture: Random Forest + Flan-T5 Reasoning.")
