import streamlit as st
import pandas as pd
import joblib
import os
import json

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("🌾 Crop Yield Prediction — Agentic Pipeline")
st.write("Upload a dataset or enter single-record values to get yield predictions and an agentic advisory report.")

# load models
linear_model = None
tree_model = None
label_encoders = {}

if os.path.exists('models/crop_yield_model.pkl'):
    linear_model = joblib.load('models/crop_yield_model.pkl')
if os.path.exists('models/crop_yield_tree.pkl'):
    tree_model = joblib.load('models/crop_yield_tree.pkl')

if os.path.exists('models/label_encoders_pipeline.pkl'):
    label_encoders = joblib.load('models/label_encoders_pipeline.pkl')
elif os.path.exists('models/label_encoders.pkl'):
    label_encoders = joblib.load('models/label_encoders.pkl')

feature_order = None
medians = {}
if os.path.exists('models/feature_order.json'):
    try:
        with open('models/feature_order.json') as f:
            feature_order = json.load(f)
    except Exception:
        feature_order = None
if os.path.exists('models/medians.json'):
    try:
        with open('models/medians.json') as f:
            medians = json.load(f)
    except Exception:
        medians = {}

train_df = None
numeric_cols = []
categorical_cols = []
if feature_order is None:
    if os.path.exists('data/yield_df.csv'):
        try:
            train_df = pd.read_csv('data/yield_df.csv')
            if 'Unnamed: 0' in train_df.columns:
                train_df = train_df.drop(['Unnamed: 0'], axis=1)
            if 'hg/ha_yield' in train_df.columns:
                train_df = train_df.dropna(subset=['hg/ha_yield'])
                X_train = train_df.drop(columns=['hg/ha_yield'])
            else:
                X_train = train_df.copy()
            numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            feature_order = numeric_cols + categorical_cols
            if len(numeric_cols) > 0 and not medians:
                medians = X_train[numeric_cols].median().to_dict()
        except Exception:
            feature_order = None
else:
    if os.path.exists('data/yield_df.csv'):
        try:
            train_df = pd.read_csv('data/yield_df.csv')
            if 'Unnamed: 0' in train_df.columns:
                train_df = train_df.drop(['Unnamed: 0'], axis=1)
            if 'hg/ha_yield' in train_df.columns:
                X_train = train_df.drop(columns=['hg/ha_yield'])
            else:
                X_train = train_df.copy()
            detected_numeric = X_train.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [c for c in feature_order if c in detected_numeric]
            categorical_cols = [c for c in feature_order if c not in numeric_cols]
            if len(numeric_cols) > 0 and not medians and os.path.exists('models/medians.json'):
                with open('models/medians.json') as f:
                    medians = json.load(f)
        except Exception:
            pass

# --- Agentic AI Layers ---

try:
    from transformers import pipeline
    @st.cache_resource
    def load_llm():
        return pipeline("text2text-generation", model="google/flan-t5-base")
    llm_pipeline = load_llm()
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
except Exception:
    LLM_AVAILABLE = False

def analyze_risks(inputs, predicted_yield_tons):
    risks = []
    try:
        rainfall = float(inputs.get('average_rain_fall_mm_per_year', 1000))
    except ValueError:
        rainfall = 1000.0
    try:
        temp = float(inputs.get('avg_temp', 20))
    except ValueError:
        temp = 20.0
    try:
        pesticide = float(inputs.get('pesticides_tonnes', 0))
    except ValueError:
        pesticide = 0.0

    if rainfall < 500:
        risks.append("Low rainfall: High drought risk")
    if temp > 35:
        risks.append("High temperature: Crop heat stress risk")
    if pesticide > 3.0:
        risks.append("High pesticide usage: Soil degradation risk")
    if predicted_yield_tons < 2.0:
        risks.append("Low predicted yield: Production risk")
    return risks

def generate_advice(risks, inputs, predicted_yield_tons):
    if not risks:
        return ["No significant risks identified. Maintain current practices."]
        
    if LLM_AVAILABLE:
        risks_str = ", ".join(risks)
        prompt = f"The predicted crop yield is {predicted_yield_tons:.2f} tons/ha. Identified risks: {risks_str}. Give 3 short farming recommendations."
        try:
            output = llm_pipeline(prompt, max_length=150, do_sample=False)
            advice_text = output[0]['generated_text']
            return [advice_text]
        except Exception:
            pass # fallback to rule-based

    advice = []
    for risk in risks:
        if "drought" in risk.lower():
            advice.append("Increase irrigation frequency; consider drip irrigation")
        elif "heat stress" in risk.lower():
            advice.append("Use shade nets; consider heat-tolerant crop varieties")
        elif "soil degradation" in risk.lower():
            advice.append("Reduce pesticide use; introduce organic alternatives")
        elif "production risk" in risk.lower():
            advice.append("Consult agronomist; review soil nutrients and fertilizer schedule")
    return list(set(advice))

def run_agent(inputs, model, medians_dict):
    # Preprocess
    X = pd.DataFrame([inputs])
    Xp = apply_simple_preprocessing(X)
    Xp_model = align_to_model(Xp, model, medians_dict)
    
    # Predict (assumes model predicts in hg/ha)
    pred_hg_ha = float(model.predict(Xp_model)[0])
    pred_tons_ha = pred_hg_ha / 10000.0
    
    # 1. Analyze Risks
    risks = analyze_risks(inputs, pred_tons_ha)
    
    # 2. Generate Advice
    advice = generate_advice(risks, inputs, pred_tons_ha)
    
    # 3. Structured Report Output
    st.markdown("---")
    st.markdown("## 📄 Agentic Pipeline: Structured Report Output")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌾 Crop Summary")
        st.write(f"**Crop Name:** {inputs.get('Item', 'N/A')}")
        st.write(f"**Region:** {inputs.get('Area', 'N/A')}")
        st.write(f"**Year:** {inputs.get('Year', 'N/A')}")
        with st.expander("View Input Values Used"):
            st.json(inputs)
            
    with col2:
        st.markdown("### 📊 Yield Interpretation")
        if pred_tons_ha < 2.0:
            yield_label = "Low"
            color = "red"
        elif pred_tons_ha < 5.0:
            yield_label = "Medium"
            color = "orange"
        else:
            yield_label = "High"
            color = "green"
        st.markdown(f"**Predicted Yield:** {pred_tons_ha:.2f} tons/ha")
        st.markdown(f"**Yield Level:** :{color}[{yield_label}]")
        
    st.markdown("### ⚠️ Risk Factors")
    if risks:
        for r in risks:
            st.markdown(f"- {r}")
    else:
        st.markdown("- No major risks identified.")
        
    st.markdown("### 💡 Recommended Actions")
    for a in advice:
        st.markdown(f"- {a}")
        
    st.markdown("### 📚 References")
    st.markdown("- FAO Crop Guidelines 2023")
    st.markdown("- ICAR Agronomy Manual")
    
    st.markdown("### ⚖️ Disclaimer")
    st.caption("This report is AI-generated. Consult a certified agronomist before making decisions.")

# --- End Agentic AI Layers ---

mode = st.sidebar.selectbox('Input mode', ['Single record', 'Upload CSV (raw or processed)'])

st.sidebar.markdown('---')
st.sidebar.header('Model artifacts')
metrics_files = []
if os.path.exists('models'):
    for fname in os.listdir('models'):
        if fname.startswith('metrics') and fname.endswith('.json'):
            metrics_files.append(os.path.join('models', fname))

for mf in metrics_files:
    try:
        with open(mf) as f:
            st.sidebar.subheader(os.path.basename(mf))
            st.sidebar.json(f.read())
    except Exception:
        pass

if os.path.exists('models/crop_yield_model_feature_importances.csv'):
    st.sidebar.write('Top linear features:')
    st.sidebar.dataframe(pd.read_csv('models/crop_yield_model_feature_importances.csv').head())
if os.path.exists('models/crop_yield_tree_feature_importances.csv'):
    st.sidebar.write('Top tree features:')
    st.sidebar.dataframe(pd.read_csv('models/crop_yield_tree_feature_importances.csv').head())

def apply_simple_preprocessing(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    for col in (numeric_cols or []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(medians.get(col, 0))
        else:
            df[col] = medians.get(col, 0)
    for col in (categorical_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)
        else:
            df[col] = 'missing'
        if col in label_encoders:
            le = label_encoders[col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                def safe_transform(val):
                    v = str(val)
                    if v in le.classes_:
                        return int(le.transform([v])[0])
                    elif 'missing' in le.classes_:
                        return int(le.transform(['missing'])[0])
                    else:
                        return 0
                df[col] = df[col].apply(safe_transform)
        else:
            df[col] = df[col].astype('category').cat.codes
    if feature_order is not None:
        cols_present = [c for c in feature_order if c in df.columns]
        df = df[cols_present]
    return df

def align_to_model(df, model, medians_dict):
    df2 = df.copy()
    expected = None
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
    elif feature_order is not None:
        expected = feature_order
    if expected is None:
        return df2
    for col in expected:
        if col not in df2.columns:
            df2[col] = medians_dict.get(col, 0)
    df2 = df2.reindex(columns=expected)
    return df2

if mode == 'Single record':
    st.header('Single record prediction & Agentic Pipeline')
    if not feature_order:
        st.warning('Training data/feature order not available. Single-record inputs cannot be generated automatically. Upload a processed CSV instead.')
    else:
        inputs = {}
        st.write('Enter feature values:')
        for col in feature_order:
            if col in categorical_cols:
                if col in label_encoders:
                    classes = list(label_encoders[col].classes_)
                    selection = st.selectbox(col, classes)
                    inputs[col] = selection
                else:
                    inputs[col] = st.text_input(col, value='missing')
            else:
                default_val = float(medians.get(col, 0)) if medians else 0.0
                inputs[col] = st.number_input(col, value=default_val)

        if st.button('Run Agent Pipeline'):
            # prefer tree_model if available
            model_to_use = tree_model if tree_model is not None else linear_model
            if model_to_use is not None:
                run_agent(inputs, model_to_use, medians)
            else:
                st.error("No trained models found in 'models/' directory.")

else:
    st.header('Batch prediction from CSV')
    uploaded_file = st.file_uploader('Upload a CSV file (raw or preprocessed)')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Preview of uploaded data:')
        st.dataframe(df.head())

        try:
            Xp = apply_simple_preprocessing(df)
            if feature_order is not None:
                Xp = Xp.reindex(columns=[c for c in feature_order if c in Xp.columns])
            st.write('Preprocessed preview:')
            st.dataframe(Xp.head())

            preds = {}
            if linear_model is not None:
                Xp_l = align_to_model(Xp, linear_model, medians)
                preds['linear_model'] = linear_model.predict(Xp_l)
            if tree_model is not None:
                Xp_t = align_to_model(Xp, tree_model, medians)
                preds['tree_model'] = tree_model.predict(Xp_t)

            results = df.copy()
            if 'linear_model' in preds:
                results['pred_linear'] = preds['linear_model']
            if 'tree_model' in preds:
                results['pred_tree'] = preds['tree_model']

            st.subheader('Predictions')
            st.dataframe(results.head())

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button('Download predictions as CSV', csv, file_name='predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f'Error during preprocessing or prediction: {e}')
