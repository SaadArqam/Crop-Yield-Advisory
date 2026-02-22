import streamlit as st
import pandas as pd
import joblib
import os
import json

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("🌾 Crop Yield Prediction — Milestone 1 UI")
st.write("Upload a dataset or enter single-record values to get yield predictions. The app applies the same simple preprocessing used during training: numeric median imputation and label encoding for categoricals.")

# load models
linear_model = None
tree_model = None
label_encoders = {}

if os.path.exists('models/crop_yield_model.pkl'):
    linear_model = joblib.load('models/crop_yield_model.pkl')
if os.path.exists('models/crop_yield_tree.pkl'):
    tree_model = joblib.load('models/crop_yield_tree.pkl')

# prefer the label encoders saved by the simple trainer
if os.path.exists('models/label_encoders_pipeline.pkl'):
    label_encoders = joblib.load('models/label_encoders_pipeline.pkl')
elif os.path.exists('models/label_encoders.pkl'):
    label_encoders = joblib.load('models/label_encoders.pkl')

# Load medians and feature order saved by the trainer (preferred)
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

# Fallback: derive from training CSV if saved order is not present
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
    # set cols lists from feature_order
    # assume numeric columns appeared first as written during training
    # try to detect numeric names by checking train CSV types
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
            # build numeric and categorical lists by intersection
            numeric_cols = [c for c in feature_order if c in detected_numeric]
            categorical_cols = [c for c in feature_order if c not in numeric_cols]
            if len(numeric_cols) > 0 and not medians and os.path.exists('models/medians.json'):
                with open('models/medians.json') as f:
                    medians = json.load(f)
        except Exception:
            pass

mode = st.sidebar.selectbox('Input mode', ['Single record', 'Upload CSV (raw or processed)'])

st.sidebar.markdown('---')
st.sidebar.header('Model artifacts')
# show metrics if present (try both naming conventions)
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

# show feature importances if present
if os.path.exists('models/crop_yield_model_feature_importances.csv'):
    st.sidebar.write('Top linear features:')
    st.sidebar.dataframe(pd.read_csv('models/crop_yield_model_feature_importances.csv').head())
if os.path.exists('models/crop_yield_tree_feature_importances.csv'):
    st.sidebar.write('Top tree features:')
    st.sidebar.dataframe(pd.read_csv('models/crop_yield_tree_feature_importances.csv').head())

# Helper: apply simple preprocessing to a dataframe (in-place copy)
def apply_simple_preprocessing(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    # Fill numeric medians
    for col in (numeric_cols or []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(medians.get(col, 0))
        else:
            df[col] = medians.get(col, 0)
    # Fill and encode categorical
    for col in (categorical_cols or []):
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)
        else:
            df[col] = 'missing'
        # apply saved label encoder if available
        if col in label_encoders:
            le = label_encoders[col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                # fallback: if unseen values exist, map them to 'missing' class if available
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
            # if no encoder saved, attempt simple label encoding on the fly
            df[col] = df[col].astype('category').cat.codes
    # ensure column order matches training
    if feature_order is not None:
        # only keep columns that are present in df
        cols_present = [c for c in feature_order if c in df.columns]
        df = df[cols_present]
    return df

# Helper to align to a model's expected features
def align_to_model(df, model, medians_dict):
    df2 = df.copy()
    expected = None
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
    elif feature_order is not None:
        expected = feature_order
    if expected is None:
        return df2
    # Add any missing columns with median/defaults
    for col in expected:
        if col not in df2.columns:
            # prefer medians for numeric; else fill with 0
            df2[col] = medians_dict.get(col, 0)
    # Reindex to expected order
    df2 = df2.reindex(columns=expected)
    return df2

if mode == 'Single record':
    st.header('Single record prediction')
    if not feature_order:
        st.warning('Training data/feature order not available. Single-record inputs cannot be generated automatically. Upload a processed CSV instead.')
    else:
        inputs = {}
        st.write('Enter feature values:')
        for col in feature_order:
            if col in categorical_cols:
                # try to show human-readable options
                if col in label_encoders:
                    classes = list(label_encoders[col].classes_)
                    selection = st.selectbox(col, classes)
                    inputs[col] = selection
                else:
                    inputs[col] = st.text_input(col, value='missing')
            else:
                default_val = float(medians.get(col, 0)) if medians else 0.0
                inputs[col] = st.number_input(col, value=default_val)

        if st.button('Predict'):
            X = pd.DataFrame([inputs])
            Xp = apply_simple_preprocessing(X)
            res = {}
            if linear_model is not None:
                try:
                    Xp_l = align_to_model(Xp, linear_model, medians)
                    res['linear_model'] = float(linear_model.predict(Xp_l)[0])
                except Exception as e:
                    res['linear_model_error'] = str(e)
            if tree_model is not None:
                try:
                    Xp_t = align_to_model(Xp, tree_model, medians)
                    res['tree_model'] = float(tree_model.predict(Xp_t)[0])
                except Exception as e:
                    res['tree_model_error'] = str(e)
            st.subheader('Prediction')
            st.write(res)

else:
    st.header('Batch prediction from CSV')
    uploaded_file = st.file_uploader('Upload a CSV file (raw or preprocessed)')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Preview of uploaded data:')
        st.dataframe(df.head())

        # Apply preprocessing
        try:
            Xp = apply_simple_preprocessing(df)
            # reindex to saved feature_order (keep only present columns)
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
