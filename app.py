import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("🌾 Crop Yield Prediction — Milestone 1 UI")
st.write("Upload a dataset or enter single-record values to get yield predictions. This UI expects the preprocessed features used during training (see `ml/preprocess.py`).")

# load available artifacts
model = None
tree_model = None
label_encoders = {}
processed_cols = None

if os.path.exists('models/crop_yield_model.pkl'):
    model = joblib.load('models/crop_yield_model.pkl')
if os.path.exists('models/crop_yield_tree.pkl'):
    tree_model = joblib.load('models/crop_yield_tree.pkl')
if os.path.exists('models/label_encoders.pkl'):
    label_encoders = joblib.load('models/label_encoders.pkl')

if os.path.exists('data/processed_data.csv'):
    processed = pd.read_csv('data/processed_data.csv')
    if 'hg/ha_yield' in processed.columns:
        processed_cols = list(processed.drop(columns=['hg/ha_yield']).columns)
    else:
        processed_cols = list(processed.columns)
else:
    processed = None

mode = st.sidebar.selectbox('Input mode', ['Single record', 'Upload CSV (raw or processed)'])

def prepare_input_df(single_inputs):
    # single_inputs is a dict feature->value
    return pd.DataFrame([single_inputs])

if mode == 'Single record':
    st.header('Single record prediction')
    if processed_cols is None:
        st.warning('No processed_data.csv found. Please run ml/preprocess.py first or upload a processed CSV.')
    else:
        inputs = {}
        st.write('Enter feature values:')
        for col in processed_cols:
            if col in label_encoders:
                # show human-readable options
                le = label_encoders[col]
                classes = list(le.classes_)
                selection = st.selectbox(col, classes)
                # transform selection to encoded value
                encoded = int(le.transform([selection])[0])
                inputs[col] = encoded
            else:
                # numeric input
                mean_val = float(processed[col].mean()) if processed is not None else 0.0
                val = st.number_input(col, value=mean_val)
                inputs[col] = val

        if st.button('Predict'):
            X = prepare_input_df(inputs)
            res = {}
            if model is not None:
                pred = model.predict(X)[0]
                res['linear_model'] = float(pred)
            if tree_model is not None:
                pred_t = tree_model.predict(X)[0]
                res['tree_model'] = float(pred_t)
            st.subheader('Prediction')
            st.write(res)

else:
    st.header('Batch prediction from CSV')
    uploaded_file = st.file_uploader('Upload a CSV file (raw or preprocessed)')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Preview of uploaded data:')
        st.dataframe(df.head())

        # try to detect if raw (has categorical names) and apply encoders
        for col, le in label_encoders.items():
            if col in df.columns and df[col].dtype == object:
                try:
                    df[col] = le.transform(df[col].astype(str))
                    st.write(f'Applied label encoding to {col}')
                except Exception as e:
                    st.write(f'Could not encode column {col}: {e}')

        # ensure columns match processed_cols
        if processed_cols is not None:
            missing = [c for c in processed_cols if c not in df.columns]
            if missing:
                st.error(f'Missing required features: {missing}. The uploaded file should contain these columns (or upload a processed CSV).')
            else:
                X = df[processed_cols]
                preds = {}
                if model is not None:
                    preds['linear_model'] = model.predict(X)
                if tree_model is not None:
                    preds['tree_model'] = tree_model.predict(X)

                # assemble results
                results = df.copy()
                if 'linear_model' in preds:
                    results['pred_linear'] = preds['linear_model']
                if 'tree_model' in preds:
                    results['pred_tree'] = preds['tree_model']

                st.subheader('Predictions')
                st.dataframe(results.head())

                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions as CSV', csv, file_name='predictions.csv', mime='text/csv')

# show feature importances if present
st.sidebar.markdown('---')
st.sidebar.header('Model artifacts')
if os.path.exists('models/linear_feature_importances.csv'):
    st.sidebar.write('Top linear features:')
    lf = pd.read_csv('models/linear_feature_importances.csv')
    st.sidebar.dataframe(lf.head())
if os.path.exists('models/tree_feature_importances.csv'):
    st.sidebar.write('Top tree features:')
    tf = pd.read_csv('models/tree_feature_importances.csv')
    st.sidebar.dataframe(tf.head())

st.sidebar.write('Saved metrics:')
if os.path.exists('models/metrics_linear.json'):
    st.sidebar.json(open('models/metrics_linear.json').read())
if os.path.exists('models/metrics_tree.json'):
    st.sidebar.json(open('models/metrics_tree.json').read())
