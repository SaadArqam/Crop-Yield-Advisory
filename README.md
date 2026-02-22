# 🌾 Crop Yield Advisory — Milestone 1 (ML-Based Yield Prediction)

This repository implements Milestone 1 of the Intelligent Crop Yield Prediction and Agentic Farm Advisory System.

Milestone 1 goal (short): build a reproducible machine learning pipeline that
- preprocesses historical farm/weather data,
- trains supervised models to predict crop yield,
- evaluates model performance using regression metrics,
- and provides a simple user interface to run predictions (single record or batch CSV).

This README documents what is implemented, how to run it locally, how to verify the results, and suggested next steps toward Milestone 2.

---

## What is implemented (Milestone 1)

- Data preprocessing and explicit training pipeline: `ml/train_pipeline.py`
  - Numeric features: median imputation (explicit, pandas-based).
  - Categorical features: missing->'missing' then `LabelEncoder` (explicit and saved so the UI can reuse encodings).
  - Saves preprocessing artifacts used at inference time: `models/feature_order.json` and `models/medians.json`.
  - Saves label encoders to `models/label_encoders_pipeline.pkl`.

- Model training and explainability:
  - `ml/train_pipeline.py` trains both:
    - `LinearRegression` → saved to `models/crop_yield_model.pkl`.
    - `DecisionTreeRegressor` → saved to `models/crop_yield_tree.pkl`.
  - Evaluation metrics are saved to `models/metrics_<model>.json`.
  - Feature importance / coefficients are exported to CSV under `models/`.

- Streamlit UI: `app.py`
  - Single-record prediction mode with form inputs (UI shows categorical options when encoders are available).
  - Batch prediction mode (CSV upload) — app applies the same simple preprocessing used during training.
  - The app now correctly aligns preprocessed inputs to the exact feature order expected by each saved model (prevents the common "feature names must match" error).
  - Sidebar shows saved metrics and top feature importances.

- Packaging: `requirements.txt` lists the Python packages required to run the pipeline and UI.

Files produced by the pipeline (under `models/`):
- `crop_yield_model.pkl` — trained linear model
- `crop_yield_tree.pkl` — trained decision tree model
- `metrics_crop_yield_model.json` — linear model evaluation
- `metrics_crop_yield_tree.json` — tree model evaluation
- `crop_yield_model_feature_importances.csv` — linear model coefficients
- `crop_yield_tree_feature_importances.csv` — tree model importances
- `label_encoders_pipeline.pkl` — saved LabelEncoder objects used during preprocessing
- `feature_order.json` — exact feature order used for training (used by the app to reindex inputs)
- `medians.json` — numeric medians used for imputation during training

---

## Quickstart — run Milestone 1 locally (macOS / zsh)

1. Clone the repository and change directory (if you haven't already):

```bash
cd /path/to/crop-yield-advisory
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

4. (Re)generate preprocessing artifacts and models:

```bash
python3 ml/train_pipeline.py
```

This script performs explicit preprocessing, trains the models, and writes all artifacts to `models/` (including `feature_order.json` and `medians.json`).

5. Run the Streamlit UI locally to make predictions:

```bash
streamlit run app.py
```

Open the displayed URL (usually `http://localhost:8501`) in your browser. Use the Single record form or upload a CSV to get predictions.

---

## How to interpret predictions (units)

- Models predict the column `hg/ha_yield` (hectograms per hectare).
  - 1 hg = 100 g = 0.1 kg.
  - To convert model output to tonnes per hectare (t/ha):
    - t/ha = (hg/ha) * 0.1 / 1000 = (hg/ha) * 0.0001
  - Example: a prediction of 100000 hg/ha → 100000 * 0.0001 = 10 t/ha.

The UI currently displays raw predictions (hg/ha). You may convert them in your presentation or I can update the UI to show converted values.

---

## Verification checklist (Milestone 1)

- [x] Data preprocessing exists and produces preprocessing artifacts — `ml/train_pipeline.py` writes `models/medians.json`, `models/feature_order.json`, and `label_encoders_pipeline.pkl`.
- [x] Models train and save artifacts — `ml/train_pipeline.py` saves models and metrics under `models/`.
- [x] Evaluation metrics are produced — check `models/metrics_crop_yield_model.json` and `models/metrics_crop_yield_tree.json`.
- [x] UI is present and can run locally — `streamlit run app.py` (single-record and CSV modes).
- [x] UI applies the same preprocessing and aligns inputs to the models' expected feature order (prevents feature-name mismatch error).

---

## Troubleshooting & common issues

- "The feature names should match those that were passed during fit": this is fixed by the app aligning the preprocessed input to each model's `feature_names_in_` (or `models/feature_order.json`) and by adding any missing features with medians/defaults. If you still see that message, ensure `models/feature_order.json` is present and up-to-date (re-run `ml/train_pipeline.py`).

- If categorical values in the UI are unseen by the saved encoders, the app maps unseen values to a fallback class (preferably the 'missing' class if it exists). For the cleanest results, re-run training using the dataset you will evaluate with.

---

## Next steps toward Milestone 2 (Agentic Advisory)

To extend this work into Milestone 2, typical next items are:

- Integrate an open-source LLM or free-tier API for natural language reasoning (only after Milestone 1 is stable).
- Build a retrieval layer (optionally a vector store like FAISS/Chroma) to provide agronomy references and best-practices during advice generation.
- Implement an agentic workflow that consumes model predictions + drivers and produces a structured advisory report (crop summary, identified risks, recommended actions, references, disclaimers).
- Add CV/hyperparameter tuning to reduce overfitting (recommended for decision tree), and produce an evaluation notebook with plots.

---

## Contributors

Saad Arqam
Priyabrata Singh
Pathan Amaan
Manu Pal

---

If you want, I can now:
- Convert displayed predictions to tonnes/ha and show rounded values in the UI, or
- Add a temporary debug panel in the UI showing `model.feature_names_in_` and the preprocessed input before predict (handy for viva), or
- Add CV/GridSearch for the tree model and present CV metrics in the sidebar.

Tell me which you'd like next and I will implement it.

