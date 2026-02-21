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

- Data preprocessing: `ml/preprocess.py`
  - Drops NaNs and an optional `Unnamed: 0` column if present.
  - Label-encodes categorical columns and saves the encoders to `models/label_encoders.pkl`.
  - Writes processed CSV to `data/processed_data.csv`.

- Model training:
  - `ml/train_model.py` — trains a Linear Regression model, saves the model to `models/crop_yield_model.pkl`, writes evaluation metrics to `models/metrics_linear.json`, and exports linear coefficients to `models/linear_feature_importances.csv`.
  - `ml/train_tree_model.py` — trains a Decision Tree Regressor, saves the model to `models/crop_yield_tree.pkl`, writes evaluation metrics to `models/metrics_tree.json`, and exports feature importances to `models/tree_feature_importances.csv`.

- Simple UI: `app.py` (Streamlit)
  - Single-record prediction mode (enter feature values or select categorical values using saved label encoders).
  - Batch prediction mode (upload a CSV — raw or preprocessed). The app attempts to apply saved encoders to categorical columns in uploaded files.
  - Sidebar displays saved metrics and top feature importances.

- Packaging: `requirements.txt` contains the Python packages required to run the pipeline and UI.

Files produced by the pipeline (under `models/`):
- `crop_yield_model.pkl` — trained linear model
- `crop_yield_tree.pkl` — trained decision tree model
- `metrics_linear.json` — linear model evaluation
- `metrics_tree.json` — tree model evaluation
- `linear_feature_importances.csv` — linear model coefficients
- `tree_feature_importances.csv` — tree model importances
- `label_encoders.pkl` — saved LabelEncoder objects used during preprocessing

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

4. Preprocess the raw data (this reads `data/yield_df.csv` and writes `data/processed_data.csv`):

```bash
python3 ml/preprocess.py
```

Expected console output:
```
Data preprocessing done
Processed shape: (N, M)
```
Where `N` is the number of rows after dropping NaNs and `M` is the number of features.

5. Train the linear regression model (and save metrics & coefficients):

```bash
python3 ml/train_model.py
```

Expected console output (example):
```
Model Evaluation:
MAE: 62444.3106
RMSE: 81501.7644
R2 Score: 0.08
Saved linear feature importances to models/linear_feature_importances.csv
Model saved.
```

After this completes, check `models/` for `crop_yield_model.pkl` and `metrics_linear.json`.

6. (Optional but recommended) Train the decision tree model:

```bash
python3 ml/train_tree_model.py
```

Expected console output (example):
```
Decision Tree Evaluation:
MAE: 4045.1174
RMSE: 13224.3041
R2: 0.97
Decision Tree model saved.
Saved tree feature importances to models/tree_feature_importances.csv
```

7. Run the Streamlit UI locally to make predictions:

```bash
streamlit run app.py
```

Open the displayed URL (usually `http://localhost:8501`) in your browser. Use the sidebar and main controls to enter a single record or upload a CSV to get predictions.

---

## How to verify Milestone 1 is satisfied

Checklist and where to look:

- [x] Data preprocessing exists and produces `data/processed_data.csv` — run `ml/preprocess.py` and confirm file exists.
- [x] Models train and save artifacts — run `ml/train_model.py` and `ml/train_tree_model.py` and confirm files in `models/`.
- [x] Evaluation metrics are produced — check `models/metrics_linear.json` and `models/metrics_tree.json`.
- [x] Basic UI is present and can run locally — run `streamlit run app.py` and test single-record and CSV modes.
- [ ] (Recommended improvement) Add cross-validation and a held-out test evaluation to detect overfitting (decision tree shows very high R2 in simple train/test split; this can be overfitting).
- [ ] (Recommended improvement) More robust preprocessing (numeric imputation, scaling, outlier handling, and feature engineering).

---

## Notes, caveats & quick observations

- The current training scripts use a single `train_test_split` split (random_state=42). The decision tree model may overfit the training data; consider using cross-validation, limiting tree depth, or using ensemble methods (RandomForest, Gradient Boosting) with proper validation.

- Preprocessing currently drops any row with missing values. If your datasets have many missing values, consider imputation instead of dropping rows.

- `ml/preprocess.py` saves label encoders to `models/label_encoders.pkl`. This lets the UI accept categorical strings and convert them to the numeric codes expected by the models.

- Feature importance is provided for both models. For Linear Regression the CSV contains coefficients; for the decision tree the CSV contains scikit-learn feature importances.

---

## Next steps toward Milestone 2 (Agentic Advisory)

To extend this work into Milestone 2, typical next items are:

- Integrate an open-source LLM or free-tier API for natural language reasoning.
- Build a retrieval layer (optionally a vector store like FAISS/Chroma) to provide agronomy references and best-practices during advice generation.
- Implement an agentic workflow (explicit state management) that:
  - accepts model predictions + feature drivers,
  - retrieves supporting agronomy material,
  - assembles a structured advisory report (crop summary, risk factors, actions, references, disclaimers).
- Add PDF export, seasonal planning, or fertilizer/irrigation optimization as extensions.

---

## Contributors

Saad Arqam
Priyabrata Singh
Pathan Amaan
Manu Pal



---

