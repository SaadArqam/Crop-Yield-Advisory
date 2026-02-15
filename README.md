# 🌾 Intelligent Crop Yield Prediction and Agentic Farm Advisory System

## 📌 Project Overview

This project is an AI-driven agricultural analytics system that:

1. Predicts crop yield using historical agricultural data.
2. Extends predictions into an Agentic AI-based farm advisory assistant.
3. Generates structured crop management recommendations.

The system combines supervised machine learning with agent-based reasoning to support data-driven agricultural decisions.

---

## 🎯 Project Objectives

- Predict crop yield using farm, soil, and seasonal data.
- Identify key factors influencing crop production.
- Provide a user-friendly prediction interface.
- Extend predictions into actionable farming advice.

---

## 🧠 System Architecture

### Milestone 1: ML-Based Yield Prediction

Data → Preprocessing → Model Training → Evaluation → Model Saving → UI Prediction

### Milestone 2: Agentic AI Advisory (Planned/Implemented)

User Input → Yield Prediction → Risk Analysis → Knowledge Retrieval → Advisory Report Generation

---

## 📂 Project Structure

crop-yield-advisory/
│
├── data/ # Local dataset (not pushed to GitHub)
├── ml/ # Data preprocessing & model training
├── models/ # Saved trained models (.pkl)
├── agent/ # Agentic AI logic (Milestone 2)
├── app.py # Streamlit UI
├── requirements.txt
└── README.md

---

## 📊 Dataset

Source:
Kaggle Crop Yield Prediction Dataset

The dataset includes:
- Crop type
- Season
- Area
- Year
- Rainfall (optional)
- Temperature (optional)
- Pesticide usage (optional)
- Yield (target variable)

---

# to run locally
```
python3 <filename>
```

## Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

# install the requirements
```
python3 -m pip install -r requirements.txt
```

