import streamlit as st
import pandas as pd
import joblib


model = joblib.load("models/crop_yield_tree.pkl")

st.title("🌾 Intelligent Crop Yield Prediction")

st.write("Enter crop details to predict yield.")


area = st.number_input("Area", min_value=0.0)
year = st.number_input("Year", min_value=1900)
crop = st.number_input("Crop (encoded value)")
season = st.number_input("Season (encoded value)")

if st.button("Predict Yield"):
    input_data = pd.DataFrame([[area, year, crop, season]],
                              columns=["Area", "Year", "Crop", "Season"])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Yield: {prediction:.2f}")
