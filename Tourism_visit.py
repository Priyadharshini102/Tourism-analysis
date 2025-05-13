# app_visit_mode.py

import streamlit as st
import pandas as pd
import joblib

# === Load Model and Encoders ===
model = joblib.load("visit_mode_classifier.pkl")
visit_mode_encoder = joblib.load("visit_mode_encoder.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")

st.title("🧳 Visit Mode Predictor")
st.markdown("Enter user and attraction details to predict the travel purpose (Business, Family, Couples, Friends, etc.)")

# === User Inputs ===
visit_year = st.selectbox("Visit Year", list(range(2015, 2026)))
visit_month = st.selectbox("Visit Month", list(range(1, 13)))

# Use label-encoded fields from training
continent = st.selectbox("Continent", feature_encoders['Continent'].classes_)
region = st.selectbox("Region", feature_encoders['Region'].classes_)
country = st.selectbox("Country", feature_encoders['Country'].classes_)
city_name = st.selectbox("City Name", feature_encoders['CityName'].classes_)
attraction_type = st.selectbox("Attraction Type", feature_encoders['AttractionType'].classes_)

# === Predict Button ===
if st.button("Predict Visit Mode"):
    # Encode text features
    continent_enc = feature_encoders['Continent'].transform([continent])[0]
    region_enc = feature_encoders['Region'].transform([region])[0]
    country_enc = feature_encoders['Country'].transform([country])[0]
    city_enc = feature_encoders['CityName'].transform([city_name])[0]
    attraction_type_enc = feature_encoders['AttractionType'].transform([attraction_type])[0]

    # Prepare input features
    input_features = pd.DataFrame([[
        visit_year, visit_month,
        continent_enc, region_enc, country_enc,
        city_enc, attraction_type_enc
    ]], columns=[
        'VisitYear', 'VisitMonth','Continent', 'Region',
        'Country', 'CityName', 'AttractionType'
    ])

    # Predict
    pred_encoded = model.predict(input_features)[0]
    pred_label = visit_mode_encoder.inverse_transform([pred_encoded])[0]

    # Display result
    st.success(f"🎯 Predicted Visit Mode: **{pred_label}**")
