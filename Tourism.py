# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load('rating_predictor_model.pkl')
encoders = joblib.load('label_encoders.pkl')

st.title("🎯 Tourist Attraction Rating Predictor")

st.markdown("Enter your travel details and get a predicted satisfaction rating (1-5).")

# === INPUT FIELDS ===
visit_year = st.selectbox("Visit Year", list(range(2015, 2026)))
visit_month = st.selectbox("Visit Month", list(range(1, 13)))

# User Location Info
continent = st.selectbox("Continent", encoders['Continent'].classes_)
region = st.selectbox("Region", encoders['Region'].classes_)
country = st.selectbox("Country", encoders['Country'].classes_)
city = st.selectbox("City", encoders['CityName'].classes_)

# Attraction Info
attraction_type = st.selectbox("Attraction Type", encoders['AttractionType'].classes_)
attraction_type_id = st.number_input("Attraction Type ID", min_value=1, max_value=50, value=1)
attraction_city_id = st.number_input("Attraction City ID", min_value=1, max_value=1000, value=100)

# Visit Mode
visit_mode = st.selectbox("Visit Mode", encoders['VisitMode'].classes_)

# === PREDICT BUTTON ===
if st.button("Predict Rating"):
    # Encode values
    continent_enc = encoders['Continent'].transform([continent])[0]
    region_enc = encoders['Region'].transform([region])[0]
    country_enc = encoders['Country'].transform([country])[0]
    city_enc = encoders['CityName'].transform([city])[0]
    attraction_type_enc = encoders['AttractionType'].transform([attraction_type])[0]
    visit_mode_enc = encoders['VisitMode'].transform([visit_mode])[0]

    # Prepare feature vector
    input_features = pd.DataFrame([[
        visit_year, visit_month,
        continent_enc, region_enc, country_enc, city_enc,
        attraction_city_id, attraction_type_id,
        visit_mode_enc, continent_enc, region_enc,
        country_enc, city_enc, attraction_type_enc
    ]], columns=[
        'VisitYear', 'VisitMonth', 'ContinentId', 'RegionId', 'CountryId', 'CityId',
        'AttractionCityId', 'AttractionTypeId', 'VisitMode', 'Continent', 'Region',
        'Country', 'CityName', 'AttractionType'
    ])

    # Predict
    prediction = model.predict(input_features)[0]
    st.success(f"⭐ Predicted Rating: {prediction:.2f} / 5")
