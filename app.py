import streamlit as st
import joblib
import numpy as np
import pandas as pd
model = joblib.load("car_price_model.pkl")
feature_names = joblib.load("feature_names.pkl")
brands = ['Audi', 'BMW', 'Mercedes', 'Toyota', 'Ford', 'Honda', 'Tesla']
st.title("🚗 Car Price Prediction App")
user_input = {}
selected_brand = st.selectbox("Select Car Brand", brands)
for brand in brands:
    user_input[f"brand_{brand}"] = 1 if brand == selected_brand else 0
user_input["car_age"] = st.number_input("Enter Car Age (years)", min_value=0, max_value=50, value=5)
user_input["horse_power_per_L"] = st.number_input("Enter Horsepower per Liter", min_value=50.0, max_value=1000.0, value=200.0)
user_input["mileage_per_year"] = st.number_input("Enter Mileage per Year", min_value=0.0, max_value=100000.0, value=15000.0)
user_input["cylinders"] = st.number_input("Enter Number of Cylinders", min_value=2, max_value=16, value=4)
input_df = pd.DataFrame([user_input])
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]
if st.button("💰 Predict Price"):
    prediction = model.predict(input_df)
    actual_price = np.exp(prediction[0])
    st.success(f"🚙 Estimated Car Price: **${actual_price:,.2f}**")
