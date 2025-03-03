import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and feature names
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

# Create a template DataFrame with zeros
template_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Streamlit UI
st.title("🚗 Car Price Prediction App")
st.info("SOME KEY POINTS TO NOTE ABOUT THE MODEL\n1. This model completely does not take into account the current "
        "conditions and specs of cars.\n2. So, suppose you enter 1000 HP/ltr. it will definetly show the price but "
        "it can be misleading, so kindly use the current specs\n3. Below are given some current specs of the car."
        "\n\nHP/ltr range: minimum 30 to maximum 400\n\nTo calculate your annual mileage, you can divide the total "
        "miles you drive in a year by the number of years: on an avg: 10,000 to 15,000 milage per year\n\n"
        "Current no. of engines: min. 2 to max 8\n\nEngine Size: min 1.0 to max. 8(Bugatti Veyron Super Sport).")
# User inputs
brand = st.selectbox("Car Brand", [col.replace("brand_", "") for col in feature_names if "brand_" in col])
fuel_type = st.selectbox("Fuel Type", [col.replace("fuel_type_", "") for col in feature_names if "fuel_type_" in col])
engine_L = st.number_input("Engine Size (L)", min_value=0.5, step=0.1)
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, step=1)
car_age = st.number_input("Car Age (Subtract the launch year from current year)", min_value=0, step=1)
mileage_per_year = st.number_input("Mileage per Year (in thousands)", min_value=0.0)
hp_per_liter = st.number_input("Horsepower per Liter", min_value=0.0)


# Prediction button
if st.button("Predict Price"):
    # Copy the template DataFrame
    data = template_df.copy()

    # Fill numeric features
    numeric_features = {
        "engine_L": engine_L,
        "cylinders": cylinders,
        "car_age": car_age,
        "mileage_per_year": mileage_per_year,
        "hp_per_liter": hp_per_liter
    }

    for feature, value in numeric_features.items():
        data.at[0, feature] = value

    # One-hot encode brand & fuel type
    brand_col, fuel_col = f"brand_{brand}", f"fuel_type_{fuel_type}"

    if brand_col in data.columns:
        data.at[0, brand_col] = 1
    if fuel_col in data.columns:
        data.at[0, fuel_col] = 1

    # Match feature order before prediction
    data = data[feature_names]

    # Predict log(price) and convert to actual price
    log_price = model.predict(data)[0]
    predicted_price = np.expm1(log_price)

    # Display prediction result
    st.success(f"💰 Predicted Price: ${predicted_price:,.2f}")
