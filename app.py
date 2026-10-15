
import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("mobile_price_prediction.pkl")

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Predict medical insurance charges using Machine Learning.")
st.divider()

# Input fields
st.subheader("Enter Patient Details")

age = st.number_input(
    "Age (years)",
    min_value=18,
    max_value=100,
    value=30
)

bmi = st.number_input(
    "BMI (Body Mass Index)",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

# Prediction button
if st.button("Predict Insurance Cost", use_container_width=True):

    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Predicted Insurance Charges",
        value=f"${prediction:,.2f}"
    )

    st.caption(
        "This is an estimated prediction, not an actual insurance quote."
    )

st.divider()
st.caption("Machine Learning Project | Multiple Linear Regression")
