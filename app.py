
import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load trained model
model = joblib.load("mobile_price_prediction.pkl")

# App title
st.title("🏥 Medical Insurance Cost Predictor")
st.write(
    "Predict medical insurance charges using Machine Learning."
)

st.divider()
st.subheader("Enter Patient Details")

# Age input - no maximum limit
age = st.number_input(
    "Age (years)",
    min_value=0,
    value=30,
    step=1,
    help="Suggested range: 0–100 years"
)

st.caption("Suggested range: 0–100 years")

# BMI input - no maximum limit
bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0,
    step=0.1,
    help="Suggested range: 0.0–100.0"
)

st.caption("Suggested range: 0.0–100.0")

st.divider()

# Prediction
if st.button("Predict Insurance Cost", use_container_width=True):

    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")

        st.metric(
            label="Predicted Insurance Charges",
            value=f"${prediction:,.2f}"
        )

        st.caption(
            "This is an estimated prediction, not an actual insurance quote."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()
st.caption(
    "Machine Learning Project | Multiple Linear Regression"
)
