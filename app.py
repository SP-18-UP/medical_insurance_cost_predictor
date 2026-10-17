
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

# Patient details
st.subheader("Enter Patient Details")

# Age input
age = st.number_input(
    "Age (years)",
    min_value=0,
    value=30,
    step=1,
    help="Suggested range: 0–100 years"
)

st.caption("Suggested range: 0–100 years")

# BMI input
bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0,
    step=0.1,
    help="Suggested range: 0.0–100.0"
)

st.caption("Suggested range: 0.0–100.0")

st.divider()

# Prediction button
if st.button(
    "Predict Insurance Cost",
    use_container_width=True
):

    # Prepare input data
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi]
    })

    try:
        # Predict insurance charges
        prediction = model.predict(input_data)[0]

        # Convert USD to INR
        usd_to_inr = 94.96
        prediction_inr = prediction * usd_to_inr

        # Display prediction
        st.success("Prediction completed!")

        st.metric(
            label="Predicted Insurance Charges",
            value=f"₹{prediction_inr:,.2f}"
        )

        st.caption(
            "Estimated amount in Indian Rupees. "
            "Conversion uses ₹94.96 per USD."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Footer
st.divider()

st.caption(
    "Machine Learning Project | Multiple Linear Regression"
)
