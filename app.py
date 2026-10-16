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
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

st.caption("Allowed range: 18–100 years")

# BMI input
bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

st.caption("Allowed range: 10–70")

st.divider()

# Prediction
if st.button(
    "Predict Insurance Cost",
    use_container_width=True
):

    # Prepare input data
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi]
    })

    # Predict insurance charges
    prediction = model.predict(input_data)[0]

    # Display result
    st.success("Prediction completed!")

    st.metric(
        label="Predicted Insurance Charges",
        value=f"${prediction:,.2f}"
    )

    st.caption(
        "This is an estimated prediction, not an actual insurance quote."
    )

# Footer
st.divider()

st.caption(
    "Machine Learning Project | Multiple Linear Regression"
)
