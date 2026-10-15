
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

model = joblib.load("mobile_price_prediction.pkl")

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Predict medical insurance charges using Machine Learning.")
st.divider()

st.subheader("Enter Patient Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)
st.caption("Allowed range: 18–100 years")

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)
st.caption("Allowed range: 10–70")

st.divider()

if st.button("Predict Insurance Cost", use_container_width=True):

    if not (18 <= age <= 100):
        st.error("Age must be between 18 and 100.")

    elif not (10 <= bmi <= 70):
        st.error("BMI must be between 10 and 70.")

    else:
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
