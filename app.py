import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load model and scaler
# ===============================
model = joblib.load("random_forest_diabetes_model.pkl")
scaler = joblib.load("scaler (2).pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Predict diabetes risk using a Machine Learning model")
st.markdown("---")

# ===============================
# User Input Section
# ===============================
st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=40)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "No Info"]
)

# ===============================
# Convert Inputs to Model Format
# ===============================
input_data = {
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose,
    "gender_Male": 1 if gender == "Male" else 0,
    "smoking_history_current": 1 if smoking == "current" else 0,
    "smoking_history_former": 1 if smoking == "former" else 0,
    "smoking_history_never": 1 if smoking == "never" else 0,
    "smoking_history_No Info": 1 if smoking == "No Info" else 0
}

input_df = pd.DataFrame([input_data])

# ===============================
# Prediction
# ===============================
if st.button("Predict Diabetes"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ Prediction: **Diabetic**")
        st.write(f"Risk Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ Prediction: **Not Diabetic**")
        st.write(f"Risk Probability: **{probability:.2%}**")

# ===============================
# Disclaimer
# ===============================
st.markdown("---")
st.warning(
    "⚠️ This application is for educational purposes only and "
    "should not be used for real medical diagnosis."
)
