import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Page settings
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your health details below. The app predicts if you may have a heart condition.</p>", unsafe_allow_html=True)

# Input form
age = st.number_input("Age", 1, 120, 25)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
restecg = st.number_input("Resting ECG Results (0–2)", 0, 2, 1)
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0–2)", 0, 2, 1)
ca = st.number_input("Major Vessels Colored (0–3)", 0, 3, 0)
thal = st.number_input("Thalassemia (0–3)", 0, 3, 1)

# Predict button
if st.button("🔍 Predict"):
    data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("💔 You may have heart disease. Please consult a doctor.")
    else:
        st.success("💖 You are healthy. No signs of heart disease detected.")
