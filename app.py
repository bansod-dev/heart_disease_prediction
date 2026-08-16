import streamlit as st
import pandas as pd
import joblib

st.title("TEST APP")

model = joblib.load("heart_prediction.pkl")
scaler = joblib.load("scalar.pkl")
expected_columns = joblib.load("columns.pkl")

st.success("All files loaded successfully!")

st.title("Heart Disease Prediction App")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120) 
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LV Hypertrophy"])
max_hr = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


if st.button("Predict"):
    raw_input = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,   
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope
    }
    
    input_df = pd.DataFrame([raw_input])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]        
    
    
    if prediction == 1:
        st.error("HIGH RISK OF HEART DISEASE!")
    else:
        st.success("LOW RISK OF HEART DISEASE!")