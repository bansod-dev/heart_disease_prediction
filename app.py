import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("heart_prediction.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# =========================================================
# TITLE
# =========================================================

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the following information to generate a prediction "
    "using a trained Logistic Regression model."
)

# =========================================================
# INPUTS
# =========================================================

age = st.slider(
    "Age",
    18,
    100,
    40
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80,
    200,
    120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)",
    100,
    600,
    200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LV Hypertrophy"]
)

max_hr = st.number_input(
    "Maximum Heart Rate Achieved",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Predict"):

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = {

        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        # Sex
        "Sex_F": 1 if sex == "F" else 0,
        "Sex_M": 1 if sex == "M" else 0,

        # Chest Pain
        "ChestPainType_ASY": 1 if chest_pain == "ASY" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,

        # Resting ECG
        "RestingECG_LVH": 1 if resting_ecg == "LV Hypertrophy" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,

        # Exercise Angina
        "ExerciseAngina_N": 1 if exercise_angina == 0 else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == 1 else 0,

        # ST Slope
        "ST_Slope_Down": 1 if st_slope == "Down" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }

    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame([input_data])

    # Make sure columns are in exactly the same
    # order as during training
    input_df = input_df[expected_columns]

    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    scaled_input = scaler.transform(input_df)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_input)[0]

    probabilities = model.predict_proba(scaled_input)[0]

    class_0_probability = probabilities[0]
    class_1_probability = probabilities[1]

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.subheader("Prediction")

    if prediction == 1:

        st.error("⚠️ HIGH RISK")

        st.metric(
            "Class 1 Probability",
            f"{class_1_probability * 100:.2f}%"
        )

    else:

        st.success("✅ LOW RISK")

        st.metric(
            "Class 0 Probability",
            f"{class_0_probability * 100:.2f}%"
        )

    # =====================================================
    # PROBABILITY
    # =====================================================

    st.subheader("📊 Model Probability")

    probability_df = pd.DataFrame({
        "Class": model.classes_,
        "Probability (%)": probabilities * 100
    })

    probability_df["Probability (%)"] = (
        probability_df["Probability (%)"].round(2)
    )

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # FEATURE CONTRIBUTIONS
    # =====================================================

    st.markdown("---")

    st.subheader("🔍 Prediction Explanation")

    # Logistic Regression coefficients
    coefficients = model.coef_[0]

    # Contribution of each feature
    contributions = scaled_input[0] * coefficients

    contribution_df = pd.DataFrame({
        "Feature": expected_columns,
        "Contribution": contributions
    })

    # Round
    contribution_df["Contribution"] = (
        contribution_df["Contribution"].round(3)
    )

    # -----------------------------------------------------
    # HIGH RISK CONTRIBUTIONS
    # -----------------------------------------------------

    high_risk = contribution_df[
        contribution_df["Contribution"] > 0
    ].sort_values(
        "Contribution",
        ascending=False
    )

    st.write("### 🔴 Pushing Toward HIGH RISK")

    if not high_risk.empty:

        st.dataframe(
            high_risk,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No features are pushing toward HIGH RISK.")

    # -----------------------------------------------------
    # LOW RISK CONTRIBUTIONS
    # -----------------------------------------------------

    low_risk = contribution_df[
        contribution_df["Contribution"] < 0
    ].sort_values(
        "Contribution"
    )

    st.write("### 🟢 Pushing Toward LOW RISK")

    if not low_risk.empty:

        st.dataframe(
            low_risk,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No features are pushing toward LOW RISK.")

# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.caption(
    "⚠️ This application is an educational machine-learning "
    "project and should not be used as a medical diagnosis."
)