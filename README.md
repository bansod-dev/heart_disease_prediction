# ❤️ Heart Disease Prediction

A Machine Learning classification project that predicts the likelihood of heart disease based on clinical and health-related patient features.

🚀 **[Live Demo](http://heartdiseaseprediction-n4wh2dzlobofjaf2ghomuh.streamlit.app/)**

## 📌 Overview

This project implements an end-to-end Machine Learning pipeline including data preprocessing, categorical encoding, feature scaling, model training, evaluation, and Streamlit deployment.

## 🧠 Models Used

The following classification algorithms were evaluated:

| Model               |   Accuracy |   F1 Score |
| ------------------- | ---------: | ---------: |
| Logistic Regression | **86.41%** | **87.92%** |
| KNN                 |     84.78% |     86.41% |
| SVM                 |     85.33% |     87.08% |
| Naive Bayes         |     84.24% |     85.57% |
| Decision Tree       |     78.26% |     78.95% |

**Logistic Regression** was used for the final Streamlit application.

## 🔧 Features

The model uses features such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise Angina
* ST Depression
* ST Slope

Categorical features were encoded and numerical features were standardized using `StandardScaler`.

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy**
* **Matplotlib & Seaborn**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Git & GitHub**

## 🌐 Streamlit App

The trained model is integrated into a Streamlit web application where users can enter patient information and receive a prediction.

```text
Patient Information
        ↓
Data Preprocessing
        ↓
Trained ML Model
        ↓
Heart Disease Prediction
```

## 📁 Project Structure

heart_disease_prediction/
│
├── app.py
├── heart_prediction.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
└── .gitignore

## 📈 Key Learning Outcomes

* Data preprocessing & encoding
* Feature scaling
* Classification algorithms
* Model evaluation
* Model comparison
* ML model serialization with Joblib
* Streamlit deployment
* Git & GitHub

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used as a substitute for professional medical diagnosis or advice.

## 👨‍💻 Author

**Ghanshyam Bansod**

AI & Data Science Student | Aspiring Data Scientist

🔗 [GitHub](https://github.com/bansod-dev)
