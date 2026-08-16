# ❤️ Heart Disease Prediction

A Machine Learning project that predicts whether a person is likely to have heart disease based on medical and demographic features.

## 🚀 Project Overview

This project uses Machine Learning to classify patients into two categories:

- 0 → No heart disease
- 1 → Heart disease

The trained model is saved using `joblib` and integrated into a Streamlit web application for making predictions.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 🤖 Machine Learning

The project includes:

- Data preprocessing
- Feature scaling
- Model training
- Model evaluation
- Prediction using the trained model

The trained model, scaler, and expected feature columns are saved as `.pkl` files.

## 📁 Project Structure

```text
heart_disease_prediction/
│
├── app.py
├── heart_prediction.pkl
├── scalar.pkl
├── columns.pkl
├── requirements.txt
└── README.md
