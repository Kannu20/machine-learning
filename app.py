# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load('Naive_Bayes_heart')
# scalar = joblib.load('scalar.pkl')
# expected_columns = joblib.load("columns.pkl")

# st.title("Heart stroke prediction by akarsh")

# st.markdown("Provide the following details")

# age = st.slider("Age",18,100,40)
# sex = st.selectbox("Sex",["Male","Female"])
# chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# # When Predict is clicked
# if st.button("Predict"):

#     # Create a raw input dictionary
#     raw_input = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#         'Sex_' + sex: 1,
#         'ChestPainType_' + chest_pain: 1,
#         'RestingECG_' + resting_ecg: 1,
#         'ExerciseAngina_' + exercise_angina: 1,
#         'ST_Slope_' + st_slope: 1
#     }

#     # Create input dataframe
#     input_df = pd.DataFrame([raw_input])

#     # Fill in missing columns with 0s
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder columns
#     input_df = input_df[expected_columns]

#     # Scale the input
#     scaled_input = scalar.transform(input_df)

#     # Make prediction
#     prediction = model.predict(scaled_input)[0]

#     # Show result
#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")

import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Load Model Artifacts ----------------
model = joblib.load('Naive_Bayes_heart')
scalar = joblib.load('scalar.pkl')
expected_columns = joblib.load("columns.pkl")

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        .main {
            background-color: #f9fafb;
        }
        .title-container {
            text-align: center;
            padding: 1.2rem 0 0.4rem 0;
        }
        .title-container h1 {
            color: #B91C1C;
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .title-container p {
            color: #6B7280;
            font-size: 1rem;
            margin-top: 0.2rem;
        }
        .section-header {
            color: #111827;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #B91C1C;
            padding-left: 0.6rem;
        }
        div.stButton > button {
            background-color: #B91C1C;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            border: none;
            width: 100%;
            transition: background-color 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #991B1B;
            color: white;
        }
        .result-box {
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "This tool estimates the risk of heart disease using a "
        "Naive Bayes model trained on clinical parameters."
    )
    st.markdown("---")
    st.markdown("**Model:** Naive Bayes")
    st.markdown("**Purpose:** Educational / Screening aid")
    st.markdown("---")
    st.caption("⚠️ Not a substitute for professional medical advice.")

# ---------------- Title ----------------
st.markdown("""
    <div class="title-container">
        <h1>❤️ Heart Stroke Prediction</h1>
        <p>by Akarsh — Enter patient details below to assess risk</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Input Sections ----------------
st.markdown('<div class="section-header">🧍 Personal Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40)
with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])

st.markdown('<div class="section-header">🫀 Cardiac Parameters</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
with col4:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

st.markdown('<div class="section-header">🏃 Exercise & ECG Response</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
with col6:
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st.markdown("---")

# ---------------- Predict Button ----------------
predict_clicked = st.button("🔍 Predict Risk")

if predict_clicked:

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scalar.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    with st.spinner("Analyzing..."):
        if prediction == 1:
            st.markdown(
                '<div class="result-box" style="background-color:#FEE2E2; color:#B91C1C;">'
                '⚠️ High Risk of Heart Disease</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box" style="background-color:#DCFCE7; color:#15803D;">'
                '✅ Low Risk of Heart Disease</div>',
                unsafe_allow_html=True
            )