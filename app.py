import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("salary_model.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Salary Prediction App (>50K)")

# --- Input fields ---
age = st.slider("Age", 18, 90, 30)
education_num = st.slider("Education Number", 1, 16, 9)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)

# Simulate encoded values (you must match what your LabelEncoder encoded)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov'])
marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Sales'])
relationship = st.selectbox("Relationship", ['Not-in-family', 'Husband', 'Wife'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander'])
gender = st.selectbox("Gender", ['Male', 'Female'])

# --- Encoding categorical values manually ---
def encode(val, options):
    return options.index(val)

# Options must match your LabelEncoder training order
workclass_opt = ['Private', 'Self-emp-not-inc', 'Local-gov']
marital_opt = ['Never-married', 'Married-civ-spouse', 'Divorced']
occupation_opt = ['Tech-support', 'Craft-repair', 'Sales']
relationship_opt = ['Not-in-family', 'Husband', 'Wife']
race_opt = ['White', 'Black', 'Asian-Pac-Islander']
gender_opt = ['Male', 'Female']

# Prepare input for model
input_data = np.array([[
    age,
    encode(workclass, workclass_opt),
    education_num,
    encode(marital_status, marital_opt),
    encode(occupation, occupation_opt),
    encode(relationship, relationship_opt),
    encode(race, race_opt),
    encode(gender, gender_opt),
    capital_gain,
    capital_loss,
    hours_per_week
]])

# --- Prediction ---
if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Predicted Salary: > $50K")
    else:
        st.warning("🔍 Predicted Salary: ≤ $50K")
