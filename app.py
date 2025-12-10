import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. LOAD THE SAVED FILES
# Make sure these 4 files are in the same folder as this script
try:
    lin_reg = joblib.load('linear_reg.pkl')
    svr_model = joblib.load('svr_model.pkl')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
except FileNotFoundError:
    st.error("Files not found! Please make sure you downloaded 'linear_reg.pkl', 'svr_model.pkl', 'scaler_x.pkl', and 'scaler_y.pkl' and put them in this folder.")
    st.stop()

# 2. APP INTERFACE
st.title("Medical Cost Prediction AI")
st.write("Enter patient details to estimate insurance charges.")

# Create the Input Form
with st.form("user_inputs"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    with col3:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)

    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        sex = st.radio("Sex", ["Male", "Female"])
    with col_cat2:
        smoker = st.radio("Smoker", ["Yes", "No"])
        
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    
    model_choice = st.selectbox("Choose Model", ["Linear Regression", "SVR (Support Vector Regression)"])
    
    submit_button = st.form_submit_button("Predict Charges")

# 3. LOGIC WHEN BUTTON IS CLICKED
if submit_button:
    # --- A. Encode Categorical Data (Manual One-Hot Encoding) ---
    # This matches exactly how 'pd.get_dummies(drop_first=True)' works
    
    # Sex: Male=1, Female=0
    sex_male = 1 if sex == "Male" else 0
    
    # Smoker: Yes=1, No=0
    smoker_yes = 1 if smoker == "Yes" else 0
    
    # Region: Northeast is the "baseline" (0,0,0)
    region_northwest = 1 if region == "Northwest" else 0
    region_southeast = 1 if region == "Southeast" else 0
    region_southwest = 1 if region == "Southwest" else 0

    # --- B. Create DataFrame ---
    # The columns must be in the EXACT same order as X_train during training
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [sex_male],
        'smoker_yes': [smoker_yes],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest]
    })

    # --- C. Scale the Numerical Inputs ---
    # We only scale 'age', 'bmi', 'children' because that's what scaler_x knows
    cols_to_scale = ['age', 'bmi', 'children']
    input_data[cols_to_scale] = scaler_x.transform(input_data[cols_to_scale])

    # --- D. Predict ---
    if model_choice == "Linear Regression":
        # The prediction comes out as a "scaled score" (e.g., 0.24)
        scaled_prediction = lin_reg.predict(input_data)
    else:
        scaled_prediction = svr_model.predict(input_data)

    # --- E. Inverse Scale the Result ---
    # Convert the "scaled score" back into Dollars using scaler_y
    # We use .reshape(-1, 1) because the scaler expects a 2D array
    prediction_dollars = scaler_y.inverse_transform(scaled_prediction.reshape(-1, 1))

    # --- F. Show Result ---
    final_amount = prediction_dollars[0][0]
    st.success(f"Estimated Cost: ${final_amount:,.2f}")