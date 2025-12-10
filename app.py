import streamlit as st
import joblib
import pandas as pd

# 1. Load the models
# Ensure these files are in the same directory as app.py
try:
    lin_reg = joblib.load('linear_reg_model.pkl')
    svr_model = joblib.load('svr_model.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'linear_reg_model.pkl' and 'svr_model.pkl' are in the same folder.")
    st.stop()

# 2. App Title and Description
st.title("Medical Cost Prediction App")
st.write("Enter the patient's details below to predict insurance charges.")

# 3. Input Form
with st.form("prediction_form"):
    # Numerical Features
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    with col3:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)

    # Categorical Features
    col4, col5 = st.columns(2)
    with col4:
        sex = st.radio("Sex", ["Male", "Female"])
    with col5:
        smoker = st.radio("Smoker", ["Yes", "No"])
    
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    
    # Model Selection
    model_choice = st.selectbox("Choose Model", ["Linear Regression", "SVR (Support Vector Regression)"])

    submit = st.form_submit_button("Predict")

if submit:
    # 4. Preprocess Inputs to match Model Training (One-Hot Encoding)
    
    # Sex: The model expects 'sex_male' (1 for Male, 0 for Female)
    sex_male = 1 if sex == "Male" else 0
    
    # Smoker: The model expects 'smoker_yes' (1 for Yes, 0 for No)
    smoker_yes = 1 if smoker == "Yes" else 0
    
    # Region: The model expects 3 columns. 'Northeast' is the reference (all 0s).
    region_northwest = 1 if region == "Northwest" else 0
    region_southeast = 1 if region == "Southeast" else 0
    region_southwest = 1 if region == "Southwest" else 0

    # Create the DataFrame with the exact column names the model expects
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

    # 5. Prediction
    if model_choice == "Linear Regression":
        prediction = lin_reg.predict(input_data)
    else:
        prediction = svr_model.predict(input_data)

    # 6. Output
    st.success(f"Estimated Medical Cost: ${prediction[0]:,.2f}")
    
    # Optional: Show the input data used
    with st.expander("See input data sent to model"):
        st.dataframe(input_data)