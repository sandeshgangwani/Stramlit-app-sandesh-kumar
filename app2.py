import streamlit as st
import pandas as pd
from joblib import load
import dill

# Load model pipeline
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

# Load schema
my_feature_dict = load('my_feature_dict.pkl')

# Streamlit UI
st.title('Employee Churn Prediction App')
st.subheader('Based on Employee Dataset')
st.subheader('Created by Sandesh kumar')  

# Get categorical and numerical schema
categorical_input = my_feature_dict.get('CATEGORICAL')
numerical_input = my_feature_dict.get('NUMERICAL')

# Collect categorical inputs
st.subheader('Categorical Features')
categorical_input_vals = {}
for i, col in enumerate(categorical_input.get('Column Name').values()):
    options = categorical_input.get('Members')[i]
    categorical_input_vals[col] = st.selectbox(f"{col}", options, key=col)

# Collect numerical inputs
st.subheader('Numerical Features')
numerical_input_vals = {}
for col in numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.number_input(f"{col}", key=col)

# Combine inputs into a DataFrame
input_data = {**categorical_input_vals, **numerical_input_vals}
input_df = pd.DataFrame([input_data])

# Prediction function
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

# Prediction on button click
if st.button('Predict'):
    prediction = predict_churn(input_df)[0]
    translation_dict = {"Yes": "Expected", "No": "Not Expected"}
    prediction_translate = translation_dict.get(prediction, prediction)
    st.success(f'Prediction: **{prediction}** - Employee is **{prediction_translate}** to churn.')
