import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction App 🚢")
st.subheader("Logistic Regression Model")

st.write("Enter the following details to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
sex_encoded = 1 if sex == "male" else 0
age = st.slider("Age", 0, 100, 25)

# Prepare input data for prediction
input_data = np.array([[pclass, sex_encoded, age]])

# Predict when button is clicked
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    output = "🎉 Survived" if prediction[0] == 1 else "❌ Did not survive"
    st.success(f"Prediction Result: {output}")