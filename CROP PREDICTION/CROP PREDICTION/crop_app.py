import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 📌 Load the trained model and label encoder
model_filename = "C:/Users/ACTW-7/Downloads/dataset/dataset/crop_predict.pkl"
encoder_filename ="C:/Users/ACTW-7/Downloads/dataset/dataset/crop_encoded.pkl"


with open(model_filename, "rb") as file:
    model = pickle.load(file)
with open(encoder_filename, "rb") as file:
    label_encoder = pickle.load(file)



# 📌 Streamlit UI
st.title(" CROP Prediction App")
st.write("Enter  crop data  to predict its label.")

# 📌 User input fields
temperature = st.number_input("temperature 	", min_value=0.0, format="%.2f")
humidity 	= st.number_input("humidity", min_value=0.0, format="%.2f")
ph = st.number_input("ph", min_value=0.0, format="%.2f")
rainfall = st.number_input("rainfall", min_value=0.0, format="%.2f")

# 📌 Predict button
if st.button("Predict"):
    # Convert input into NumPy array
    input_data = np.array([[temperature,humidity,ph,rainfall]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Decode predicted label
    predicted_label=label_encoder.inverse_transform(prediction)[0]
    
    # Display the result
    st.success(f"Predicted crop: **{predicted_label}**")
