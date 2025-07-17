import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer using joblib
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Streamlit UI
st.set_page_config(page_title="Udaan – Job Recommender", layout="centered")
st.title("🧠 Udaan – Job Recommender")
st.write("Enter your skills, interests, or qualifications and get job recommendations!")

# User input
user_input = st.text_area("💬 Describe your background:", "")

if st.button("🔍 Get Recommendation"):
    if user_input.strip() == "":
        st.warning("Please enter some skills or interests to get a recommendation.")
    else:
        # Transform input
        transformed = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(transformed)[0]

        # Output
        if prediction == 1:
            st.success("🎯 A suitable job **is recommended** based on your profile!")
        else:
            st.error("❌ No job match is currently recommended based on your input.")
