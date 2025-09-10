import streamlit as st
import pickle
import joblib
from train import fake_news_det 

try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    vectorization = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop()

# Set the title and description of the app
st.title("Fake News Classifier")
st.write("Enter a news article in the text box below and click 'Predict' to see if it's classified as real or fake news.")

# Create a text area for user input
user_input = st.text_area("News Article Text", height=250)

# Create a button to trigger the prediction
if st.button("Predict"):
    if user_input:
        # Call your model's prediction function
        prediction = fake_news_det(user_input, model, vectorization)

        # Display the prediction result
        st.subheader("Prediction:")
        st.write(f"The news article is classified as **{prediction}**.")
    else:
        st.warning("Please enter some text to get a prediction.")
