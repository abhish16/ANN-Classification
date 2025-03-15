import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import boto3

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#sentiment Analysis

def analyze_sentiment(user_feedback, language):
    client = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id='AKIA3CMCCRAS6WCZ7ESH',aws_secret_access_key='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')

    try:
        response = client.detect_sentiment(Text=user_feedback, LanguageCode=language)
        user_sentiment = response['Sentiment']
        return user_sentiment
    except Exception as e:
        return f"Error: {str(e)}"

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

#Sentiment
#analyze_sentiment(user_feedback,language)
import streamlit as st
from comprehend import detect_language, analyze_sentiment

#st.title("AWS Comprehend Sentiment Analysis with Auto Language Detection")

# Text input
user_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_text:
        # Detect language
        language_code, confidence = detect_language(user_text)
        
        if language_code:
            st.subheader("Detected Language:")
            st.write(f"**Language:** {language_code.upper()} (Confidence: {confidence:.2f})")

            # Perform sentiment analysis
            sentiment_result = analyze_sentiment(user_text, language_code)
            
            if sentiment_result:
                st.subheader("Sentiment Analysis Result:")
                st.write(f"**Sentiment:** {sentiment_result['Sentiment']}")

                st.subheader("Confidence Scores:")
                st.json(sentiment_result["ConfidenceScores"])
            else:
                st.warning(f"Sentiment analysis is not supported for detected language: {language_code.upper()}")
        else:
            st.error("Could not detect a language. Please enter more text.")
    else:
        st.warning("Please enter some text for analysis.")


st.write(f'Churn Probability: {prediction_proba:.2f}')
#st.write(f'User sentiment is : {user_sentiment}' )

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')





