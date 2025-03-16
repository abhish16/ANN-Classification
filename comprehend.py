import boto3
import os
from sns_notification import send_sns_notification
os.environ["ACCESS"] 
os.environ["KEY"] 

# AWS Comprehend Client
comprehend = boto3.client("comprehend", region_name="us-east-1",aws_access_key_id= os.environ.get("ACCESS"),aws_secret_access_key=os.environ.get("KEY"))

def detect_language(text):
    """Detects the dominant language in the given text."""
    response = comprehend.detect_dominant_language(Text=text)
    detected_languages = response["Languages"]
    
    if detected_languages:
        # Get the language with the highest confidence
        detected_language = max(detected_languages, key=lambda x: x["Score"])
        return detected_language["LanguageCode"], detected_language["Score"]
    return None, None

def analyze_sentiment(text, language_code):
    """Analyzes sentiment of the given text in the specified language."""
    supported_languages = ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "hi", "ar"]  # Comprehend-supported languages for sentiment analysis
    
    if language_code in supported_languages:
        response = comprehend.detect_sentiment(Text=text, LanguageCode=language_code)
        Sentiment = response["Sentiment"]
        # If sentiment is Negative, send an SNS notification
        if Sentiment == "NEGATIVE":
            send_sns_notification()
        return {
            "Sentiment": response["Sentiment"],
            "ConfidenceScores": response["SentimentScore"]
        }
    return None
