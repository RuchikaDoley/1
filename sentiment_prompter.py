import joblib
import re
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the saved model components
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Detect emotion
def detect_emotion(text):
    cleaned = preprocess_text(text)
    tfidf = vectorizer.transform([cleaned])
    pred = model.predict(tfidf)
    return label_encoder.inverse_transform(pred)[0]

# System prompt generator
def get_emotion_prompt(emotion):
    prompts = {
        "joy": "You are cheerful, upbeat, and playful.",
        "sadness": "You are gentle and compassionate.",
        "anger": "You are calm and patient.",
        "fear": "You are reassuring and supportive.",
        "neutral": "You are thoughtful and balanced."
    }
    return prompts.get(emotion, "You are helpful and friendly.")

# Final response function
def generate_response(user_input):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    genai.configure(api_key=GOOGLE_API_KEY)
    chat = genai.GenerativeModel("gemini-1.5-flash-002").start_chat()

    emotion = detect_emotion(user_input)
    system_prompt = get_emotion_prompt(emotion)
    chat.send_message(system_prompt)
    response = chat.send_message(user_input)
    return emotion, response.text
