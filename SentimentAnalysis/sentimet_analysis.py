import google.generativeai as genai
import pandas as pd
import numpy as np
import re
import nltk
import joblib
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
#from tensorflow.keras.callbacks import EarlyStopping

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

import os

# Download necessary NLTK resources
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
df = pd.read_csv("emotion_dataset.csv", names=["sentence", "emotion"])
print(df.head())
print(df['emotion'].value_counts()) 
df = df[df['emotion'].map(df['emotion'].value_counts()) > 2]  # Remove classes with <=2 samples

# -------- CONFIG --------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"API Key: {GOOGLE_API_KEY}")  # Debugging line

genai.configure(api_key=GOOGLE_API_KEY)
chat = genai.GenerativeModel("gemini-1.5-flash-002").start_chat()

# -------- EMOTION DETECTION --------
emotion_keywords = {
    "joy": ["happy", "excited", "great", "awesome", "joy", "glad", "delighted", "ecstatic", "love"],
    "sadness": ["sad", "down", "depressed", "unhappy", "miserable", "crying", "blue"],
    "anger": ["angry", "mad", "furious", "frustrated", "annoyed", "hate", "irritated"],
    "fear": ["afraid", "scared", "fear", "nervous", "worried", "anxious"],
}

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Convert text to numerical format using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Keeping top 5000 words
df['cleaned_sentence'] = df['sentence'].apply(preprocess_text)
X_tfidf = vectorizer.fit_transform(df['cleaned_sentence'])

# Encode labels
label_encoder = LabelEncoder()
df['emotion_encoded'] = label_encoder.fit_transform(df['emotion'])

# Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['emotion_encoded'])

# Convert back to DataFrame
df_balanced = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
df_balanced['emotion'] = label_encoder.inverse_transform(y_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Initialize and train the model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


# Save model, vectorizer, and label encoder
joblib.dump(nb_model, "emotion_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")


# Predictions
y_pred = nb_model.predict(X_test)

# Model evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))



def detect_emotion(text):
    text_tfidf = vectorizer.transform([text])  # Convert input text to TF-IDF
    prediction = nb_model.predict(text_tfidf)  # Predict class
    emotion = label_encoder.inverse_transform(prediction)[0]
    return emotion

# -------- SYSTEM PROMPT BASED ON EMOTION --------
def get_emotion_prompt(emotion):
    prompts = {
        "joy": "You are cheerful, upbeat, and playful. Celebrate the user's happiness and match their excitement.",
        "sadness": "You are gentle and compassionate. Be a good listener, validate their feelings, and offer comforting responses.",
        "anger": "You are calm and patient. Help the user process their anger without judgment, and gently guide them toward understanding and peace.",
        "fear": "You are reassuring and supportive. Ease the user's worries, validate their fears, and offer calming perspectives.",
        "neutral": "You are thoughtful and balanced. Keep the conversation open and engaging without taking a strong emotional stance."
    }
    return prompts[emotion]

# -------- CHAT HANDLER --------
def generate_response(user_input):
    emotion = detect_emotion(user_input)
    system_prompt = get_emotion_prompt(emotion)

    # Send mood-aligned system prompt once
    chat.send_message(system_prompt)

    # Send user input
    response = chat.send_message(user_input)

    return emotion, response.text

# # -------- MAIN LOOP --------
def main():
    pass
    print("🧠 Emotion-Aware Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Bot: Wishing you well! 👋")
            break

        emotion, reply = generate_response(user_input)
        print(f"[Detected Emotion: {emotion}]")
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()