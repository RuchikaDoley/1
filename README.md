
# 🧠 Emotion-Aware Chatbot with Gemini API

A sentiment-aware chatbot that uses natural language processing and Google Gemini to detect user emotions and respond in a tone aligned with the detected sentiment. This project includes emotion detection using a trained Naive Bayes classifier and dynamically prompts Gemini with a tone-matching system message for more emotionally intelligent responses.

---

## 🚀 Features

- Detects user emotions (`joy`, `sadness`, `anger`, `fear`)
- Tailors chatbot responses based on detected emotion
- Uses Google Gemini API for emotionally intelligent replies
- Handles imbalanced datasets using SMOTE
- Web interface powered by Flask

---

## 📁 Project Structure

```
.
├── sentiment_analysis.py       # Model training and emotion detection logic
├── sentiment_prompter.py       # Handles real-time prediction and prompt generation
├── server.py                   # Flask backend server
├── emotion_dataset.csv         # Raw dataset for training
├── templates/
│   └── index.html              # Frontend UI for the chatbot
├── .env                        # Contains GOOGLE_API_KEY
├── requirements.txt            # Python dependencies
└── README.md                   # You're here!
```

---
## 💡 How It Works

1. **Preprocessing**: Input is lowercased, punctuation and stopwords are removed.
2. **Vectorization**: Text is transformed into TF-IDF vectors.
3. **Prediction**: A Naive Bayes model predicts emotion.
4. **Prompt Tuning**: A system prompt based on emotion is sent to Gemini.
5. **Response Generation**: Gemini responds in an emotionally appropriate tone.

---

## 📦 Requirements

```
Flask
google-generativeai
nltk
pandas
python-dotenv
gunicorn
numpy
scikit-learn
```
