from flask import Flask, request, jsonify, render_template
from sentiment_prompter import generate_response
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided."}), 400

    try:
        emotion, reply = generate_response(user_input)
        return jsonify({
            "emotion": emotion,
            "response": reply
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
