from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
import shap
from nltk.sentiment import SentimentIntensityAnalyzer
import random

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
tfidf = joblib.load("vectorizer.pkl")

sia = SentimentIntensityAnalyzer()

sample_text = ["sample text"]
sample_vec = tfidf.transform(sample_text).toarray()
dummy_extra = np.zeros((1, 9))
sample_features = np.hstack([sample_vec, dummy_extra])

explainer = shap.Explainer(model, sample_features)

stress_keywords = [
    "stress", "anxiety", "pressure", "overwhelmed",
    "tired", "exhausted", "deadline", "panic"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s!?]", "", text)
    return text

def emotional_features(text):
    scores = sia.polarity_scores(text)
    words = text.split()
    keyword_count = sum(1 for w in words if w in stress_keywords)

    return [
        scores["neg"],
        scores["neu"],
        scores["pos"],
        scores["compound"],
        keyword_count
    ]

def structural_features(text):
    words = text.split()

    return [
        len(words),
        text.count("!"),
        text.count("?"),
        sum(1 for c in text if c.isupper())
    ]

# 🔥 IMPROVED SHAP EXPLANATION (ONLY CHANGE)
def generate_explanation(feature_names, values, top_indices):
    explanations = []
    important_words = []

    ignore_words = [
        "feeling", "good", "hi", "years", "thing", "things",
        "people", "place", "time", "really", "just", "and",
        "very", "quite", "much", "many", "some"
    ]

    strong_words = [
        "stress", "stressed", "anxiety", "panic",
        "sad", "depressed", "overwhelmed", "tired",
        "exhausted", "pressure", "low"
    ]

    for i in top_indices:
        word = feature_names[i]

        if (
            word not in ["neg", "neu", "pos", "compound",
                         "keyword_count", "length",
                         "exclam", "question", "uppercase"]
            and len(word) >= 3
            and word not in ignore_words
        ):
            important_words.append(word)

    # 🔥 PRIORITY: choose meaningful emotional word
    chosen_word = None
    for w in important_words:
        if w in strong_words:
            chosen_word = w
            break

    if not chosen_word and important_words:
        chosen_word = important_words[0]

    if chosen_word:
        explanations.append(
            f"Words like '{chosen_word}' indicate emotional stress"
        )
    else:
        explanations.append(
            "Certain words in your message indicate emotional stress"
        )

    explanations.append("Your message shows signs of negative emotions")

    return explanations[:3]

def get_suggestions(text):
    text = text.lower()
    suggestions = []

    if any(w in text for w in ["stress", "pressure", "deadline", "overwhelmed"]):
        suggestions.append(
            "You seem to be under pressure. Try breaking your tasks into smaller steps and focus on one thing at a time instead of handling everything at once."
        )

    if any(w in text for w in ["sad", "down", "low", "demotivated"]):
        suggestions.append(
            "It looks like you're feeling low. Talking to someone you trust or expressing your thoughts through writing can really help lighten your mood."
        )

    if any(w in text for w in ["tired", "exhausted", "sleep"]):
        suggestions.append(
            "You seem mentally or physically tired. Getting proper rest or even a short nap can help you recover and think more clearly."
        )

    if any(w in text for w in ["anxiety", "panic", "overthinking"]):
        suggestions.append(
            "You might be feeling anxious. Try deep breathing or grounding exercises to calm your mind and bring your focus back to the present."
        )

    if any(w in text for w in ["happy", "good", "great", "excited"]):
        suggestions.append(
            "It's great to see you feeling positive. Try to maintain this by continuing activities that make you happy and relaxed."
        )

    if not suggestions:
        suggestions.append(
            "Try taking a short mindful break and focus on something relaxing to refresh your thoughts."
        )

    return suggestions

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not data or "text" not in data or not data["text"].strip():
            return jsonify({"error": "Invalid input"}), 400

        raw_text = data["text"]
        text = clean_text(raw_text)

        lex = tfidf.transform([text]).toarray()
        emo = np.array([emotional_features(text)])
        stru = np.array([structural_features(raw_text)])

        features = np.hstack([lex, emo, stru])

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        text_lower = raw_text.lower()

        negative_words = [
            "sad", "down", "depressed", "demotivated",
            "hopeless", "low", "lost"
        ]

        stress_words = [
            "stress", "pressure", "overwhelmed",
            "panic", "anxiety"
        ]

        if any(w in text_lower for w in negative_words + stress_words):
            result = "Stress"
        elif proba > 0.6:
            result = "Stress"
        else:
            result = "No Stress"

        shap_values = explainer(features)

        feature_names = list(tfidf.get_feature_names_out()) + [
            "neg", "neu", "pos", "compound", "keyword_count",
            "length", "exclam", "question", "uppercase"
        ]

        values = shap_values.values[0]
        top_indices = np.argsort(np.abs(values))[-5:][::-1]

        explanations = generate_explanation(feature_names, values, top_indices)
        suggestions = get_suggestions(raw_text)

        return jsonify({
            "prediction": result,
            "confidence": float(proba),
            "suggestions": suggestions,
            "explanations": explanations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)