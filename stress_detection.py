import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import nltk
nltk.download('vader_lexicon')   # auto download (safe)
from nltk.sentiment import SentimentIntensityAnalyzer

# 1. LOAD DATA
data = pd.read_csv("reddit_stress.csv")

data = data[["text", "label"]]

texts = data["text"].astype(str)
labels = data["label"]


# 2. PREPROCESSING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s!?]", "", text)
    return text


texts = texts.apply(clean_text)

# 3. LEXICAL FEATURES (TF-IDF)
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_lexical = tfidf.fit_transform(texts).toarray()


# 4. EMOTIONAL FEATURES
sia = SentimentIntensityAnalyzer()

stress_keywords = [
    "stress", "anxiety", "pressure", "overwhelmed",
    "tired", "exhausted", "deadline", "panic"
]

def emotional_features(text):
    scores = sia.polarity_scores(text)
    keyword_count = sum(text.count(w) for w in stress_keywords)

    return [
        scores["neg"],
        scores["neu"],
        scores["pos"],
        scores["compound"],
        keyword_count
    ]


X_emotion = np.array([emotional_features(t) for t in texts])


# 5. STRUCTURAL FEATURES
def structural_features(text):
    words = text.split()

    sentence_length = len(words)
    exclam = text.count("!")
    question = text.count("?")
    upper = sum(1 for c in text if c.isupper())

    return [sentence_length, exclam, question, upper]


X_struct = np.array([structural_features(t) for t in texts])


# 6. FEATURE FUSION
X = np.hstack([X_lexical, X_emotion, X_struct])


# 7. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)


# 8. MODELS
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

svm = LinearSVC()
svm.fit(X_train, y_train)


# 9. EVALUATION
def evaluate(model, name):
    preds = model.predict(X_test)
    print("\n", "=" * 20, name, "=" * 20)
    print("F1 Score:", f1_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))


evaluate(lr, "Logistic Regression")
evaluate(svm, "SVM")


print("\n✅ Training Complete!")

import joblib

joblib.dump(lr, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("✅ Model Saved!")


print("\nTest your own sentence!")
user_input = input("Enter text: ")

t = clean_text(user_input)

lex = tfidf.transform([t]).toarray()
emo = np.array([emotional_features(t)])
stru = np.array([structural_features(t)])

features = np.hstack([lex, emo, stru])

pred = lr.predict(features)[0]

if pred == 1:
    print("Prediction: STRESS")
else:
    print("Prediction: NO STRESS")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# GRAPH 1 – MODEL COMPARISON (F1)
lr_preds = lr.predict(X_test)
svm_preds = svm.predict(X_test)

lr_f1 = f1_score(y_test, lr_preds)
svm_f1 = f1_score(y_test, svm_preds)

models = ["Logistic Regression", "SVM"]
scores = [lr_f1, svm_f1]

plt.figure()
plt.bar(models, scores)
plt.title("Model Comparison (F1 Score)")
plt.ylabel("F1 Score")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()


# GRAPH 2 – CONFUSION MATRIX (LR)
plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, lr_preds)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("cm_lr.png")
plt.close()


# GRAPH 3 – CONFUSION MATRIX (SVM)
plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, svm_preds)
plt.title("Confusion Matrix - SVM")
plt.tight_layout()
plt.savefig("cm_svm.png")
plt.close()

print("✅ Graphs saved!")