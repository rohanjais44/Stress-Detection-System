# Stress Detection System (AI + NLP)

## 📌 Overview

This project is an AI-based Stress Detection System that analyzes user text and predicts whether a person is experiencing stress using Natural Language Processing (NLP) and Machine Learning techniques. It transforms textual input into numerical features using TF-IDF vectorization and applies classification models to detect stress patterns.

A key highlight of this project is the integration of **Explainable AI using SHAP**, which provides transparency by identifying the specific words contributing to the prediction. This makes the system more interpretable and reliable.

---

## 🚀 Features

* Detects stress from user text input
* NLP preprocessing with TF-IDF vectorization
* Machine Learning models: Logistic Regression & SVM
* Model comparison using performance metrics
* SHAP-based Explainable AI for prediction interpretation
* Visualizations: confusion matrices & model comparison
* Simple UI for real-time user interaction

---

## 🛠 Tech Stack

* Python
* Scikit-learn
* NLP (TF-IDF)
* Pandas, NumPy
* SHAP (Explainable AI)
* Flask (Backend)
* HTML, CSS, JavaScript (Frontend)

---

## 📊 Models Used

* Logistic Regression
* Support Vector Machine (SVM)

---

## 📁 Project Structure

```
Stress-Detection-System/
│── app.py
│── stress_detection.py
│── model.pkl
│── vectorizer.pkl
│── stress-ui/
│── utils/
│── reddit_stress.csv
│── cm_lr.png
│── cm_svm.png
│── model_comparison.png
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/rohanjais44/Stress-Detection-System.git
cd Stress-Detection-System
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
python app.py
```

4. Open in browser:

```
http://127.0.0.1:5000/
```

---

## 📷 Output

* Confusion matrices for model evaluation
* Model comparison graph
* Real-time stress prediction via UI
* SHAP explanations showing important words influencing predictions

---

## 🎯 Future Improvements

* Integrate Deep Learning models (LSTM / BERT)
* Improve dataset size and diversity
* Deploy as a web application (Render / AWS)
* Enhance UI/UX for better interaction

---

## 📌 Conclusion

This project demonstrates how NLP and Machine Learning can be applied to mental health analysis, with Explainable AI ensuring transparency and trust in predictions.
