# Amazon Product Reviews — Sentiment Analysis

A production-ready **Machine Learning web application** that predicts the sentiment of Amazon product reviews using an optimized **Support Vector Machine (SVM)** pipeline with advanced feature engineering.

---

## 🌐 Live Demo

[![Open App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit)](https://amazon-reviews-sentiment-analysis-ml.streamlit.app/)

---

## 📌 Overview

This project builds an intelligent sentiment analysis system that classifies product reviews as **Positive** or **Negative**. It combines traditional NLP techniques with lexicon-based sentiment features to improve prediction accuracy.

The system is deployed as an interactive **Streamlit web app**, allowing users to input reviews and instantly get predictions.

---

## 🧠 Key Features

* 🔍 Real-time sentiment prediction
* ⚡ Fast and optimized **LinearSVC model**
* 🧪 Predefined sample reviews for testing
* 📊 Model performance visualization
* 🎯 High accuracy with balanced classification
* 🖥️ Interactive and user-friendly UI

---

## 🏗️ Project Architecture

```text
Text Input
   ↓
Preprocessing (cleaning, normalization)
   ↓
TF-IDF Vectorization (Trigrams)
   ↓
VADER Sentiment Features
   ↓
Feature Combination
   ↓
LinearSVC (Calibrated)
   ↓
Prediction (Positive / Negative)
```

---

## ⚙️ Tech Stack

### 🧪 Machine Learning

* Scikit-learn
* Imbalanced-learn (SMOTE)
* NLTK (VADER Sentiment Analysis)

### 🖥️ Frontend

* Streamlit

### 📊 Data Handling

* Pandas
* NumPy

---

## 🔬 Model Details

* **Model**: Linear Support Vector Machine (LinearSVC)
* **Optimization**: GridSearchCV
* **Class Imbalance Handling**: SMOTE
* **Calibration**: CalibratedClassifierCV
* **Features Used**:

  * TF-IDF (Trigrams)
  * VADER sentiment scores
  * Phrase detection

---

## 📈 Performance

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~88%  |
| Macro F1 | ~88%  |
| ROC-AUC  | ~0.90 |

---

## 📂 Project Structure

```text
Amazon_Reviews_Sentiment_Analysis/
│
├── app.py
├── Models/
│   ├── model_svm.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   └── metrics.json
│
├── assets/
│   └── amazon.svg
│
├── requirements.txt
├── README.md
```

---

# 📊 Dataset

This project uses an **Amazon product reviews dataset** for training and evaluation.

⚠️ **Dataset is NOT included in this repository** due to GitHub file size limitations.

---

## 🔗 Download Dataset

You can download a similar dataset from:

* https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews

---

## 📥 Steps to Use Dataset

1. Download dataset from Kaggle
2. Extract the files
3. Place them inside your project folder:

```text
Amazon_Reviews_Sentiment_Analysis/
│
|── train.csv
│── test.csv
|
```

---

## ⚙️ Preprocessing (Important)

The dataset is:

* Cleaned (removal of stopwords, punctuation)
* Balanced using SMOTE
* Converted to TF-IDF features

👉 After preprocessing, trained models are saved in:

```text
Models/
```

---

## 💡 Note

Once models are trained:

👉 You **DO NOT need dataset for running the app**

The Streamlit app directly uses:

* `model_svm.pkl`
* `tfidf_vectorizer.pkl`
* `scaler.pkl`

---

# ▶️ How to Run Locally

---

## 1️⃣ Clone the repository

```bash
git clone https://github.com/riku-d/Amazon_Reviews_Sentiment_Analysis.git
cd Amazon_Reviews_Sentiment_Analysis
```

---

## 2️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the app

```bash
streamlit run app.py
```

---

# 🧪 Sample Inputs

| Review                         | Expected Sentiment |
| ------------------------------ | ------------------ |
| "Waste of money"               | Negative           |
| "Stopped working after 2 days" | Negative           |
| "Absolutely love this product" | Positive           |
| "Best purchase ever"           | Positive           |

---

# 🧠 Key Concepts Used

* Support Vector Machines (SVM)
* TF-IDF Vectorization
* Sentiment Lexicons (VADER)
* Class Imbalance Handling (SMOTE)
* Model Calibration
* Feature Engineering

---

# 💡 Why LinearSVC?

* Efficient for high-dimensional text data
* Faster than kernel-based SVM
* No need for `gamma` parameter
* Works best with TF-IDF

---

# 🔍 Why VADER?

VADER helps capture:

* Negation handling ("not good")
* Emotional intensity
* Contextual sentiment

👉 It improves SVM performance by adding semantic features.

---

# 🚀 Future Improvements

* Deep learning models (LSTM, BERT)
* Multi-class sentiment classification
* Explainable AI (SHAP, LIME)
* REST API deployment
