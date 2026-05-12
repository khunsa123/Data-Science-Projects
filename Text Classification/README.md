# 📰 Text Classification — Model Comparison

## 📌 Overview

A complete **NLP text classification pipeline** built on the 20 Newsgroups dataset, comparing five machine learning models using a TF-IDF feature extraction approach. The project evaluates model performance with detailed metrics and visualises results through an accuracy comparison chart and confusion matrix for the best-performing model.

---

## 🎯 Objectives

- Build a reusable text classification pipeline with CountVectorizer and TF-IDF
- Train and compare five ML classifiers on a 20-class newsgroup dataset
- Identify the best-performing model and visualise its confusion matrix
- Produce a ranked model comparison table for reproducible benchmarking

---

## 🧠 Methods & Techniques

- Text preprocessing via **CountVectorizer** (bag-of-words) and **TF-IDF transformation**
- Scikit-learn `Pipeline` for clean, reproducible model training
- Multi-class classification across 20 newsgroup topic categories
- Accuracy, precision, recall, and F1-score evaluation per model

### 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score (per class)
- Confusion Matrix (best model)

---

## 🤖 Models Compared

| Model | Notes |
|-------|-------|
| Decision Tree | Baseline tree-based classifier |
| K-Nearest Neighbors (KNN) | Instance-based, distance metric classification |
| Naive Bayes (Multinomial) | Probabilistic text classification baseline |
| Logistic Regression | Linear model with L2 regularisation |
| Support Vector Machine (SVM) | LinearSVC — strong performer on high-dimensional text |

---

## 📊 Dataset

- **20 Newsgroups Dataset** — loaded directly via `sklearn.datasets.fetch_20newsgroups`
- 20 topic categories (politics, religion, sports, technology, etc.)
- No manual download required

---

## 📈 Key Results

- **SVM and Logistic Regression** achieved the highest accuracy on high-dimensional TF-IDF features
- **Decision Tree and KNN** underperformed relative to linear models on sparse text representations
- Results visualised as a ranked accuracy bar chart and best-model confusion matrix

---

## 📂 Outputs

| File | Description |
|------|-------------|
| `model_accuracy_comparison.png` | Bar chart comparing accuracy across all 5 models |
| `confusion_matrix.png` | Confusion matrix for the best-performing model |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, Pandas, Matplotlib

---

## ⚙️ Usage

```bash
pip install scikit-learn pandas matplotlib
python "Model Comparison V1.py"
```

---

## 📬 Contact

**Khunsa Iftikhar**
📧 [khunsaiftikhar123@gmail.com](mailto:khunsaiftikhar123@gmail.com)
🔗 [linkedin.com/in/khunsa-iftikhar](https://www.linkedin.com/in/khunsa-iftikhar/)
