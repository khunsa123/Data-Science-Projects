# =========================================
# TEXT CLASSIFICATION - PREMIUM VERSION
# Model Comparison + Visualization
# =========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree, metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# -----------------------------------------
# Load Dataset
# -----------------------------------------
print("Loading dataset...")
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# -----------------------------------------
# Model Training Function
# -----------------------------------------
def train_and_evaluate(model, model_name):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', model),
    ])
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(metrics.classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy, predictions, pipeline

# -----------------------------------------
# Train All Models
# -----------------------------------------
results = {}
predictions_store = {}
pipelines = {}

models = {
    "Decision Tree": tree.DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

for name, model in models.items():
    acc, preds, pipe = train_and_evaluate(model, name)
    results[name] = acc
    predictions_store[name] = preds
    pipelines[name] = pipe

# -----------------------------------------
# Results Table
# -----------------------------------------
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("="*60)
print(results_df)

# -----------------------------------------
# Visualization: Accuracy Comparison
# -----------------------------------------
plt.figure(figsize=(10,6))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

# -----------------------------------------
# Confusion Matrix (Best Model)
# -----------------------------------------
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest Model: {best_model_name}")

best_predictions = predictions_store[best_model_name]

ConfusionMatrixDisplay.from_predictions(y_test, best_predictions)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig("confusion_matrix.png")
plt.show()



# ============================================================
# Project Summary
# ============================================================
# This project demonstrates a complete text classification pipeline
# using machine learning techniques on the 20 Newsgroups dataset.
# The workflow includes text preprocessing using Count Vectorization
# and TF-IDF transformation, followed by the implementation and
# comparison of five different models: Decision Tree, K-Nearest
# Neighbors (KNN), Naive Bayes, Logistic Regression, and Support
# Vector Machine (SVM). The models were evaluated using accuracy,
# precision, recall, and F1-score metrics. Additionally, the results
# were visualized through an accuracy comparison chart and a confusion
# matrix for the best-performing model. The experiment highlights that
# SVM and Logistic Regression perform best for high-dimensional text
# data, while simpler models like Decision Tree and KNN are less
# effective in this context.
