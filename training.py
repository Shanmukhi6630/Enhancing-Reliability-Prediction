import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("/Users/shanmukhimudundi/Desktop/MLOps/cleaned_for_modeling.csv")

# Define columns
text_column = 'Review Text'
label_column = 'Reliability'

# Label encoding
label_mapping = {'UNRELIABLE': 0, 'RELIABLE': 1}
df = df[df[label_column].isin(label_mapping)]
y = df[label_column].map(label_mapping)
X_text = df[text_column]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Classifier": SVC(),
    "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Naive Bayes": MultinomialNB(),
    "KNN Classifier": KNeighborsClassifier()
}

# Evaluate models
results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
        "Recall": round(recall_score(y_test, y_pred, average='weighted'), 4),
        "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 4),
        "Train Time (s)": round(end - start, 3)
    })

# Create DataFrame
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

# Show table
print("\n📊 Classification Model Evaluation Summary:")
print(results_df.to_string(index=False))

# Plot all metrics including Train Time
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
models_list = results_df['Model'].tolist()

x = np.arange(len(models_list))
width = 0.15

plt.figure(figsize=(14, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, results_df[metric], width=width, label=metric)

plt.xticks(x + width * 2, models_list, rotation=30, ha='right')
plt.ylabel("Score / Time")
plt.title("Model Performance Comparison ")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['Train Time (s)'], color='skyblue')
plt.title('Model Training Time Comparison')
plt.xlabel('Model')
plt.ylabel('Train Time (seconds)')
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

'''knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

joblib.dump(knn_model, 'knn_model.pkl')
print("KNN model saved successfully")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("TF-IDF saved successfully")'''