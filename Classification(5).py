import joblib
import pandas as pd

model = joblib.load('knn_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

new_reviews = ["I don't like this product. i hate it.","This object is so good."]

X_new = vectorizer.transform(new_reviews)

predicted_reliability = model.predict(X_new)

predictions = []
for reliability in predicted_reliability:
    if reliability == 0:
        predictions.append("Unreliable")
    else:
        predictions.append("Reliable")

for review, reliability in zip(new_reviews, predictions):
    print(f"Review: {review}\nPredicted Reliability: {reliability}\n")
