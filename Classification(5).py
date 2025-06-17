import joblib
import pandas as pd

model = joblib.load('random_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

new_reviews = ["This object is so good, the quality is top notch","sjdfbnjkewfbgbgn","very poor quality , amazon needs to change its vendors, these are all fake"]

X_new = vectorizer.transform(new_reviews)

predicted_reliability = model.predict(X_new)

predictions = []
for reliability in predicted_reliability:
    if reliability == 1:
        predictions.append("Reliable")
    else:
        predictions.append("Unreliable")

for review, reliability in zip(new_reviews, predictions):
    print(f"Review: {review}\nPredicted Reliability: {reliability}\n")
