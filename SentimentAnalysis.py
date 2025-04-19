import time
import pandas as pd
import spacy
import requests

TOGETHER_API_KEY = "0f204cdb020f5899698bbb8c8d1b12a74dc9ee54103d0272440e182e07037ea3"
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

nlp = spacy.load("en_core_web_sm")

def sentence_splitter(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def count_non_verbs(text):
    if pd.isnull(text) or not text.strip():
        return 0
    doc = nlp(text)
    excluded_tags = {"VERB", "AUX", "DET", "PRON", "CCONJ", "SCONJ", "PART"}
    imp_words = [token.text for token in doc if token.pos_ not in excluded_tags and token.is_alpha]
    return len(imp_words)

def clean_data(csv_file_path, na_values=None):
    if na_values is None:
        na_values = ["None", "none", "NA", "N/A", "n/a", "null", "NULL", "-", ""]

    print(f"\n📥 Reading file: {csv_file_path}")
    df = pd.read_csv(csv_file_path, na_values=na_values)

    if "Verified Purchase" in df.columns:
        df["Verified Purchase"] = df["Verified Purchase"].apply(
            lambda x: "Yes" if str(x).strip().lower() == "yes" else "No"
        )
        print("🔹 Cleaned 'Verified Purchase' column to Yes/No")

    df.dropna(inplace=True)
    print(f"🔹 Dropped missing values. Shape: {df.shape}")

    if "Review Text" in df.columns:
        original_len = len(df)
        df = df[df["Review Text"].str.len() > 10]
        print(f"🔹 Filtered short reviews (<10 chars): {original_len - len(df)} removed")

    df.drop_duplicates(inplace=True)
    print(f"🔹 Dropped duplicates. Shape: {df.shape}")
    df["Imp_Words"] = df["Review Text"].apply(count_non_verbs)

    return df

def query_together_api(prompt, max_tokens=10):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "top_p": 1.0
    }

    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip().upper()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return "ERROR"

def analyze_review_sentiment(review_text, delay=1.2):
    sentences = sentence_splitter(review_text)
    pos_count = 0
    neg_count = 0
    neutral_count = 0
    score_total = 0
    scored_sentences = 0

    for sent in sentences:
        if not sent.strip():
            continue
        prompt = f"""Classify the sentiment of the following sentence. Respond with POSITIVE, NEGATIVE, or NEUTRAL, followed by a number between -1 and 1 as the sentiment score.

Sentence: "{sent}"
Response:"""
        response = query_together_api(prompt, max_tokens=20)
        label = "NEUTRAL"
        score = 0.0
        if "POSITIVE" in response:
            label = "POSITIVE"
            pos_count += 1
        elif "NEGATIVE" in response:
            label = "NEGATIVE"
            neg_count += 1
        elif "NEUTRAL" in response:
            label = "NEUTRAL"
            neutral_count += 1

        try:
            parts = response.replace(",", "").split()
            score = float([s for s in parts if s.replace('.', '', 1).replace('-', '', 1).isdigit()][-1])
        except Exception:
            score = 0.0

        score_total += score
        scored_sentences += 1
        time.sleep(delay)

    avg_score = round(score_total / scored_sentences, 3) if scored_sentences > 0 else 0.0

    if avg_score > 0.1:
        sentiment = "POSITIVE"
    elif avg_score < -0.1:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return sentiment, avg_score, pos_count, neg_count, neutral_count

def determine_reliability_based_on_sentences(pos_count, neg_count, total_sentences, avg_score):
    if abs(avg_score) <= 0.2:  # Neutral score condition
        return "UNRELIABLE"
    else:
        return "RELIABLE"

def apply_detailed_sentiment(df, review_column="Review Text"):
    print("\n🧠 Running Sentiment Analysis...")
    sentiments = []
    scores = []
    positive_counts = []
    negative_counts = []
    neutral_counts = []
    reliability = []

    for i, text in enumerate(df[review_column]):
        print(f"🔍 Processing {i+1}/{len(df)}")
        sentiment, avg_score, pos_count, neg_count, neutral_count = analyze_review_sentiment(text)
        total_sentences = pos_count + neg_count + neutral_count
        sentence_reliability = determine_reliability_based_on_sentences(
            pos_count, neg_count, total_sentences, avg_score)
        
        sentiments.append(sentiment)
        scores.append(avg_score)
        positive_counts.append(pos_count)
        negative_counts.append(neg_count)
        neutral_counts.append(neutral_count)
        reliability.append(sentence_reliability)

    df["Sentiment"] = sentiments
    df["Sentiment Score"] = scores
    df["Positive_Sentence_Count"] = positive_counts
    df["Negative_Sentence_Count"] = negative_counts
    df["Neutral_Sentence_Count"] = neutral_counts
    df["Reliability"] = reliability

    print("\n📊 Sentiment Summary:")
    print(df["Sentiment"].value_counts())
    return df

if __name__ == "__main__":
    input_csv = "amazon_reviews.csv"
    output_clean_csv = "amazon_reviews_cleaned.csv"
    output_sentiment_csv = "amazon_reviews_sentiment.csv"

    df_clean = clean_data(input_csv)
    df_clean.to_csv(output_clean_csv, index=False)
    print(f"\n✅ Cleaned data saved to {output_clean_csv}")

    df_with_sentiment = apply_detailed_sentiment(df_clean)
    df_with_sentiment.to_csv(output_sentiment_csv, index=False)
    print(f"✅ Full sentiment results saved to {output_sentiment_csv}")
