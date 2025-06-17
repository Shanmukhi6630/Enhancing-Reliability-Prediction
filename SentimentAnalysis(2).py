import time
import pandas as pd
import spacy
import requests
import re
import json

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

def is_gibberish(text, min_meaningful_words=2, max_non_alpha_ratio=0.3):
    if not text or not text.strip():
        return True
    doc = nlp(text)
    meaningful_words = [token for token in doc if token.is_alpha and not token.is_stop]
    if len(meaningful_words) < min_meaningful_words:
        return True
    non_alpha_chars = sum(1 for c in text if not c.isalpha() and not c.isspace())
    total_chars = len(text)
    if total_chars == 0:
        return True
    if (non_alpha_chars / total_chars) > max_non_alpha_ratio:
        return True
    return False

def query_api(prompt, max_tokens=300):
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
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return "ERROR"

def build_batch_prompt(reviews):
    prompt = (
        "Classify the sentiment of the following reviews. "
        "For each review, respond with a list of labels and scores for each sentence in this format: "
        "[POSITIVE 0.8, NEGATIVE -0.6, NEUTRAL 0.0, ...] (in order).\n\n"
    )
    for i, review in enumerate(reviews, start=1):
        prompt += f"Review {i}:\n"
        sentences = sentence_splitter(review)
        for idx, sent in enumerate(sentences, start=1):
            prompt += f"{idx}. {sent}\n"
        prompt += "\n"

    prompt += "Respond with the sentiment list per review, one line per review, starting with 'Review 1:', 'Review 2:', etc.\n"
    return prompt

def parse_batch_response(response, batch_size):
    results = []
    split_reviews = re.split(r'Review\s+\d+:', response, flags=re.IGNORECASE)
    split_reviews = [part.strip() for part in split_reviews if part.strip()]
    
    for part in split_reviews[:batch_size]:
        items = re.findall(r'(POSITIVE|NEGATIVE|NEUTRAL)\s*(-?\d*\.?\d+)', part.upper())
        pos_count = neg_count = neutral_count = 0
        score_total = 0.0
        scored_sentences = 0
        for label, score_str in items:
            score = float(score_str)
            score_total += score
            scored_sentences += 1
            if label == "POSITIVE":
                pos_count += 1
            elif label == "NEGATIVE":
                neg_count += 1
            elif label == "NEUTRAL":
                neutral_count += 1

        avg_score = round(score_total / scored_sentences, 3) if scored_sentences > 0 else 0.0
        if avg_score > 0.1:
            sentiment = "POSITIVE"
        elif avg_score < -0.1:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        results.append({
            "Sentiment": sentiment,
            "Sentiment Score": avg_score,
            "Positive_Sentence_Count": pos_count,
            "Negative_Sentence_Count": neg_count,
            "Neutral_Sentence_Count": neutral_count,
        })

    while len(results) < batch_size:
        results.append({
            "Sentiment": "NEUTRAL",
            "Sentiment Score": 0.0,
            "Positive_Sentence_Count": 0,
            "Negative_Sentence_Count": 0,
            "Neutral_Sentence_Count": 0,
        })
    return results

def determine_reliability(avg_score, review_text):
    if is_gibberish(review_text):
        return "UNRELIABLE"
    if not review_text.strip() or len(review_text.split()) < 3:
        return "UNRELIABLE"
    if abs(avg_score) <= 0.2:
        return "UNRELIABLE"
    return "RELIABLE"

def apply_batched_sentiment(df, review_column="Review Text", batch_size=20, delay=1.5):
    print("\n🧠 Running Batched Sentiment Analysis...")
    all_sentiments = []
    all_scores = []
    all_pos_counts = []
    all_neg_counts = []
    all_neutral_counts = []
    all_reliability = []

    num_reviews = len(df)
    for start_idx in range(0, num_reviews, batch_size):
        batch_reviews = df[review_column].iloc[start_idx:start_idx + batch_size].tolist()
        print(f"🔍 Processing batch {start_idx // batch_size + 1} / {(num_reviews + batch_size -1)//batch_size} ({len(batch_reviews)} reviews)")

        gibberish_flags = [is_gibberish(r) for r in batch_reviews]

        if all(gibberish_flags):
            batch_results = [{
                "Sentiment": "NEUTRAL",
                "Sentiment Score": 0.0,
                "Positive_Sentence_Count": 0,
                "Negative_Sentence_Count": 0,
                "Neutral_Sentence_Count": 1,
            }] * len(batch_reviews)
        else:
            reviews_to_api = [r for i,r in enumerate(batch_reviews) if not gibberish_flags[i]]
            prompt = build_batch_prompt(reviews_to_api)
            response = query_api(prompt, max_tokens=4000)
            api_results = parse_batch_response(response, len(reviews_to_api))

            batch_results = []
            api_idx = 0
            for is_gib in gibberish_flags:
                if is_gib:
                    batch_results.append({
                        "Sentiment": "NEUTRAL",
                        "Sentiment Score": 0.0,
                        "Positive_Sentence_Count": 0,
                        "Negative_Sentence_Count": 0,
                        "Neutral_Sentence_Count": 1,
                    })
                else:
                    batch_results.append(api_results[api_idx])
                    api_idx += 1

        for idx, res in enumerate(batch_results):
            review_text = batch_reviews[idx]
            reliability = determine_reliability(res["Sentiment Score"], review_text)
            all_sentiments.append(res["Sentiment"])
            all_scores.append(res["Sentiment Score"])
            all_pos_counts.append(res["Positive_Sentence_Count"])
            all_neg_counts.append(res["Negative_Sentence_Count"])
            all_neutral_counts.append(res["Neutral_Sentence_Count"])
            all_reliability.append(reliability)

        time.sleep(delay)

    df["Sentiment"] = all_sentiments
    df["Sentiment Score"] = all_scores
    df["Positive_Sentence_Count"] = all_pos_counts
    df["Negative_Sentence_Count"] = all_neg_counts
    df["Neutral_Sentence_Count"] = all_neutral_counts
    df["Reliability"] = all_reliability

    print("\n📊 Sentiment Summary:")
    print(df["Sentiment"].value_counts())
    print("\n📊 Reliability Summary:")
    print(df["Reliability"].value_counts())
    return df

def export_full_finetune_json(df, output_file="finetune_data.jsonl"):
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.dropna().to_dict()
            f.write(json.dumps(record) + "\n")

    print(f"✅ Full fine-tuning JSONL saved to {output_file}")


if __name__ == "__main__":
    input_csv = "amazon_reviews.csv"
    output_clean_csv = "cleaned.csv"
    output_sentiment_csv = "sentiment_batched.csv"

    df_clean = clean_data(input_csv)
    df_clean.to_csv(output_clean_csv, index=False)
    print(f"\n✅ Cleaned data saved to {output_clean_csv}")

    df_with_sentiment = apply_batched_sentiment(df_clean, batch_size=20, delay=1.5)
    df_with_sentiment.to_csv(output_sentiment_csv, index=False)
    print(f"✅ Full sentiment results saved to {output_sentiment_csv}")

export_full_finetune_json(df_with_sentiment)
