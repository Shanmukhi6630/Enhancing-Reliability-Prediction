import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def eda_amazon_reviews(csv_file_path, na_values=None):
    if na_values is None:
        na_values = ["None", "none", "NA", "N/A", "n/a", "null", "NULL", "-", ""]

    print(f"\n📥 Reading file: {csv_file_path}")
    df = pd.read_csv(csv_file_path, na_values=na_values)

    print("\n🔹 Basic Info:")
    print(df.info())
    print("\n🔹 First 5 Rows:")
    print(df.head())

    print("\n🔹 Numeric Summary:")
    print(df.describe())
    print("\n🔹 Categorical Summary:")
    print(df.describe(include='object'))

    print("\n🔹 Missing Values:")
    print(df.isnull().sum())

    df.dropna(inplace=True)
    print(f"\n✅ Dropped NA values. New shape: {df.shape}")

    print(f"\n🔹 Duplicate Rows: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)
    print(f"✅ Dropped duplicates. New shape: {df.shape}")

    print("\n🔹 Unique Values Per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")

    if "Review Text" in df.columns:
        df["Review_Length"] = df["Review Text"].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10, 4))
        sns.histplot(df["Review_Length"], bins=50, kde=True)
        plt.title("Distribution of Review Length")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.show()

    if "Sentiment" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Sentiment", order=df["Sentiment"].value_counts().index)
        plt.title("Sentiment Distribution")
        plt.ylabel("Count")
        plt.show()

    if "Rating" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Rating")
        plt.title("Rating Distribution")
        plt.show()

    if "Verified Purchase" in df.columns:
        plt.figure(figsize=(5, 3))
        sns.countplot(data=df, x="Verified Purchase")
        plt.title("Verified Purchase Distribution")
        plt.show()

    df_encoded = df.copy()

    for col in df_encoded.select_dtypes(include=['object', 'string']):
        if df_encoded[col].nunique() < 20:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            df_encoded.drop(columns=[col], inplace=True)

    corr = df_encoded.corr(numeric_only=True)
    print("\n🔹 Correlation Matrix:")
    print(corr)

    if not corr.empty:
        plt.figure(figsize=(12, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    print("\n✅ EDA complete.")
    return df

df = eda_amazon_reviews("sentiment_batched.csv")
df.to_csv("cleaned_modeling.csv", index=False)
