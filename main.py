# ==============================
# SENTIMENT ANALYSIS PROJECT
# ==============================

import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud
import nltk

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')

# ------------------------------
# 1. DATASET GENERATION
# ------------------------------

years = list(range(2018, 2025))
products = ["iPhone", "iPad", "Apple Watch", "Mac"]
sources = ["Twitter", "Reddit", "News", "Reviews"]

positive_reviews = [
    "Amazing camera quality and smooth performance",
    "Excellent battery life and fast processor",
    "Love the ecosystem integration",
    "Very secure and great privacy features",
    "Beautiful design and premium feel"
]

negative_reviews = [
    "Too expensive for base storage",
    "Battery drains very quickly",
    "Camera struggles in low light",
    "App permissions are too restrictive",
    "Overpriced compared to competitors"
]

neutral_reviews = [
    "The device was launched this year",
    "It comes with multiple storage options",
    "Available in different colors",
    "Supports latest software update",
    "New model released recently"
]

data = []

for _ in range(2000):
    year = random.choice(years)
    product = random.choice(products)
    source = random.choice(sources)

    sentiment_type = random.choice(["positive", "negative", "neutral"])

    if sentiment_type == "positive":
        review = random.choice(positive_reviews)
    elif sentiment_type == "negative":
        review = random.choice(negative_reviews)
    else:
        review = random.choice(neutral_reviews)

    data.append([year, product, review, source, sentiment_type])

df = pd.DataFrame(data, columns=["year", "product", "review_text", "source", "actual_sentiment"])

print("Dataset Created Successfully!")
print(df.head())

# ------------------------------
# 2. DATA PREPROCESSING
# ------------------------------

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_text"] = df["review_text"].apply(preprocess)

# ------------------------------
# 3. SENTIMENT ANALYSIS USING VADER
# ------------------------------

sia = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df["predicted_sentiment"] = df["cleaned_text"].apply(predict_sentiment)

# ------------------------------
# 4. MODEL EVALUATION
# ------------------------------

print("\nAccuracy:", accuracy_score(df["actual_sentiment"], df["predicted_sentiment"]))
print("\nClassification Report:\n")
print(classification_report(df["actual_sentiment"], df["predicted_sentiment"]))

print("\nConfusion Matrix:\n")
print(confusion_matrix(df["actual_sentiment"], df["predicted_sentiment"]))

# ------------------------------
# 5. YEAR-WISE ANALYSIS
# ------------------------------

year_analysis = df.groupby(["year", "predicted_sentiment"]).size().unstack()
year_analysis.plot(kind="line", marker="o")
plt.title("Year-wise Sentiment Trend")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# ------------------------------
# 6. PRODUCT-WISE ANALYSIS
# ------------------------------

product_analysis = df.groupby(["product", "predicted_sentiment"]).size().unstack()
product_analysis.plot(kind="bar")
plt.title("Product-wise Sentiment Comparison")
plt.xlabel("Product")
plt.ylabel("Count")
plt.show()

# ------------------------------
# 7. SOURCE-WISE ANALYSIS
# ------------------------------

source_analysis = df.groupby(["source", "predicted_sentiment"]).size().unstack()
source_analysis.plot(kind="bar")
plt.title("Source-wise Sentiment Comparison")
plt.xlabel("Source")
plt.ylabel("Count")
plt.show()

# ------------------------------
# 8. BEFORE VS AFTER LAUNCH
# ------------------------------

before_launch = df[df["year"] < 2021]
after_launch = df[df["year"] >= 2021]

print("\nBefore 2021 Sentiment Distribution:")
print(before_launch["predicted_sentiment"].value_counts())

print("\nAfter 2021 Sentiment Distribution:")
print(after_launch["predicted_sentiment"].value_counts())

# ------------------------------
# 9. PIE CHART
# ------------------------------

df["predicted_sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Overall Sentiment Distribution")
plt.ylabel("")
plt.show()

# ------------------------------
# 10. WORDCLOUD
# ------------------------------

text = " ".join(df["cleaned_text"])
wordcloud = WordCloud(width=800, height=400).generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.title("WordCloud of Reviews")
plt.show()

print("\nPROJECT EXECUTED SUCCESSFULLY!")