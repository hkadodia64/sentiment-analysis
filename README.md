# sentiment-analysis
# Importing the required libraries
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd
df = pd.read_csv('/content/sentiment140.csv', encoding='ISO-8859-1', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'])
df.head()
import re

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = re.sub(r'\W', ' ', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text)
    return processed_text

df['processed_text'] = df['text'].apply(preprocess_text)
from textblob import TextBlob

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'
sentiments = []
for text in df['processed_text']:
    sentiment = get_sentiment(text)
    sentiments.append(sentiment)
import matplotlib.pyplot as plt

# Count the number of tweets with each sentiment label
sentiment_counts = df['sentiment'].value_counts()

# Create a bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values)

# Add axis labels and title
plt.xlabel('Sentiment Label')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Distribution')

# Show the chart
plt.show()
