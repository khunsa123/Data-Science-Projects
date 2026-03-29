# twitter_sentiment_analysis.ipynb
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Create small sample dataset
data = {
    'Tweet': [
        "I love Calgary! Beautiful city and great people.",
        "The traffic in Edmonton is terrible today.",
        "Loving the sunny weather in Red Deer.",
        "Medicine Hat winters are too cold for me.",
        "Lethbridge has amazing restaurants and cafes.",
        "Edmonton's nightlife is fun but expensive.",
        "Calgary Stampede is the best event ever!",
        "Not happy with the public transport in Calgary."
    ]
}

df = pd.DataFrame(data)

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Tweet'].apply(get_sentiment)

# Count of each sentiment
sentiment_count = df['Sentiment'].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(6,5))
sns.barplot(x=sentiment_count.index, y=sentiment_count.values)
plt.title("Sentiment Analysis of Sample Tweets (Alberta)")
plt.ylabel("Count")
plt.xlabel("Sentiment")
plt.tight_layout()
plt.savefig("tweet_sentiment.png")
plt.show()

print(df)
