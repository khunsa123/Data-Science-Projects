# 🐦 Twitter Sentiment Mini-Analysis

## 📌 Overview

A mini **NLP sentiment analysis pipeline** applied to a sample set of tweets about Alberta cities. The project uses TextBlob to compute polarity scores and classifies each tweet as Positive, Negative, or Neutral, with results visualised as a sentiment distribution chart.

---

## 🎯 Objectives

- Apply lexicon-based sentiment analysis to short social media text
- Classify tweets into Positive, Negative, and Neutral categories using TextBlob polarity
- Visualise the sentiment distribution across the sample dataset

---

## 🧠 Methods

- Sentiment polarity scoring via **TextBlob** (`sentiment.polarity`)
- Rule-based classification: Positive (score > 0.1), Negative (score < -0.1), Neutral (otherwise)
- Bar chart visualisation of sentiment label counts with Seaborn

---

## 📊 Dataset

- **Type:** Sample dataset of tweets about Alberta cities (created programmatically)
- **Content:** Short social media style text referencing Calgary, Edmonton, Red Deer, Lethbridge, and Medicine Hat
- **Labels:** Positive, Negative, Neutral (assigned via TextBlob polarity)

---

## 📈 Outputs

| File | Description |
|------|-------------|
| `tweet_sentiment.png` | Bar chart of sentiment distribution across sample tweets |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** TextBlob, Pandas, Matplotlib, Seaborn

---

## ⚙️ Usage

```bash
pip install textblob pandas matplotlib seaborn
python -m textblob.download_corpora
python twitter_sentiment_analysis.py
```

---

## 📬 Contact

**Khunsa Iftikhar**
📧 [khunsaiftikhar123@gmail.com](mailto:khunsaiftikhar123@gmail.com)
🔗 [linkedin.com/in/khunsa-iftikhar](https://www.linkedin.com/in/khunsa-iftikhar/)
