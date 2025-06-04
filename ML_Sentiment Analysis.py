import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if you haven't already
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Read the CSV file
df = pd.read_csv('C:/Users/csteeves/Desktop/Seller Analysis/CX-ProWeb.csv')


# Perform sentiment analysis on the 'DESCRIPTION' column
def analyze_sentiment(text):
    # Handle NaN values
    if pd.isnull(text):
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    return sia.polarity_scores(str(text))

df['sentiment_scores'] = df['SUBJECT'].apply(analyze_sentiment)

# Extract individual sentiment scores
df['neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])
df['neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])
df['pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])
df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])

# Determine the overall sentiment
def overall_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05 or score == 0:
        return 'Negative'
    else:
        return 'Neutral'

df['overall_sentiment'] = df['compound'].apply(overall_sentiment)

# Save the results to a new CSV file
df.to_csv('CX-ProWeb.csv', index=False)

#print("Sentiment analysis results saved to 'sentiment_analysis_results.csv'")
