
#       ---- PYTHON_CODE:_SENTIMENT_(SENTENCE)_ANALYSIS_USING_NLTK ----
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# nltk: Is a natural language processing library in Python
# SentimentIntensityAnalyzer: Is used for analyzing sentiment
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer 


# Download the VADER lexicon
# This downloads the "VADER lexicon sentiment", which 
# contasins words with predefined sentiment scores
nltk.download( 'vader_lexicon' )


#       ---- SENTIMENT_ANALYSIS_FUNCTION ----
def analyze_sentiment( sentence ):
    # Create a sentiment analyzerr object
    # SentimentIntensityAnalyzer(): Loads the VADER lexicon
    sia = SentimentIntensityAnalyzer()
    # Get seentiment scores
    sentiment_score = sia.polarity_scores( sentence )
    
    # Determine overall sentiment
    # The compund score (ranges from -1 to 1)
    if sentiment_score[ 'compound' ] >= 0.05:
        # Positive sentiment
        sentiment = "Positive"
    elif sentiment_score[ 'compound' ] <= -0.05:
        # Negative sentiment
        sentiment = "Negative"
    else:
        # Neutral sentiment
        sentiment = "Neutral"
        
    return sentiment, sentiment_score

# Example usage
# Asks the user to enter a sentence, retreives that
# sentence, and assigns it variable sentence 
sentence = input( "Enter a Sentence: " )
sentiment, scores = analyze_sentiment( sentence )
print( f"Sentiment: {sentiment}" )
print( f"Scores: {scores}" )