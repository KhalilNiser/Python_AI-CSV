#       ---- PYTHON_CODE:_SENTIMENT_(PARAGRAPH)_ANALYSIS_USING_TextBlob ----
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# TextBlob is a NLP library built on top of 
# NLTK. It provides "simple sentiment analysis, 
# word_tokenization, spelling correction" and 
# more. 
# NOTE: Since I am analysing the sentiment of 
# an entire paragraph. I will use the "TextBlob" 
# library. Which is more adequate for handling 
# "longer texts" and it provides a simple way to 
# get the "polarity" (sentiment) of a paragraph 
from textblob import TextBlob


#       ---- PARAGRAPH_SENTIMENT_ANALYSIS_FUNCTION ----
def analyze__paragraph_sentiment( paragraph ):
    # Converts paragraph into a TextBlob object: 
    # Which allows for better sentiment analysis
    blob = TextBlob( paragraph )
    # Extract the polarity (-1 to 1)
    polarity = blob.polarity.semtiment 
    
    
    # Determines sentiment based on its polarity 
    # score.
    # Polarity sentiment scale (-1 to 1): 
    # -1 = "Completely Negative"; 
    # 0 = "Neutral"; 
    # +1 = "Completely Positive"
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    
    return sentiment, polarity


# Example usage
paragraph = input( "Enter a paragraph" )
sentiment, polarity_score = analyze__paragraph_sentiment( paragraph )
print( f"Sentiment: { sentiment }" )
print( f"Polarity Score: { polarity_score }" )
    