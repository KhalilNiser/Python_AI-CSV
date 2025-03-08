
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# nltk (Natural Language Toolkit): Is a popular 
# NLP library in Python. 
# word_tokenize: Is a function that splits text 
# into words.
import nltk
from nltk.tokenize import word_tokenize 


# Dowmload the necessary (punkt) NLTK tokenizer 
# package.
# NOTE: The punkt tokenizer is required for 
# word_tokenize to function
nltk.download( 'punkt' )


# Sample paragraph for tokenization
paragraph = "Hello! How are you doing today? This is asimple NLP task"


# Tokenize the paragraph into words
# word_tokenize(paragraph) splits the paragrapg 
# into words and puntuation marks
word_tokens = word_tokenize( paragraph )


# Tokenize each word into individual characters. 
# List comprehension is used to split each word 
# into individual characters. 
char_tokens = [ list( word ) for word in word_tokens ]


# Display the results
# Print, displays the list of word tokens
print( "Word Tokens:", word_tokens )
print( "\nCharacter Tokens:" )

# The for-loop prints each word token followeed 
# by its character tokens
for word, chars in zip( word_tokens, char_tokens ):
    print( f"{ word }: { chars }" )