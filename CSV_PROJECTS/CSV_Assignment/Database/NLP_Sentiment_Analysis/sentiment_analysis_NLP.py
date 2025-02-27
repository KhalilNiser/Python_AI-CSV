
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# NOTE: Pandas: Handles dataset loading and manipulation. 
# CounterVectorizer: Converts text data into numerical 
# form for the SVM Model. 
# train_test_split: Splits the dataset into training and 
# testing sets. 
# SVC: Support Vector Classifier for sentiment analysis. 
# accuracy_score: Evaluates the model's performance
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Dataset file loading: Reads a CSV file into a pandas 
# DataFrame
file_path = "example_dataset_3.csv"
df = pd.read_csv( file_path )


# Select the columns for sentiment analysis
# The column containing the text data
text_column = "text"
# Column containing sentiment labels
sentiment_column = "sentiment"


# Split the data into features (x) and target (y)
# Stores text data (features)
x = df[ text_column ]
# Stores sentiment labels (target)
y = df[ sentiment_column ]


# Split the data into training and testing sets. 
# "train_test_spli()" - Splits the dataset into: 
# 80% training data (x_train, y_train), and 20% 
# testing data (x_test, y-test)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42 )


# Vectorize the text data
# Converts raw text into a matrix of token counts
vectorizer = CountVectorizer()
# Fit and transform the training data
# Fits the vectorizer on training data and 
# transforms it
x_train_vectorized = vectorizer.fit_transform( x_train )
# Transform the test data
# Converts the test data into the same format
x_test_vectorized = vectorizer.transform( x_test )


# Initializing the SVM classifier
# Creates an insatnce of the Suport Vector 
# Classifier (SVC)
svm_classifier = SVC()


# Train the model
svm_classifier.fit( x_train_vectorized, y_train )


# Make predictions on the testing set
y_pred = svm_classifier.predict( x_test_vectorized )


# Evaluate the model's performance
# Calculates the accuracy by comparing 
# predictions to actual labels
accuracy = accuracy_score( y_test, y_pred )


# Print the model's accuracy
# Displays the accuracy score as a percentage 
# (e.g., 0.89 --> 89% accuracy)
print( f"Model Accuracy: {accuracy:.4f}" )
