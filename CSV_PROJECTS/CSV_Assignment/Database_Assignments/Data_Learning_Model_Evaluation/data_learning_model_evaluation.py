
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# pandas (pd): Used to load and manipulate the dataset.
# numpy (np): Although not explicitly used, it's commonly 
# imported for numerical operations.
# sklearn.model_selection.train_test_split: Splits the 
# dataset into training and testing subsets.
# sklearn.linear_model.LogisticRegression: Implements the 
# Logistic Regression model.
# sklearn.metrics.accuracy_score: Calculates the accuracy 
# score of my model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the dataset
# Reads the data from a CSV file and loads it 
# into a Pandas DataFrame
data_frame = pd.read_csv( "example_dataset_2" )


# Split the dataset into features (x) target (y)
# Features variable
# data.drop(columns=['Target']): Drops the Target 
# column and stores the remaining features in X.
# NOTE: This step separates the independent variables 
# (features) and the dependent variable (target/class label).
x = data_frame.drop( columns=[ 'target' ] )

# Target variable
# data['Target']: Extracts the Target column and 
# stores it in y.
y = data_frame[ 'target' ]


# Split the data into training and testing 
# sets (80% training, 20% testing)
# X, y: Inputs the feature set and the target variable.
# test_size=0.2: Allocates 20% of the dataset for testing 
# and 80% for training.
# random_state=42: Ensures reproducibility (so the same 
# split occurs each time).
#       ---- SPLITTING_RESULTS ----
# X_train: Training features (80% of the data)
# X_test: Testing features (20% of the data)
# y_train: Training labels (80% of the target values)
# y_test: Testing labels (20% of the target values)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42 )


# Initialize the logistic regression model
# Creates an instance of the LogisticRegression model.
# NOTE: Logistic Regression is used for binary 
# classification. Meaning the target variable (y) has 
# two possible values (0 or 1).
model = LogisticRegression()


# Train the logistic regression model 
# using the training data
# Trains (fits) the logistic regression model using 
# the training data.
# The model learns the relationship between features 
# (X_train) and labels (y_train).
# After training, the model can predict target values 
# for new inputs.
model_fit = ( x_train, y_train )


# Make predictions on the testing set using 
# the trained model
# Uses the trained model to predict target values 
# (0 or 1) for the test dataset.
# Stores the predicted values in y_pred.
y_pred = model.predict( x_test )


# Calculate and print the accuracy score of 
# the model's prediction
# Compares the predicted values (y_pred) with 
# the actual values (y_test).
# Returns the proportion of correctly predicted 
# values (i.e., accuracy).
accuracy = accuracy_score( y_test, y_pred )
# String Formats the accuracy score to 2 decimal 
# places and prints it (Example: 6.358569743201 as 6.36).
print( f"Accurac Score: [ accuracy:.2f ]" )



