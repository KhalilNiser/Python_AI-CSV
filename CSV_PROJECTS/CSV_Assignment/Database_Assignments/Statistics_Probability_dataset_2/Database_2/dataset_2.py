
#               ---- IMPORT_REQUIRED_LIBRARIES ----  
# IMPORT_(os.environ[]): Ignores TensorFlow CPU Instruction warnings
import os
os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import random

# Define the number of samples and features
num_samples = 100
num_features = 5

# Generater Random Data from the dataset
data = 
{
    f"feature_{ i + 1 }": [ random.uniform( 0, 1 ) for _ in range( num_samples ) ]
    for i in range( num_features )
}

data[ "species" ] = [ random.choice( [ "cat", "dog" ] ) for _ in range( num_samples ) ]

# Create a DataFrame from the data
dataset = pd.DataFrame( data )

# Save the dataset to a CSV file
dataset.to_csv( "dataset_2.csv", index = False )

# Preprocess the data
# Remove the 'species' column as an input feature
x = dataset.drop( columns = [ 'species' ] )
# Set the 'species' column as the "target" variable
y = dataset[ 'species' ] 
# Initialize the label encoder
le = LabelEncoder()
# Encode the target variable with numerical labels
y = le.fit_transform( y )

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 42 )

# Create the model 
model = Sequential()
model.add( Dense( 10, activation = 'relu', input_shape = ( x_train.shape[ 1 ], ) ) )
model.add( Dense( 10, activation = 'relu' ) )
model.add( Dense( len( le.classes_), activation = 'softmax' ) )

# Compile and train the model
model.compile( loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = [ 'accuracy' ] )
model.fit( x.train.values, y.train, epochs = 10, batch_size = 32 )

# Evaluate the model on the test set
_, accuracy = model.evaluate( x_test.values, y_test )
print( f"Accuracy: {accuracy}" )

# NOTE: Epoch: One complete pass through the entire training 
# dataset during the training of a Neural Network. In normal 
# English "Epoch refers to a period of time or an age"

# NOTE: Step: Refers to the processing of one batch of the dataset 
# during training. It involves forward propagation (the process of 
# passing input data through a neural network to generate predictions), 
# computing the loss, and updating the model's parameters using backpropagation 
# (the process of calculating gradients to update weights [numerical values 
# representing the strength of connections between neural network units] 
# and biases [numerical values added to neuron inputs to adjust activation 
# thresholds] in a neural network).

# NOTE: Loss: A measure of how well the model's predictions match the 
# desired output. It quantifies the error between predicted and actual 
# values.

# NOTE: Accuracy: A metric that measures the percentage of correctly 
# predicted outputs compared to the total number of inputs in the 
# evaluation or testing phase.

# NOTE: Batch size: The number of samples used in each step of training 
# before updating the model's parameters. It allows for more efficient 
# computation by processing data in smaller subsets instead of the 
# entire dataset.

# NOTE: Loss function: A quantification of the model's prediction error. 
# It calculates the difference between predicted and actual values, 
# providing a measure for the model to minimize during training.

# NOTE: Optimizer: An algorithm used to update the model's parameters 
# based on the computed gradients of the loss function. It determines 
# how the model learns and adjusts its internal weights to minimize 
# the loss.

# NOTE: Accuracy metric: Accuracy is a metric used to evaluate the 
# performance of a model. It measures the percentage of correctly 
# predicted outputs compared to the total number of inputs in the 
# evaluation phase.

#               ---- CODE_SUMMARY ----
# To summarize the code in this assignment: It generates random 
# data for a dataset with a specified number of samples and 
# features. It saves the dataset to a CSV file. The data is 
# preprocessed by removing the ‘species’ column as input 
# features and encoding the ‘species’ column with numerical labels. 
# The dataset is then split into training and testing sets. A 
# sequential model is created with multiple dense layers. The 
# model is compiled with a specified loss function and optimizer, 
# and it is trained on the training data. The accuracy of the 
# model is evaluated on the test set, and the result is printed.






