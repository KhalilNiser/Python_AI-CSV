
#       IMPORT_REQUIRED_LIBRARIES
import os
# Supress Tensorflow info/warning messages
os.environ[ 'TP_CPP_MIN_LOG_LEVEL' ] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split


# Load the MNIST dataset
( x_train, y_train ), ( x_test, y_test ) = mnist.load_data()


# Preprocess the images
x_train = x_train / 255.0
x_test = x_test / 255.0


# Split the dataset into training and validation sets (https://www.learncodinganywhere.com/Student/PageView/ViewPage?courseId=387&pageNumber=442)
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.2, random_state=42 )


# TODO: Define the Neural Network Architechture.
model = Sequential([
    # Faltten 28x28 images into a 1D array
    Flatten( input_shape = ( 28, 28 ) ),
    # Frist hidden layer
    Dense( 128, activation = 'relu' ),
    # Output layer (10 classes for digits 0-9)
    Dense( 10, activation = 'softmax' )
])


# Compile the model
model.compile( optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[ "accuracy" ] ) 


# TODO: Train the model.
model.fit( x_train, y_train, validation_data = ( x_val, y_val ), epochs = 10 )


# TODO: Evaluate the model on the test set and print the loss and the accuracy.
test_loss, test_accuracy = model.evaluate( x_test, y_test )
print( f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}" )
