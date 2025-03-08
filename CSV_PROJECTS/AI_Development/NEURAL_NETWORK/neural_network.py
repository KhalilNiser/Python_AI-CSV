
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# Tensorflow: Is the core deep learning library.
# Keras: Is the high-level API within "tensorflow", 
# for building neural networks. 
# Sequential: Is a simple way to create a model by 
# stacking layers. 
# Dense: Is a fully connected layer in a neural 
# network.
# NumPy: Used to create and manipulate data arrays.
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 


# Generate a random trianing data. 
# For reproducibility: Ensures the random numbers 
# are the same every time.
np.random.seed( 42 )

# 100 samples of 2 features: It's generating 100 
# random samples, each with 2 features between 
# 0 and 1. 
x_train = np.random.rand( 100, 2 )
# Simple binary classification: Labels "y_train": 
# Are assigned 1 if the sum of the 2 features is 
# greater than 1, otherwise 0. This creates a 
# simple binary classification problem
y_train = ( x_train[ :, 0 ] + x_train[ :, 1 ] > 1 ).astype( int )


# Define the neural network model
# Sequential model: Layers are stacked in order.
model = Sequential([
    # Hidden layer with 4 neurons
    # Four neurons with ReLU activation: 
    # NOTE: ReLU (Rectified Linear Unit), is a 
    # commonly used activation function in 
    # artificial neural networks that outputs the 
    # input value if it's positive, and outputs 
    # zero if it's negative, essentially 
    # introducing a non-linearity to the network 
    # by "rectifying" the input to only positive 
    # values; mathematically represented as 
    # f(x) = max(0, x) making it a simple yet 
    # effective function for deep learning models
    Dense( 4, activation = 'relu', input_shape = ( 2, ) ),
    # Output layer with 1 neuron (binary 
    # classification) Sigmoid outputs a probability 
    # between 0 and 1.
    Dense( 1, activation = 'sigmoid' )
])


# Compile the model
# Optimizer: adam (adaptive learning rate). 
# Loss Function: binary_crossentropy (Used for 
# binary classification). Metrics (Monitors the 
# performance for "accuracy"). 
model.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = [ 'accuracy' ] )


# Train the model
model.fit( x_train, y_train, epochs = 50, verbose = 1 )


# Make a prediction
# Exact input
sample_input = np.array( [ [ 0.2, 0.8, ] ] )
prediction = model.predict( sample_input )

print( f"Predicted Probability: {prediction[ 0 ][ 0 ]:.4f}" )