
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# TensorFlow/Keras: Used to build and train 
# the CNN model
# NumPy: Helps in handling arrays
# Matplotlib: Used to visualize predictions
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 
import numpy as np 
import matplotlib.pyplot as plt 


# Load the MNIST dataset (handwritten digits 0 - 9)
( x_train, y_train ), ( x_test, y_test ) = keras.dataset.mnist.load_data()


# Normalize pixel (scales pixel values between 
# 0 and 1) for better training
x_train, x_test = x_train / 255.0, x_test / 255.0


# Reshape images to [28, 28, 1], since CNNs 
# expect 3D input (height, weight, channels). 
# So CNN can process "grayscale images"
x_train = x_train.reshape( -1, 28, 28, 1 )
x_test = x_test.reshape( -1, 28, 28, 1 )


# Define the CNN Model
model = keras.Sequential ([
    # First convertion layer (32 filters of 
    # size 3x3)
    layers.Conv2D( 32, ( 3, 3 ), activation = 'relu', input_shape = ( 28, 28, 1 ) ),
    # Pooling layer (Reduces spatial size to 
    # "prevent overfitting")
    layers.MaxPooling2D( ( 2, 2 ) ),
    # Second convertion layer
    layers.Conv2D( 64, ( 3, 3 ), activation = 'relu' ),
    # Pooling layer
    layers.MaxPooling2D( ( 2, 2 ) ),
    # Flatten into a 1D vector (Converts 2D 
    # feature maps into "1D vector")
    layers.Flatten(),
    # Fully connected layer
    layers.Dense( 128, activation = 'relu' ),
    # Output layer of 10 digit classification 
    layers.Dense( 10, activation = 'softmax' )
])


# Compile the model
# Adam Optimizer: Efficient learning 
# rate adaptation
model.compile( optimizer = 'adam',
            # Sparse categorical Crossentropy: 
            # Best loss function for "multi-class 
            # classification"  
              loss = 'sparse_categorical_crossentropy',
              metrics = [ 'accuracy' ] )


# Train the model (Trains for 5 epochs. 
# Uses validation data (test set) to 
# check performance)
model.fit( x_train, y_train, epochs = 5, validation_data = ( x_test, y_test ) )


# Evaluate the model
# Computes accuracy on unseen dataset
test_loss, test_acc = model.evaluate( x_test, y_test )
print( f"Test Accuracy: { test_acc:.4f}" )


# Make predictions (Model "predicts digit 
# class" for each test image)
predictions = model.predict( x_test )


# Display a sample image of the prediction: 
# Test Accuracy Printed around 98% accuracy
plt.imshow( x_test[ 0 ].reshape( 28, 28 ), cmap = 'gray' )
plt.title( f"Predicted Label: {np.argmax( predictions[ 0 ] ) }" )
plt.show()