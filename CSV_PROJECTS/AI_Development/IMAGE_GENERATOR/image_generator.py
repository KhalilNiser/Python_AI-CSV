
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# Tensorflow: Provides deep learning functionality.
# Keras Models(Sequential): Allows us to build a 
# "layerby-layer" neural network. 
# Dense Layers: Fully connected layers for both 
# Generator and Discriminator. 
# LeakyReLU: Activation function to prevent 
# "vanishing gradients". 
# BatchNormalization: Helps stabilize trianing by 
# normalizing activations. 
# Adam Optimizer: Used to optimize both networks. 
# NumPy: For handling arrays and generating random noise. 
# Matplotlib: Used to visualize generated images
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten 
from tensorflow.keras.optimizers import Adam 
import numpy as np
import matplotlib.pyplot as plt 


#       ---- DEFINE_THE_GENERATOR_FUNCTION ----
# This "Generator", takes random noise as 
# input and produces a 28x28 grayscale image.
def build_generator():
    model = Sequential ([
        # Input: Random noise (100 random numbers)
        # Uses "Dense (fully connected) layers" to expand this noise into a "28x28 image"  
        Dense( 128, input_dim = 100 ),
        # Helps avoid vanishing gradients: 
        # For Better gradient flow. 
        LeakyReLU( alpha = 0.2 ),
        # Stabalizes training by normalizing activations
        BatchNormalization(),
        Dense( 256 ),
        LeakyReLU( alpha = 0.2 ),
        BatchNormalization(),
        Dense( 512 ),
        LeakyReLU( alpha = 0.2 ),
        BatchNormalization(),
        # Output: Image(28x28 pixels): The final 
        # "Dense layer outputs 784 values (which 
        # represent a 28x28 grayscale image)". The
        # "tanh activation" function ensures the 
        # pixel values are between "-1 and 1 (which 
        # helps the GAN learn better)"  
        Dense( 28*28, activation = 'tanh' ),
        # Reshape the 28x28 image
        Reshape( ( 28, 28 ) )
    ])
    return model


#       ---- DEFINE_THE_DISCRIMINATOR_FUNCTION ----
def build_discriminator():
    model = Sequential ([
        # Input: 28x28 image: The "Discriminator" 
        # takes a "28x28 image" as input. "Flattens" 
        # the image into "1D array (784 values)"
        Flatten( input_shape = ( 28, 28 ) ),
        # Passes through "Dense layers" with 
        # "LeakyReLU activations"
        Dense( 512 ),
        LeakyReLU( alpha = 0.2 ),
        Dense( 256 ),
        LeakyReLU( alpha = 0.2 ),
        # Output: Probability (real/fake): The 
        # "final layer outputs a single value 
        # (0 = fake or 1 = real)". Uses "sigmoid 
        # avtivation", to ensure the output is 
        # between 0 and 1 (binary digits)
        Dense( 1, activation = 'sigmoid' )
    ])
    return model


#       ---- COMPILE_THE_DISCRIMINATOR ----
discriminator = build_discriminator()
# "Binary cross-entropy loss": Used for 
# binary classification (real or fake). 
# "Adam optimizer": Helps stabilize training. 
# "Metrics": Allows in tracking the 
# discriminator's performance
discriminator.compile( loss = 'binary_crossentropy', optimizer = Adam( 0.0002, 0.5 ), metrics = [ 'accuracy' ] )


#       ---- BUILD_AND_COMPILE_THE_GAN ----
# Freeze discriminator when training GAN
discriminator.trainable = False

# Input: Random noise (100 numbers)
gan_input = tf.keras.Input( shape = ( 100, ) )
# Output: Produces a "fake image"
generated_image = build_generator()( gan_input )
# Classification (real/fake)
gan_output = discriminator( generated_image )
# NOTE: The "GAN's goal" is to train the Generator 
# so that the "Discriminator gets fooled" 

gan = tf.keras.Model( gan_input, gan_output )
gan.compile( loss = 'binary_crossetropy', optimizer = Adam( 0.0002, 0.5 ) )


#       ---- TRAIN_THE_GAN ----
# Load MNIST dataset
( x_train, _ ), ( _, _ ) = tf.keras.datasets.mnist.load_data()
# Normalize images to [-1, 1]
x_train = ( x_train - 127.5 ) / 127.5
# Add channel dimension
x_train = np.expand_dims( x_train, axis = -1 )


# Training parameters
epochs = 5000
batch_size = 64

half_batch = batch_size // 2


for epoch in range( epochs ):
    # Train Discriminator
    idx = np.random.randint( 0, x_train.shape[0], half_batch )
    real_images = x_train[idx]
    
    # Random Noise
    noise = np.random.normal(0, 1 (half_batch, 100))
    fake_images = build_generator().predict(noise) 
    
    
    # Real = 1
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
    # Fake = 0
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
    # Average loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    # Try ti fool discriminator
    valid_y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)
    
    
    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}: D Loss = {d_loss[0]}, G Loss = {g_loss}")
        
        
        # Cretae and save an image sample
        noise = np.random.normal(0, 1, (1, 100))
        gen_image = build_generator().predict(noise)[0]
        plt.imshow(gen_image, cmap='gray')
        plt.title(f"Generated Image at Epoch: {epoch}")
        plt.show()


