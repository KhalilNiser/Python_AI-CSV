
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# Imports "Tensorflow and Keras" for deep learning
# Loads "VGG16" from "Keras applications"
# Uses "Dense, Flatten, Dropout", to modify layers
# "Adam Optimizer": Is used for better learning
# "ImageDataGenerator": Helps load and preprocess images
import tensorflow as tf 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras.optimizsrs import Adam 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt 

#       ---- LOAD_PRE_TRAINED_VGG16_MODEL ----
# "weights=imagenet": Loads the model trained 
# on "ImageNet"
# "include_top=False": Removes the original 
# classifier layers
# "input_shape=(224, 224, 3)": Takes "RGB 
# images of size 224x224"
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


#       ---- FREEZE_PRE_TRAINED_LAYERS ---- 
# Freezes the "VGG16 convolutional layers". 
# Prevents them from "updating weights" while 
# training. This ensures the model "retains 
# its learned features"
for layer in base_model.layers:
    layer.trainable = False
    
    
#       ---- ADD_NEW_FULLY_CONNECTED_LAYERS ----
# Flatten(): Flatten the feature maps. Convert 
# the feture maps into a "1D array"
x = Flatten()(base_model.output)
# Fully connected layer (256 neurons)
x = Dense( 256, activation='relu')(x)
# Dropout to rpevent overfitting: Prevents 
# overfitting by randomly "deactivating 50% 
# of neurons"
x = Dropout(0.5)(x)
# Another fully connected dense layer
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
# Uses "softmax activation" for "2-class 
# classification". If more than 2 classes change 
# "Dense(2)", to the number of classes
output_layer = Dense(2, activation='softmax')(x)


#       ---- CREATE_THE_FINAL_MODEL ----
# Creates a new model that connects VGG16 
# with "custom layers"
model = Model(inputs=base_model.input, outputs=output_layer)


#       ---- COMPILE_THE_MODEL ----
# Adam Optimizer: Adjusts learning dynamically
# Categorical Crossentropy: Used for "multi-class classification"
# Mertics[Accuracy]: Tracks the performance
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#       ---- LOAD_AND_PREPROCESS_THE_DATASET ----
# ImageDataGenerator: Prepares images for trianing
# rescale=1./255: Normalizes pixel values "between 0 and 1"
# rotation_range, zoom_range, horizontal_flip: Augments images to improve learning
# flow_from_directory: Loads images from folders
# dataset/train/test: Trains and tests the images
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('dataset/train', target_size=(224,224), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory('dataset/test', target_size=(224, 224), batch_size=32, class_mode='categorical')



#       ---- TRAIN_THE_MODEL ----
# Trains the model for 10 epochs
# validation_data: Helps track performance on 10 images
history = model.fit(train_data, epochs=10, validation_data=test_data)


#       ---- PLOT_TRAINING_RESULTS ----
# Plots "train vs validation accuracy"
# Helps visualize training performance
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='Validation_Accuracy')
plt.legend()
plt.title('Training Progress')
plt.show()



