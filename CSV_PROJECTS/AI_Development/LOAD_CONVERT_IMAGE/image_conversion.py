
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# cv2: Is the openCV library, which is used 
# for image processing and computer vision
import cv2

# Load the image
image = cv2.imread( "dog.jpg" )


# NOTE: OpenCV loads images in "BGR (BLUE_
# GREEN_RED) format", by default (not RGB).
# Convert the image to grayscale
gray_image = cv2.cvtColor( image, cv2.COLOR_BGRGRAY )


# Display the original and the grayscale images
# Original image
cv2.imshow( "Original Image", image )
# Grayscale image
cv2.imshow( "Grayscale Image", gray_image )


# Wait for a key press and close all windows
# Pauses execution "until a key is pressed"
cv2.waitkey( 0 )
# Closes all OpenCV image windows display
cv2.destroyAllWindows()

#       ---- CONCLUSION ----
# OpenCV (cv2) allows easy image loading and 
# processing.
# Grayscale conversion reduces complexity 
# (removes color information).
# This method is useful for face detection, 
# edge detection, and machine learning 
# applications.
