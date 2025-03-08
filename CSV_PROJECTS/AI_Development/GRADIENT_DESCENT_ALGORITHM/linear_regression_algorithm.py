
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# NumPy (np): Used for numerical operations like 
# generating random numbers and performing matrix 
# calculations. 
# matplotlib.pyplot (plt): Used to visualize the 
# data and the regression line
import numpy as np
import matplotlib.pyplot as plt

# Ensures the random numbers generated are the 
# same every time the script runs. Explain This Code more
np.random.seed( 13 )

# 100 random values between 0 and 2
x = 2 * np.random.rand( 100, 1 )

# Adds random noise to make the data more realistic
y = 4 + 3 * x + np.random.randn( 100, 1 )



# Gradient Descent Function
# theta_0: Represents the intercept (starting value = 0).
# theta_1: Represents the slope (starting value = 0).
# learning_rate: Controls how much the parameters update each iteration.
# num_iterations: Defines how many times gradient descent runs.
# m: Stores the number of data points (100 in this case).
def gradient_descent( x, y, learning_rate = 0.1, iterations = 1000 ):
    # Number of training examples
    m = len( y )
    # Random Initialization
    theta_0, theta_1 = np.random.randn( 2 )
    
    for i in range( iterations ):
        
        # Compute predictions
        # Computes y values based on current 
        # theta_0 and theta_1
        y_pred = theta_0 + theta_1 * x
        
        # Compute error
        # Measures how far predictions (y_pred) 
        # are from actual values (y)
        error = y_pred - y
        
        # Compute gradients
        # Derivative of the loss function w.r.t. theta_0
        gradient_0 = ( 2 / m ) * np.sum( error )
        # Gradient for slope
        # Derivative of the loss function w.r.t. theta_1
        gradient_1 = ( 2 / m ) * np.sum( error * x )
        
        # Update parameters using gradient descent
        # Updates theta_0 and theta_1 by moving in 
        # the negative gradient direction.
        # Update intercept
        theta_0 -= learning_rate * gradient_0
        # Update slope
        theta_1 -= learning_rate * gradient_1
        
    return theta_0, theta_1


# Train the model
theta_0, theta_1 = gradient_descent( x, y )

print( f"Final parameters: theta_0 = { theta_0 }, theta_1 = { theta_1 }" )

# Plot the results
# Plots original data points.
plt.scatter( x, y, label = "Data" )
# draws the regression line using the optimized theta_0 and theta_1
plt.plot( x, theta_0 + theta_1 * x, color = 'red', label = "Best Fit Line" )
# Label axis
plt.xlabel( "x" )
plt.ylabel( "y" )
# Adds a legend to the plot
plt.legend()
# Displays the final graph
plt.show()