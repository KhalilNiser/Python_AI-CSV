
# Assign X to equal zero (0)
X = 0

# Learning rate/step size/gradient
learning_rate = 0.1

# Assigning the number of iterations
number_of_iterations = 100

# Function minimize_function: This function is to minimize 
# (That is what they say, at the least). The only thing I 
# can sense out of looking at this code is: X squared i. e 
# (Math.pow( X, 2 ) ) multiplied by 2, plus 5, times X, plus 6
def minimize_function( X ):
    return X ** 2 + 5 * X + 6

# Define the derivative of the function
def define_derivative( X ):
    return 2 * X + 5

# Peforming gradient descent iterations
# _: This is a convention in Python for a variable that you don't 
# plan to use within the loop. It signifies that you only care 
# about the number of iterations, not the actual value of the 
# loop variable.
for _ in range( number_of_iterations ):
    gradient = define_derivative( X )
    X =- learning_rate * gradient
    
# Print/Display the result
print( "The minimum value found at X =", X )



               
