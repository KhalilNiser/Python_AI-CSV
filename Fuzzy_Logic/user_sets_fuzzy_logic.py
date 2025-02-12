
# IMPORT_NUMPY_SKFUZZY
import numpy as np
import skfuzzy as fuzz

# Accept User Input as a decimal number
user_input = input( "Enter a Value Between 0 and 20!" )
user_input_value = float( user_input )

# Input Variables range between 0 and 20
X = np.arange( 0, 21, 1 )

# Define fuzzy sets for the input variable
# Triangular fuzzy sets for low values
low = fuzz.trimf( X, [ 0, 0, 5 ] )

# Triangular fuzzy sets for medium values
medium = fuzz.trimf( X, [ 2, 5, 8 ] )

# Triangular fuzzy sets for high values
high = fuzz.trimf( X, [ 5, 10, 10 ] )

# Triangle fuzzy sets for "custom" values
custom = fuzz.trimf( X, [ 3, 6, 9, ] )

# Get membership values for the user_input_values
low_degree = fuzz.interp_membership( X, low, user_input_value )
medium_degree = fuzz.interp_membership( X, medium, user_input_value )
high_degree = fuzz.interp_membership( X, high, user_input_value )
custom_degree = fuzz.interp_membership( X, custom, user_input_value )

# Define the fuzzy rules
# If, the input is low or medium, then the output is high
rule1 = np.fmax( low_degree, medium_degree )
# If, the input is medium or high, then the output is low
rule2 = np.fmin( medium_degree, high_degree )

# Apply the fuzzy rules by to determine the fuzzy 
# relation between input and output
relation = np.fmax( rule1, rule2 )

# Agregate the fuzzy relation using the maximum operator
aggregated = np.fmax( low, relation )
activated = np.fmin( aggregated, medium )

# Defuzzify the activated fuzy relation in order to obtain a
# crisp output 
# NOTE: CENTROID is a measure in defuzzification to calculate
# the center or the average of a fuzzy set
output = fuzz.defuzz( X, activated, 'centroid' )

# Display the output
print( "Output: ", output )

