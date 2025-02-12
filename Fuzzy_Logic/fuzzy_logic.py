
# IMPORT_NUMPY_SKFUZZY
import numpy as np
import skfuzzy as fuzz

# Input variables
# Input range from 0 - 20 (Start = 0; Stop = 21; Step = 1)
# 0, 0 + 1 = 1 + 1 = 2 + 1 = 3 + 1 = 4, ...,20 
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


# Define fuzzy rules
# Rule1: If input is low or medium, then the output is high
rule1 = np.fmax( low, medium )
# Rule2: If the input is medium or high, then the ouput is low
rule2 = np.fmin( medium, high )


# Apply the fuzzy rules by to determine the fuzzy 
# relation between input and output
relation = np.fmax( rule1, rule2 )


# Defuzzify the relation in order to obtain a crsip output
# NOTE: CENTROID is a measure in defuzzification to calculate
# the center or the average of a fuzzy set
output = fuzz.defuzz( X, relation, 'centroid' )

# Display the crisp output
print( "Output: ", output)





