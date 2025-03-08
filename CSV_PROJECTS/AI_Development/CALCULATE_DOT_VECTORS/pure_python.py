
#       ---- FUNCTION_(dot_product)_IMPLEMENTATION
#       ---- (MANUAL_CALCULATION) ----

# Pure Python (Manual Calculation) Implementation: 
# Pros: Simple, no dependecies are needed. 
# Cons: Slower for larger vectors (data) 
# NOTE: Optimal for smaller datasets. As it is not 
# intended for large data

# Takes two vectors (vector1 and vector2) as input.  
def dot_product( vector1, vector2 ):
    
    """
    Computes the dot product of two vectors using basic python.
    
    
    Parameters:
    
    vector1 (list): First vector
    vector2 (list): Second vector
    
    
    Returns:
    float: Dot product of vector1 and vector2
    
    """
    # Checks if the two vectors have the same length. 
    # If they don't, raise an error (value error). 
    # NOTE: As the "dot-product" can only be defined 
    # for vectors of the same size length
    if len(vector1) != len(vector2):
        raise ValueError( "Vectors must be of the same length!" )
    
    # Use "zip(vector1, vector2)" method, to iterate 
    # over both vectors simultaneously. Multiply 
    # Corresponding elements ( a * b ). Sum all the 
    # products using the sum() method. 
    # Example calculation: 
    # Step by step for v1 = [1, 2, 3] v2 = [4, 5, 6]: 
    # (1 * 4) + (2 * 5) + (3 * 6) = 4 + 10 + 18 = 32
    return sum(a * b for a, b in zip(vector1, vector2))

# Example usage:
v1 = [1, 2, 3]
v2 = [4, 5, 6]

result = dot_product(v1, v2) 
print( f"DOT Product (Pure Python): {result}" )