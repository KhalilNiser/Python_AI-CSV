
#       ---- NumPy_IMPLEMENTATION_(OPTIMIZED) ----
#       ---- COMPARISON_SUMMARY ---- 
# Pros: Faster (Optimized for performance) 
# Cons: Requires NumPy 
# NOTE: In conclusion. If working with larger 
# datasets or machine learning. Use NumPy 
# for efficiency!

#       ---- IMPORT_REQUIRED_LIBRARIES ----
# NumPy: Provides optimized mathematical functions 
# for vector and matrix operations
import numpy as np  

# Function "dot_product_numpy": Accepts two 
# vectors (as list or NumPy arrays)
def dot_product_numpy(vector1, vector2):
    
    """
    
    Computes the dot product of two vectors using NumPy.
    
    
    Parameters:
    vector1 (list or np.array): First vector
    vector2 (list or np.array): Second vector
    
    Returns:
    
    float: Dot product of vector1 and vector2
    
    """
    
    # Convert list to NumPy array
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Use NumPy's built-in dot function. 
    # NOTE: NumPy's "np.dot()" method. Performs the 
    # same calculation as the pure python version 
    # but is highly optimized. Example Calculation: 
    # Same as pure_python 
    return np.dot(vector1, vector2)


# Example usage:
v1 = [1, 2, 3]
v2 = [4, 5, 6]

result = dot_product_numpy(v1, v2)

print( f"Dot Product (NumPy): {result}" )