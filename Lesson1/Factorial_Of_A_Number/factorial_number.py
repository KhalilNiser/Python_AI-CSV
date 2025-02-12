

# Function factorial
def factorial( n ):
    
    factor = 1
    # For-loop iterates through the said number "n"
    # NOTE: If I don't specify a range. It will 
    # by default, Satrt at 0 and end at 4
    for i in range( 1, n + 1 ):
        factor *= i
        
    return factor

x = 5

result = factorial( x )

print( result )