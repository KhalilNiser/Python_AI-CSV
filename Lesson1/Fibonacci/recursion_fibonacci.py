#               ---- FIBONACCI_SEQUENCE_USING_RECURSION ----
# Sequence of numbers where a number is the sum 
# of the 2 numbers that came before. For example:
# The sequence: Frist digits are 0 and 1
# (0, ) 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...
# NOTE: The zero (0), sometimes is not mentioned

# Function fibonacci( n ), that takes in the number "n".
# NOTE: Here I'm using "n", as an "index".
def fibonacci( n ):
    
    if n == 0:
        return 0
    elif n == 1:
        return 1
    # return the sum of the two digits that came before
    else:
        return fibonacci( n - 2 ) + ( n - 1 )
    
for i in range( 13 ):
    
    print( fibonacci( i ) )