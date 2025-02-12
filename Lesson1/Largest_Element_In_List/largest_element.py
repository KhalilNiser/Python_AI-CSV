

# Method 1: Iteration
# Function find_largest_number 
def findLargestNumberInArray( current_list ):
    
    print( current_list )
    
    largest = current_list[ 0 ]
    current_list = [ 1, 2, 9, 4, 5, 10, 33, 45, 0 ]
    # Iterate through the list of given numbers
    for num in current_list:
        
        # If "i" is greater than 
        if num > largest:
            
            largest = num
            
        return largest
    
    print( "Method 1: ", findLargestNumberInArray[ current_list ] )
    
    # Method 2: Sorting the Array
    sorting_list = sorted( current_list )
    
    method_2_largest = sorting_list[ -1 ]
    
    print( "Method 2: ", method_2_largest )
    