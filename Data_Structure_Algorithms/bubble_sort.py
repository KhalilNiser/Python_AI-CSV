

#  Function bubble_sort: takes an integer array (int_arr) as a parameter
def bubble_sort( int_arr ):
    # Outer-for-loop: Iterates through the list n times
    # Starting at index zero (0) all the way to (n-1)
    for n in range( len( int_arr ) -1, 0, -1 ):
        
        # Initialize swap function: If any swap occurs
        swap = False
        
        # Inner-for-loop: Compares adjacent elements
        for i in range( n ):
            
            if int_arr[ i ] > int_arr[ i + 1 ]:
                
                # Swap elements if in the wrong order
                int_arr[ i ], int_arr[ i + 1 ] = int_arr[ i + 1 ], int_arr[ i ]
                
                # Swap has been occured
                swap = True
                
                # If no swap occurs, means the list is sorted. Exit
                if not swap:
                    break
                
                # List to be sorted
                int_arr = [ 39, 12, 18, 85, 72, 10, 2, 18 ]
                
                # Print the Unsorted List
                print( "Unsorted List: " )
                print( int_arr )
                
                # Calling on the bubble_sort function
                bubble_sort( int_arr )
                
                print( "Sorted List:" )
                print( int_arr )