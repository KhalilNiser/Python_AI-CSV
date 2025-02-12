
# IMPORT_CLASS_COUNTER_FROM_COLLECTIONS:_PYTHON
from collections import Counter

# Function find_first_non_repeated
def find_first_non_repeated( charList ):
    
    charCounts = Counter( charList )
    
    # Iterates through the list of characters
    for char in charList:
        
        if charCounts[ char ] == 1:
            
            return char
        return None
    print( find_first_non_repeated( "abracadabra" ) )