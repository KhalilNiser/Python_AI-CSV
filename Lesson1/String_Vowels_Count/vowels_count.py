
# Function count_vowels
def vowels_count( str ):
    
    vowels = "a e i o u"
    
    count = 0
    
    # For-loop iterates through String (Str)
    for char in str:
        
        if char.lower( ) in vowels:
            
            count += 1
            
            return count
        
        input_string = "Hello World!"
        
        result = vowels_count( input_string )
        
        str_result = f"Number of Vowels found in {input_string}: {result}"
        
        print( str_result )