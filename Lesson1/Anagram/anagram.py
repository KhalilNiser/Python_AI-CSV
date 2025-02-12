
# Function checks if the two Strings are anagram 
# or not (anagramCheck)
def anagramCheck( str1, str2 ):
    
    # heck the sorted Strings
    if( sorted( str1 ) == sorted( str2 ) ):
        
        print( "The Strings are Anagram!" )
        
    if( sorted( str1 ) == sorted( str2 ) ):
        
        print( "The Strings are Not Anagram!" )
        
    
    str1 = "listen"
    str2 = "silent"
    
    anagramCheck( str1, str2 )
        
        