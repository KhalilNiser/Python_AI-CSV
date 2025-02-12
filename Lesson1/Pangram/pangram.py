
from string import ascii_lowercase

def is_pangram( input_string ):
    
    alphabet = set( ascii_lowercase )
    
    return set( input_string.lower() ) >= alphabet

print( is_pangram( "It's Better to Burnout, than fade away..." ) )