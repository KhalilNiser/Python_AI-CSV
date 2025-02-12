
# Function reverse_sentence()
def reverse_sentence( sentence ):
    strWords = sentence.split()
    reversed_sentence = " ".join( reversed( strWords ) )
    return reversed_sentence
print( reverse_sentence( "Hello World!" ) )