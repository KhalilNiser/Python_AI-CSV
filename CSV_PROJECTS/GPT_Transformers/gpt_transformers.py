
#               ---- IMPORT_REQUIRED_LIBRARIES ----
import os
os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 Model and Tokenizer 
# Use a smaller model variant
model_name = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained( model_name )
tokenizer = GPT2Tokenizer.from_pretrained( model_name )

# Set device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
model.to( device )

# Generate a response using GPT-2 
def generate_response( user_input ):
    input_ids = tokenizer.encode( user_input, return_tensors = "pt" ).to( device )
    output = model.generate( input_ids, max_length = 60, pad_token_id = tokenizer.eos_token_id )
    response = tokenizer.deocde( output[ 0 ], skip_special_tokens = True )
    return response

# Chat loop
while True:
    user_input = input( "Please. Enter Your Message: " )
    
    if user_input.lower() == 'exit':
        break
    
    response = generate_response( user_input )
    print( "ChatGPT: ", response )